from __future__ import annotations

import json
import importlib
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from agent_cap.core.evaluator import EvalResult
from agent_cap.runner.llm_client import _chat_with_fallback
from agent_cap.runner.tool_backends import ToolBackend

_DEFAULT_USER_MODEL = "openai/gpt-5.1"
_DEFAULT_USER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_USER_PROVIDER = ""


def _ensure_tau2_importable(repo_root: Optional[Path] = None) -> Path:
    base_root = repo_root or Path(__file__).resolve().parents[2]
    tau2_src = (base_root / "third_party" / "tau2-bench" / "src").resolve()
    if not tau2_src.exists():
        raise FileNotFoundError(
            f"tau2-bench src not found at {tau2_src}. "
            "Clone tau2-bench into third_party/tau2-bench first."
        )
    tau2_src_str = str(tau2_src)
    if tau2_src_str not in sys.path:
        sys.path.insert(0, tau2_src_str)
    return tau2_src


_TAU2_SRC = _ensure_tau2_importable()

_TAU2_CACHE: Optional[Dict[str, Any]] = None


def _tau2() -> Dict[str, Any]:
    global _TAU2_CACHE
    if _TAU2_CACHE is None:
        message_mod = importlib.import_module("tau2.data_model.message")
        simulation_mod = importlib.import_module("tau2.data_model.simulation")
        tasks_mod = importlib.import_module("tau2.data_model.tasks")
        env_mod = importlib.import_module("tau2.domains.banking_knowledge.environment")
        evaluator_mod = importlib.import_module("tau2.evaluator.evaluator")
        modes_mod = importlib.import_module("tau2.orchestrator.modes")
        _TAU2_CACHE = {
            "AssistantMessage": message_mod.AssistantMessage,
            "ToolCall": message_mod.ToolCall,
            "ToolMessage": message_mod.ToolMessage,
            "UserMessage": message_mod.UserMessage,
            "SimulationRun": simulation_mod.SimulationRun,
            "TerminationReason": simulation_mod.TerminationReason,
            "Task": tasks_mod.Task,
            "get_db": env_mod.get_db,
            "get_environment": env_mod.get_environment,
            "EvaluationType": evaluator_mod.EvaluationType,
            "evaluate_simulation": evaluator_mod.evaluate_simulation,
            "CommunicationMode": modes_mod.CommunicationMode,
        }
    return _TAU2_CACHE


class Tau2BankingAdapter(ToolBackend):
    """Adapter that wraps tau2 banking_knowledge env for TeamRunner.

    It exposes tools in OpenAI function schema format and can run a user
    simulator LLM as the customer.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        repo_root: Optional[Path] = None,
        retrieval_variant: str = "bm25",
        user_model: str = _DEFAULT_USER_MODEL,
        user_base_url: str = _DEFAULT_USER_BASE_URL,
        user_api_key: str = "",
        user_temperature: float = 0.0,
        user_max_tokens: int = 1024,
        user_openrouter_provider: str = _DEFAULT_USER_PROVIDER,
        user_use_streaming: bool = False,
    ):
        self._session = session
        self._repo_root = repo_root or Path(__file__).resolve().parents[2]
        self._retrieval_variant = retrieval_variant

        self._user_model = user_model
        self._user_base_url = user_base_url
        self._user_api_key = user_api_key
        self._user_temperature = user_temperature
        self._user_max_tokens = user_max_tokens
        self._user_openrouter_provider = user_openrouter_provider
        self._user_use_streaming = user_use_streaming

        self._env = None
        self._task: Optional[Any] = None
        self._active_retrieval_variant = retrieval_variant
        self._tool_schemas: List[Dict[str, Any]] = []
        self._tool_map: Dict[str, Any] = {}
        self._messages: List[Any] = []
        self._user_system_prompt: str = ""
        self._user_done = False
        self._transfer_requested = False
        self._tool_counter = 0

    @property
    def task(self) -> Optional[Any]:
        return self._task

    @property
    def transfer_requested(self) -> bool:
        return self._transfer_requested

    @property
    def user_done(self) -> bool:
        return self._user_done

    @property
    def db_snapshot(self) -> Dict[str, Any]:
        if self._env is None or self._env.tools is None or self._env.tools.db is None:
            return {}
        try:
            return self._env.tools.db.model_dump()
        except Exception:
            return {}

    @property
    def conversation_messages(self) -> List[Any]:
        return list(self._messages)

    @property
    def agent_policy(self) -> str:
        if self._env is not None:
            return str(self._env.policy)
        return ""

    async def setup(self, task_config: Dict[str, Any]) -> bool:
        raw_task = (
            task_config.get("tau2_task") if isinstance(task_config, dict) else None
        )
        if not isinstance(raw_task, dict):
            return False

        try:
            task_cls = _tau2()["Task"]
            self._task = task_cls.model_validate(raw_task)
        except Exception:
            return False

        self._tool_schemas = []
        self._tool_map = {}
        self._messages = []
        self._user_done = False
        self._transfer_requested = False
        self._tool_counter = 0

        try:
            self._active_retrieval_variant = str(
                task_config.get("retrieval_variant", self._retrieval_variant)
            )
            get_db = _tau2()["get_db"]
            get_environment = _tau2()["get_environment"]

            db = get_db()
            self._env = get_environment(
                db=db,
                retrieval_variant=self._active_retrieval_variant,
                task=self._task,
                solo_mode=False,
            )
            initial_state = self._task.initial_state if self._task is not None else None
            self._env.set_state(
                initialization_data=(
                    initial_state.initialization_data
                    if initial_state is not None
                    else None
                ),
                initialization_actions=(
                    initial_state.initialization_actions
                    if initial_state is not None
                    else None
                ),
                message_history=(
                    list(initial_state.message_history)
                    if initial_state is not None
                    and initial_state.message_history is not None
                    else []
                ),
            )
            self._env.sync_tools()
        except Exception:
            self._env = None
            return False

        if self._env is None:
            return False

        # Build assistant tool list for team_runner executor
        for tool in self._env.get_tools():
            schema = tool.openai_schema
            fn = schema.get("function", {}) if isinstance(schema, dict) else {}
            name = str(fn.get("name", ""))
            if not name:
                continue
            self._tool_map[name] = tool
            self._tool_schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(fn.get("description", "")),
                        "parameters": fn.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )

        persona = ""
        instructions = ""
        if (
            self._task is not None
            and getattr(self._task, "user_scenario", None) is not None
        ):
            persona = self._task.user_scenario.persona or ""
            instructions = str(self._task.user_scenario.instructions)
        self._user_system_prompt = (
            "You are simulating the customer in a banking customer-support conversation. "
            "Stay strictly in character, follow the task instructions exactly, and keep responses concise. "
            "When the customer should end the conversation, output exactly '###STOP###'.\n\n"
            f"Persona:\n{persona}\n\nInstructions:\n{instructions}"
        )
        return True

    async def list_tools(self) -> List[Dict[str, Any]]:
        return list(self._tool_schemas)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if self._env is None:
            raise RuntimeError("Tau2 banking environment not initialized")
        if name not in self._tool_map:
            raise RuntimeError(f"Unknown tau2 tool: {name}")

        tau2_types = _tau2()
        ToolCall = tau2_types["ToolCall"]
        AssistantMessage = tau2_types["AssistantMessage"]

        tool_call = ToolCall(
            id=f"ac_tool_{self._tool_counter}",
            name=name,
            arguments=dict(arguments or {}),
            requestor="assistant",
        )
        self._tool_counter += 1
        tool_msg = self._env.get_response(tool_call)
        self._messages.append(
            AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])
        )
        self._messages.append(tool_msg)

        if name == "transfer_to_human_agents":
            self._transfer_requested = True

        return [{"type": "text", "text": tool_msg.content or ""}]

    async def start_user_turn(self) -> str:
        """Get the initial user message for the task conversation."""
        return await self._generate_user_reply(last_agent_text=None)

    async def next_user_turn(self, agent_text: str) -> str:
        """Generate the next user message based on latest agent response."""
        return await self._generate_user_reply(last_agent_text=agent_text)

    async def _generate_user_reply(self, last_agent_text: Optional[str]) -> str:
        if self._user_done:
            return "###STOP###"

        tau2_types = _tau2()
        UserMessage = tau2_types["UserMessage"]
        AssistantMessage = tau2_types["AssistantMessage"]
        ToolMessage = tau2_types["ToolMessage"]
        ToolCall = tau2_types["ToolCall"]

        chat_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._user_system_prompt}
        ]

        # Reconstruct conversation for the user simulator LLM
        for msg in self._messages:
            if isinstance(msg, UserMessage) and msg.content is not None:
                chat_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                if msg.content:
                    chat_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                # Tool outcomes are seen as assistant-side context
                if msg.content is not None:
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": f"[tool result] {msg.content}",
                        }
                    )

        if last_agent_text is not None and last_agent_text.strip():
            chat_messages.append(
                {"role": "assistant", "content": last_agent_text.strip()}
            )

        # Enable user tool calls for realism
        user_tools = []
        if self._env is not None and self._env.user_tools is not None:
            include = self._task.user_tools if self._task is not None else None
            for tool in self._env.get_user_tools(include=include):
                schema = tool.openai_schema
                fn = schema.get("function", {}) if isinstance(schema, dict) else {}
                user_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": str(fn.get("name", "")),
                            "description": str(fn.get("description", "")),
                            "parameters": fn.get(
                                "parameters", {"type": "object", "properties": {}}
                            ),
                        },
                    }
                )

        errors: List[str] = []
        timed = await _chat_with_fallback(
            session=self._session,
            base_url=self._user_base_url,
            api_key=self._user_api_key,
            model=self._user_model,
            messages=chat_messages,
            tools=user_tools if user_tools else None,
            max_tokens=self._user_max_tokens,
            temperature=self._user_temperature,
            openrouter_provider=self._user_openrouter_provider,
            use_streaming=self._user_use_streaming,
            errors=errors,
        )
        result = timed.response_json
        choices = result.get("choices") or []
        if not choices:
            self._user_done = True
            reply = "###STOP###"
            self._messages.append(UserMessage(role="user", content=reply))
            return reply

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            pieces = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
            content_text = "\n".join(pieces).strip()
        else:
            content_text = str(content or "").strip()

        # If user model calls tools, execute user-side tools then ask for final utterance
        tool_calls = message.get("tool_calls") or []
        if tool_calls and self._env is not None:
            for tc in tool_calls:
                fn = tc.get("function") or {}
                tool_name = str(fn.get("name", ""))
                raw_args = fn.get("arguments", "{}")
                try:
                    parsed_args = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except json.JSONDecodeError:
                    parsed_args = {}

                user_tool_call = ToolCall(
                    id=str(tc.get("id") or f"user_tool_{uuid.uuid4().hex[:8]}"),
                    name=tool_name,
                    arguments=parsed_args if isinstance(parsed_args, dict) else {},
                    requestor="user",
                )
                self._messages.append(
                    UserMessage(role="user", content=None, tool_calls=[user_tool_call])
                )
                tool_msg = self._env.get_response(user_tool_call)
                self._messages.append(tool_msg)

            # Ask for textual follow-up after tool execution
            followup_messages = list(chat_messages)
            for msg in self._messages[-2 * len(tool_calls) :]:
                if isinstance(msg, ToolMessage):
                    followup_messages.append(
                        {
                            "role": "assistant",
                            "content": f"[user tool result] {msg.content}",
                        }
                    )
            followup_messages.append(
                {
                    "role": "system",
                    "content": "Now provide the customer's next short utterance only. "
                    "If conversation is done, output exactly ###STOP###.",
                }
            )
            timed2 = await _chat_with_fallback(
                session=self._session,
                base_url=self._user_base_url,
                api_key=self._user_api_key,
                model=self._user_model,
                messages=followup_messages,
                tools=None,
                max_tokens=self._user_max_tokens,
                temperature=self._user_temperature,
                openrouter_provider=self._user_openrouter_provider,
                use_streaming=self._user_use_streaming,
                errors=errors,
            )
            result2 = timed2.response_json
            choices2 = result2.get("choices") or []
            if choices2:
                content2 = choices2[0].get("message", {}).get("content", "")
                content_text = str(content2 or "").strip()

        if not content_text:
            content_text = "###STOP###"

        self._messages.append(UserMessage(role="user", content=content_text))
        if "###STOP###" in content_text:
            self._user_done = True
        return content_text

    def append_agent_text(self, text: str) -> None:
        AssistantMessage = _tau2()["AssistantMessage"]
        self._messages.append(AssistantMessage(role="assistant", content=text or ""))

    def build_tau2_simulation(
        self,
        *,
        simulation_id: str,
        task_id: str,
        duration_s: float,
        termination_reason: Any,
    ) -> Any:
        tau2_types = _tau2()
        SimulationRun = tau2_types["SimulationRun"]
        CommunicationMode = tau2_types["CommunicationMode"]
        return SimulationRun(
            id=simulation_id,
            task_id=task_id,
            start_time="",
            end_time="",
            duration=duration_s,
            termination_reason=termination_reason,
            reward_info=None,
            agent_cost=None,
            user_cost=None,
            messages=list(self._messages),
            mode=CommunicationMode.HALF_DUPLEX.value,
        )

    def evaluate(self, simulation: Any) -> EvalResult:
        if self._task is None:
            return EvalResult(
                passed=False,
                score=0.0,
                details={"error": "missing_task"},
            )

        tau2_types = _tau2()
        evaluate_simulation = tau2_types["evaluate_simulation"]
        EvaluationType = tau2_types["EvaluationType"]
        CommunicationMode = tau2_types["CommunicationMode"]

        reward_info = evaluate_simulation(
            simulation=simulation,
            task=self._task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain="banking_knowledge",
            mode=CommunicationMode.HALF_DUPLEX,
            env_kwargs={"retrieval_variant": self._active_retrieval_variant},
        )
        score = float(reward_info.reward or 0.0)
        return EvalResult(
            passed=score >= 1.0,
            score=score,
            details=reward_info.model_dump(),
        )

    def evaluate_conversation(
        self,
        *,
        task_id: str,
        simulation_id: str,
        duration_s: float,
        termination_reason: str,
    ) -> EvalResult:
        TerminationReason = _tau2()["TerminationReason"]
        reason_map = {
            "agent_stop": TerminationReason.AGENT_STOP,
            "user_stop": TerminationReason.USER_STOP,
            "max_steps": TerminationReason.MAX_STEPS,
            "timeout": TerminationReason.TIMEOUT,
        }
        reason = reason_map.get(termination_reason, TerminationReason.AGENT_STOP)
        simulation = self.build_tau2_simulation(
            simulation_id=simulation_id,
            task_id=task_id,
            duration_s=duration_s,
            termination_reason=reason,
        )
        return self.evaluate(simulation)

    async def teardown(self) -> None:
        self._env = None
        self._task = None
        self._tool_schemas = []
        self._tool_map = {}
        self._messages = []
        self._user_system_prompt = ""
        self._user_done = False
        self._transfer_requested = False

    async def get_patch(self) -> str:
        return ""


def make_default_tau2_adapter(session: aiohttp.ClientSession) -> Tau2BankingAdapter:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    return Tau2BankingAdapter(
        session=session,
        user_model=os.environ.get("TAU2_USER_MODEL", _DEFAULT_USER_MODEL),
        user_base_url=os.environ.get("TAU2_USER_BASE_URL", _DEFAULT_USER_BASE_URL),
        user_api_key=api_key,
        user_temperature=0.0,
        user_max_tokens=int(os.environ.get("TAU2_USER_MAX_TOKENS", "1024") or 1024),
        user_openrouter_provider=os.environ.get(
            "TAU2_USER_OPENROUTER_PROVIDER", _DEFAULT_USER_PROVIDER
        ),
        user_use_streaming=False,
    )
