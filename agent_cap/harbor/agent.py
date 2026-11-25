"""CAPAgent: run an AgentCAP strategy as a Harbor agent.

Harbor loads this class by import path. Either give every role explicitly::

    agents:
      - import_path: agent_cap.harbor.agent:CAPAgent
        kwargs:
          strategy: single            # any registered AgentCAP strategy
          tool_backend: math-python   # none | shell | math-python
          roles:
            agent: {name: gpt-oss, base_url: http://127.0.0.1:8000, protocol: harmony}

or use Harbor's standard model connection, as for any other Harbor agent; every
role of the strategy then uses that one endpoint::

    harbor run --agent agent_cap.harbor:CAPAgent --model Qwen/Qwen3.5-4B \
        --base-url http://127.0.0.1:30000/v1 --ak strategy=plan-execute ...

The strategy loop and every LLM request run on the host, exactly as in
``python -m agent_cap.agents``; tool calls execute inside the Harbor task
environment through ``environment.exec`` (see ``agent_cap.harbor.tools``).

Each trial writes ``<trial>/agent/capagent_result.json``: the same per-task row
the AgentCAP CLI writes to ``results.jsonl`` (usage per role, ``turn_stats``
with per-request TTFT/decode time/tokens). ``agent_cap.harbor.export_teas``
joins these rows with Harbor's verifier rewards into the TEAS files.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import aiohttp
from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.model_api_format import ModelApiFormat

import agent_cap
from agent_cap.agents.cli import (
    _build_agents,
    _expand_env,
    _partial_failure_result,
    _serialize_result,
)
from agent_cap.agents.registry import get_strategy
from agent_cap.agents.types import AgentSpec, Task
from agent_cap.harbor import RESULT_FILENAME
from agent_cap.harbor.connection import connection_spec, resolve_endpoint
from agent_cap.harbor.tools import HarborTools

# Roles each built-in strategy reads; used when the team comes from --model.
DEFAULT_ROLES = {
    "single": ["agent"],
    "plan-execute": ["planner", "executor"],
    "supervisor": ["supervisor", "worker"],
}


class CAPAgent(BaseAgent):
    """Runs an AgentCAP strategy on the host; tool calls run in the Harbor environment."""

    # Harbor's --model/--base-url/--api-key-name; AgentCAP clients speak chat completions.
    capabilities = AgentCapabilities(base_url=True)
    MODEL_CONNECTION = connection_spec(ModelApiFormat.OPENAI_CHAT_COMPLETIONS)

    def __init__(
        self,
        logs_dir: Path,
        model_name: Optional[str] = None,
        *,
        strategy: str = "single",
        roles: Optional[Mapping[str, Mapping[str, Any]]] = None,
        role_names: Optional[list] = None,
        role_options: Optional[Mapping[str, Any]] = None,
        tool_backend: str = "none",
        max_turns: int = 20,
        system_prompt: Optional[str] = None,
        answer_file: Optional[str] = None,
        workdir: str = "/workspace",
        task_timeout_s: Optional[float] = None,
        sequence: Optional[list] = None,
        tool_options: Optional[Mapping[str, Any]] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, model_name, **kwargs)
        if not roles and not model_name:
            raise ValueError(
                "CAPAgent needs a model: Harbor's --model (with --base-url/--api-key-name), "
                "or `roles` in its kwargs, e.g. {agent: {name: ..., base_url: ..., api_key: ...}}"
            )
        self.strategy_name = strategy
        self._roles = {str(r): dict(d) for r, d in (roles or {}).items()}
        self.role_names = [str(r) for r in role_names] if role_names else None
        self.role_options = dict(role_options or {})
        self._specs: Optional[Dict[str, AgentSpec]] = None
        self.tool_backend = tool_backend
        self.max_turns = int(max_turns)
        self.system_prompt = system_prompt or ""
        self.answer_file = answer_file
        self.workdir = workdir
        self.task_timeout_s = float(task_timeout_s) if task_timeout_s else None
        self.sequence = list(sequence) if sequence else None
        self.tool_options = dict(tool_options or {})
        self.task_id = task_id
        self._tools: Optional[HarborTools] = None

    @staticmethod
    def name() -> str:
        return "capagent"

    @property
    def specs(self) -> Dict[str, AgentSpec]:
        """Role -> AgentSpec: the `roles` kwarg, else every strategy role on Harbor's connection."""
        if self._specs is None:
            if self._roles:
                roles = self._roles
            else:
                endpoint = resolve_endpoint(self)
                names = self.role_names or self._default_role_names()
                shared = {
                    "use_streaming": True,  # TTFT and decode time need a stream (as in configs/agents_*.yaml)
                    **self.role_options,
                    "name": endpoint.model,
                    "base_url": endpoint.base_url or "http://localhost:30000/v1",
                    "api_key": endpoint.api_key or "EMPTY",
                }
                roles = {name: dict(shared) for name in names}
            self._specs = {
                role: AgentSpec.from_dict(role, _expand_env(dict(data))) for role, data in roles.items()
            }
        return self._specs

    def _default_role_names(self) -> list:
        if self.strategy_name == "sequential" and self.sequence:
            return list(dict.fromkeys(str(r) for r in self.sequence))
        return list(DEFAULT_ROLES.get(self.strategy_name, ["agent"]))

    def version(self) -> Optional[str]:
        return agent_cap.__version__

    async def setup(self, environment: BaseEnvironment) -> None:
        self._tools = HarborTools(
            environment, self.tool_backend, workdir=self.workdir, **self.tool_options
        )
        await self._tools.prepare()

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._tools is None or self._tools.environment is not environment:
            await self.setup(environment)
        tools = self._tools
        strategy = self._make_strategy()
        task = Task(task_id=self.task_id or _trial_task_id(self.logs_dir), user_prompt=instruction)

        connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
        timeout = aiohttp.ClientTimeout(total=1800, connect=60, sock_connect=60, sock_read=1800)
        started = time.perf_counter()
        agents: Dict[str, Any] = {}
        run_res = None
        error: Optional[BaseException] = None
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            agents = _build_agents(self.specs, None, tools, session=session)
            self._apply_system_prompt(agents)
            try:
                coro = strategy.run(task, agents, tools)
                run_res = (
                    await asyncio.wait_for(coro, timeout=self.task_timeout_s)
                    if self.task_timeout_s
                    else await coro
                )
            except BaseException as exc:  # includes Harbor's cancellation on agent timeout
                error = exc
                run_res = _partial_failure_result(
                    task=task,
                    strategy_name=self.strategy_name,
                    agents=agents,
                    elapsed_s=time.perf_counter() - started,
                    exc=exc if isinstance(exc, Exception) else TimeoutError("agent cancelled"),
                    timeout_s=self.task_timeout_s,
                )
            finally:
                row = _serialize_result(run_res, verbose=0) if run_res is not None else {}
                row.update(
                    {
                        "task_id": task.task_id,
                        "agentcard": {
                            "agent": self.name(),
                            "agentcap_version": self.version(),
                            "strategy": self.strategy_name,
                            "tool_backend": self.tool_backend,
                            "roles": {r: s.endpoint.name for r, s in self.specs.items()},
                        },
                    }
                )
                self._write_row(row)
                self._fill_context(context, row)

        if error is not None:
            raise error
        if self.answer_file and run_res is not None:
            await tools.write_file(self.answer_file, run_res.output_text or "")

    # ------------------------------------------------------------------ #

    def _make_strategy(self):
        cls = get_strategy(self.strategy_name)
        strategy = cls(sequence=self.sequence) if self.strategy_name == "sequential" else cls()
        strategy.max_turns = self.max_turns
        return strategy

    def _apply_system_prompt(self, agents: Dict[str, Any]) -> None:
        # Same placement as the AgentCAP CLI's per-task system prompt.
        if not self.system_prompt:
            return
        for ag in agents.values():
            if not ag.state.messages or ag.state.messages[0].get("role") != "system":
                ag.state.messages.insert(0, {"role": "system", "content": self.system_prompt})
            else:
                ag.state.messages[0] = {"role": "system", "content": self.system_prompt}

    def _write_row(self, row: Dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / RESULT_FILENAME).write_text(
            json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @staticmethod
    def _fill_context(context: AgentContext, row: Dict[str, Any]) -> None:
        usage = row.get("total_usage") or {}
        context.n_input_tokens = int(usage.get("input_tokens") or 0)
        context.n_output_tokens = int(usage.get("output_tokens") or 0)
        context.n_cache_tokens = int(usage.get("cached_tokens") or 0)
        context.metadata = {
            "per_role_usage": row.get("per_role_usage") or {},
            "num_requests": int(usage.get("requests") or 0),
            "errors": row.get("errors") or [],
            "capagent_result": RESULT_FILENAME,
        }


def _trial_task_id(logs_dir: Path) -> str:
    # <jobs>/<job>/<trial>/agent -> <trial>; Harbor names trials "<task>__<suffix>".
    trial = logs_dir.parent.name if logs_dir.name == "agent" else logs_dir.name
    return trial.rsplit("__", 1)[0] if "__" in trial else trial


__all__ = ["CAPAgent", "DEFAULT_ROLES", "RESULT_FILENAME"]
