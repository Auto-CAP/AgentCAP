"""Agent runtime: drives one role through tool-using turns."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_cap.agents.llm import LLMClient
from agent_cap.agents.tools import ToolProvider
from agent_cap.agents.types import AgentSpec, TurnRecord, Usage


@dataclass
class AgentState:
    """Mutable per-agent state across one task."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    turns: List[TurnRecord] = field(default_factory=list)


class Agent:
    """One role with its own message history, prompted by a single endpoint."""

    def __init__(
        self,
        spec: AgentSpec,
        llm: LLMClient,
        tools: Optional[ToolProvider] = None,
    ) -> None:
        self.spec = spec
        self.llm = llm
        self.tools = tools if spec.can_call_tools else None
        self.state = AgentState()
        if spec.system_prompt:
            self.state.messages.append({"role": "system", "content": spec.system_prompt})

    @property
    def role(self) -> str:
        return self.spec.role

    def reset(self) -> None:
        self.state = AgentState()
        if self.spec.system_prompt:
            self.state.messages.append({"role": "system", "content": self.spec.system_prompt})

    def add_user(self, text: str) -> None:
        self.state.messages.append({"role": "user", "content": text})

    async def step(
        self,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> TurnRecord:
        """Make one LLM call. Does not auto-execute tool calls (see `run`)."""
        msgs_in = list(self.state.messages)
        reply = await self.llm.chat(self.spec.endpoint, msgs_in, tool_schemas)
        tool_calls = list(reply.assistant.get("tool_calls") or [])
        if not tool_calls:
            from agent_cap.runner.unified_runner import _recover_tool_calls_from_content

            content = reply.assistant.get("content") or ""
            recovered = _recover_tool_calls_from_content(content, len(self.state.turns))
            if recovered:
                tool_calls = recovered
                reply.assistant["tool_calls"] = recovered
        self.state.messages.append(reply.assistant)
        self.state.usage.add(reply.usage)
        record = TurnRecord(
            role=self.role,
            model=self.spec.endpoint.name,
            messages_in=msgs_in,
            assistant=reply.assistant,
            tool_calls=tool_calls,
            usage=reply.usage,
            latency_s=reply.latency_s,
            ttft_s=getattr(reply, "ttft_s", 0.0) or 0.0,
            decode_s=getattr(reply, "decode_s", 0.0) or 0.0,
        )
        self.state.turns.append(record)
        return record

    async def run(self, user_prompt: str, max_turns: int = 8) -> List[TurnRecord]:
        """Drive a full tool-use loop until the model emits no tool calls."""
        self.add_user(user_prompt)
        tool_schemas = await self.tools.list_tools() if self.tools else None
        for _ in range(max_turns):
            record = await self.step(tool_schemas=tool_schemas)
            if not record.tool_calls:
                return self.state.turns
            await self._execute_tool_calls(record)
        return self.state.turns

    async def _execute_tool_calls(self, record: TurnRecord) -> None:
        if not self.tools:
            return
        for tc in record.tool_calls:
            fn = (tc.get("function") or {})
            name = str(fn.get("name", ""))
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            t0 = time.perf_counter()
            result = await self.tools.call(name, args)
            latency = time.perf_counter() - t0
            self.state.messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })
            record.tool_results.append({
                "tool_call_id": tc.get("id", ""),
                "name": name,
                "arguments": args,
                "result": result,
                "latency_s": round(latency, 4),
            })

    def final_text(self) -> str:
        for msg in reversed(self.state.messages):
            if msg.get("role") != "assistant":
                continue
            c = msg.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, list):
                parts = [b.get("text", "") for b in c if isinstance(b, dict)]
                joined = "\n".join(p for p in parts if p)
                if joined.strip():
                    return joined.strip()
            r = msg.get("reasoning_content") or msg.get("reasoning")
            if isinstance(r, str) and r.strip():
                return r.strip()
        return ""


__all__ = ["Agent", "AgentState"]
