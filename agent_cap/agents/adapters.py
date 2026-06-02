"""Adapters that expose existing AgentCAP backends/evaluators to the
agents module without depending on them at import time."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import aiohttp

from agent_cap.agents.tools import ToolProvider


class MCPProviderAdapter:
    """Wraps `agent_cap.runner.tool_backends.MCPToolBackend` as a ToolProvider.

    `set_task_allowlist` filters `list_tools()` output per task to match
    official mcp-atlas behavior (see sandbox_client.py L44-47 in scaleapi/mcp-atlas).
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        mcp_server_url: str,
        enabled_tools: Optional[List[str]] = None,
    ) -> None:
        from agent_cap.runner.tool_backends import MCPToolBackend

        self._backend = MCPToolBackend(
            session=session,
            mcp_server_url=mcp_server_url,
            enabled_tools=enabled_tools or [],
        )
        self._setup_done = False
        self._task_allowlist: Optional[set] = None

    def set_task_allowlist(self, enabled_tools: Optional[List[Any]]) -> None:
        if not enabled_tools:
            self._task_allowlist = None
            return
        self._task_allowlist = {
            t if isinstance(t, str) else (t.get("name", "") if isinstance(t, dict) else "")
            for t in enabled_tools
        }

    async def _ensure_setup(self) -> None:
        if not self._setup_done:
            ok = await self._backend.setup({})
            if not ok:
                raise RuntimeError("MCPToolBackend setup failed")
            self._setup_done = True

    async def list_tools(self) -> List[Dict[str, Any]]:
        await self._ensure_setup()
        tools = await self._backend.list_tools()
        if self._task_allowlist is not None:
            tools = [
                t for t in tools
                if t.get("function", {}).get("name", "") in self._task_allowlist
            ]
        return tools

    async def call(self, name: str, arguments: Dict[str, Any]) -> str:
        await self._ensure_setup()
        try:
            result = await self._backend.call_tool(name, arguments)
        except Exception as exc:
            return f"ERROR: tool '{name}' failed: {exc}"
        if isinstance(result, list):
            parts = []
            for blk in result:
                if isinstance(blk, dict):
                    parts.append(str(blk.get("text") or blk.get("content") or blk))
                else:
                    parts.append(str(blk))
            return "\n".join(parts)
        return str(result)

    async def teardown(self) -> None:
        try:
            await self._backend.teardown()
        except Exception:
            pass


def load_dataset_as_tasks(name: str, num_tasks: int = 0) -> List[Dict[str, Any]]:
    """Use unified_runner's dataset loader; convert UnifiedTask to plain dicts
    that `agents.Task.from_dict` can consume."""
    from agent_cap.runner.unified_runner import _load_dataset_tasks

    raw = _load_dataset_tasks(name, int(num_tasks or 0))
    out: List[Dict[str, Any]] = []
    for t in raw:
        msgs = list(getattr(t, "messages", []) or [])
        prompt = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                prompt = str(m.get("content", ""))
                break
        system_prompt = ""
        for m in msgs:
            if m.get("role") == "system":
                system_prompt = str(m.get("content", ""))
                break
        entry: Dict[str, Any] = {
            "task_id": getattr(t, "task_id", "") or f"task-{len(out)}",
            "user_prompt": prompt,
            "_unified_task": t,
        }
        if system_prompt:
            entry["system_prompt"] = system_prompt
        out.append(entry)
    return out


class MathPythonProviderAdapter:
    """Wraps `agent_cap.backends.math_python_backend.MathPythonBackend` as a ToolProvider.

    Exposes one tool named `python` that executes Python code in a persistent
    Jupyter kernel sandbox (same backend used by `run_imo_answerbench_*.py`).
    """

    def __init__(
        self,
        startup_timeout: float = 30.0,
        exec_timeout: float = 30.0,
        preload: str = "minimal",
        auto_print_last_expr: bool = True,
    ) -> None:
        from agent_cap.backends.math_python_backend import (
            MathPythonBackend,
            PYTHON_TOOL_DEFINITIONS,
        )

        self._backend = MathPythonBackend(
            startup_timeout=startup_timeout,
            exec_timeout=exec_timeout,
            preload=preload,
            auto_print_last_expr=auto_print_last_expr,
        )
        self._tool_defs = PYTHON_TOOL_DEFINITIONS
        self._setup_done = False

    def _ensure_setup(self) -> None:
        if not self._setup_done:
            ok = self._backend.setup({})
            if not ok:
                raise RuntimeError("MathPythonBackend setup failed")
            self._setup_done = True

    async def list_tools(self) -> List[Dict[str, Any]]:
        self._ensure_setup()
        return list(self._tool_defs)

    async def call(self, name: str, arguments: Dict[str, Any]) -> str:
        self._ensure_setup()
        import asyncio
        import uuid

        call_id = f"call_{uuid.uuid4().hex[:8]}"
        result = await asyncio.to_thread(
            self._backend.execute, name, call_id, arguments
        )
        return getattr(result, "output", str(result))

    async def teardown(self) -> None:
        try:
            self._backend.teardown()
        except Exception:
            pass


__all__ = [
    "MCPProviderAdapter",
    "MathPythonProviderAdapter",
    "load_dataset_as_tasks",
]
