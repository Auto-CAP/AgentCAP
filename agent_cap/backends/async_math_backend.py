from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from agent_cap.backends.math_python_backend import MathPythonBackend
from agent_cap.runner.tool_backends import ToolBackend


class AsyncMathPythonBackend(ToolBackend):
    def __init__(
        self,
        startup_timeout: float = 30.0,
        exec_timeout: float = 5.0,
        preload: str = "minimal",
        auto_print_last_expr: bool = True,
    ):
        self._backend = MathPythonBackend(
            startup_timeout=startup_timeout,
            exec_timeout=exec_timeout,
            preload=preload,
            auto_print_last_expr=auto_print_last_expr,
        )

    async def setup(self, task_config: Dict[str, Any]) -> bool:
        return await asyncio.to_thread(self._backend.setup, task_config)

    async def list_tools(self) -> List[Dict[str, Any]]:
        return self._backend.get_tool_definitions()

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        result = await asyncio.to_thread(self._backend.execute, name, "call", arguments)
        if result.success:
            return [{"type": "text", "text": result.output}]
        raise RuntimeError(result.output)

    async def teardown(self) -> None:
        await asyncio.to_thread(self._backend.teardown)
