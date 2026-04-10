import abc
from typing import Any, Dict, List, Optional, Sequence

import aiohttp


class ToolBackend(abc.ABC):
    @abc.abstractmethod
    async def setup(self, task_config: Dict[str, Any]) -> bool: ...

    @abc.abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]: ...

    @abc.abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any: ...

    @abc.abstractmethod
    async def teardown(self) -> None: ...

    async def get_patch(self) -> str:
        return ""


class MCPToolBackend(ToolBackend):
    def __init__(
        self,
        session: aiohttp.ClientSession,
        mcp_server_url: str,
        enabled_tools: Sequence[str] = (),
    ):
        self._session = session
        self._mcp_url = mcp_server_url
        self._enabled_tools = enabled_tools
        self._tools: List[Dict[str, Any]] = []

    async def setup(self, task_config: Dict[str, Any]) -> bool:
        return True

    async def list_tools(self) -> List[Dict[str, Any]]:
        if self._tools:
            return self._tools

        async with self._session.post(
            f"{self._mcp_url.rstrip('/')}/list-tools"
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"list-tools failed ({resp.status}): {body}")
            payload = await resp.json()

        from agent_cap.runner.llm_client import _fix_tool_schema

        enabled = set(self._enabled_tools)
        for tool in payload:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", ""))
            if not name:
                continue
            if enabled and name not in enabled:
                continue
            self._tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(tool.get("description", "")),
                        "parameters": _fix_tool_schema(
                            tool.get("input_schema", {}), name
                        ),
                    },
                }
            )
        return self._tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        async with self._session.post(
            f"{self._mcp_url.rstrip('/')}/call-tool",
            json={"tool_name": name, "tool_args": arguments},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"tool call failed ({resp.status}): {body}")
            return await resp.json()

    async def teardown(self) -> None:
        pass


class SWEBenchToolBackend(ToolBackend):
    def __init__(self, runtime: str = "docker"):
        self._runtime = runtime
        self._backend: Optional[Any] = None
        self._tools: List[Dict[str, Any]] = []

    async def setup(self, task_config: Dict[str, Any]) -> bool:
        from agent_cap.backends.swebench_backend import SWEBenchBackend

        self._backend = SWEBenchBackend(runtime=self._runtime)
        return self._backend.setup(task_config)

    async def list_tools(self) -> List[Dict[str, Any]]:
        if self._tools:
            return self._tools
        from agent_cap.backends.tool_executor import TOOL_DEFINITIONS

        self._tools = list(TOOL_DEFINITIONS)
        return self._tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if not self._backend:
            raise RuntimeError("Backend not set up")
        result = self._backend.execute(name, "call", arguments)
        if result.success:
            return [{"type": "text", "text": result.output}]
        raise RuntimeError(result.output)

    async def teardown(self) -> None:
        if self._backend:
            self._backend.teardown()
            self._backend.cleanup()
            self._backend = None

    async def get_patch(self) -> str:
        if self._backend:
            return self._backend.get_patch()
        return ""

    def run_tests(self, timeout: int = 300) -> Dict[str, Any]:
        if self._backend:
            return self._backend.run_tests(timeout=timeout)
        return {"passed": False, "reason": "no backend"}
