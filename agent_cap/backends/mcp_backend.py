import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import aiohttp

from agent_cap.core.tool_backend import ToolBackend, ToolResult


class MCPBackend(ToolBackend):
    def __init__(self, mcp_server_url: str = "http://localhost:1984"):
        self.mcp_url = mcp_server_url
        self._enabled_tools: List[str] = []
        self._tool_defs: List[Dict[str, Any]] = []

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return self._tool_defs

    def setup(self, task_config: Dict[str, Any]) -> bool:
        self._enabled_tools = task_config.get("enabled_tools", [])
        if isinstance(self._enabled_tools, str):
            self._enabled_tools = json.loads(self._enabled_tools)

        def _fetch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._fetch_tools())
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        with ThreadPoolExecutor(max_workers=1) as pool:
            self._tool_defs = pool.submit(_fetch).result(timeout=30)
        return len(self._tool_defs) > 0

    async def _fetch_tools(self) -> List[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.mcp_url}/tools/list") as resp:
                data = await resp.json()
        all_tools = data.get("tools", [])
        enabled_set = set(self._enabled_tools)
        result = []
        for t in all_tools:
            name = t.get("name", "")
            if name in enabled_set:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": t.get("description", ""),
                            "parameters": t.get("inputSchema", {}),
                        },
                    }
                )
        return result

    def execute(
        self, tool_name: str, tool_call_id: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        def _call():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._call_tool(tool_name, arguments))
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                payload = pool.submit(_call).result(timeout=120)
            parts = []
            for item in payload:
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(json.dumps(item))
            output = "\n".join(parts)
            success = True
        except Exception as exc:
            output = f"ERROR: {exc}"
            success = False
        latency_ms = (time.perf_counter() - t0) * 1000

        return ToolResult(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            output=output[:16000],
            latency_ms=latency_ms,
            success=success,
        )

    async def _call_tool(self, name: str, args: Dict) -> List[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.mcp_url}/tools/call",
                json={"name": name, "arguments": args},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
        return data.get("content", [])

    def teardown(self) -> None:
        pass

    def get_patch(self) -> str:
        return ""
