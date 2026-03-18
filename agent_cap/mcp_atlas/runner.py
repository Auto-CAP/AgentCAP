import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp
from datasets import load_dataset

from agent_cap.server.cpu_monitor import CPUMonitor
from agent_cap.server.gpu_monitor import GPUMonitor
from agent_cap.server.streaming_client import StreamingChatClient

logger = logging.getLogger("agent_cap.mcp_atlas")

SYSTEM_PROMPT = (
    "You are a factual, tool-aware assistant connected to a variety of tools. "
    "Use the available tools to answer the user query. Do not ask the user for "
    "clarification; fully complete the task using the information provided."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass
class MCPAtlasTask:
    task_id: str
    prompt: str
    enabled_tools: List[str]
    gtfa_claims: List[str]
    trajectory: List[Dict[str, Any]]


@dataclass
class MCPAtlasResult:
    task_id: str
    response: str
    tool_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float
    tpot_ms_avg: float
    errors: List[str] = field(default_factory=list)


def load_mcpatlas_tasks(limit: int = 0) -> List[MCPAtlasTask]:
    ds = load_dataset("ScaleAI/mcp-atlas", split="train")
    tasks = []
    for ex in ds:
        tools = ex.get("ENABLED_TOOLS", [])
        if isinstance(tools, str):
            tools = json.loads(tools)
        claims = ex.get("GTFA_CLAIMS", [])
        if isinstance(claims, str):
            claims = json.loads(claims)
        traj = ex.get("TRAJECTORY", [])
        if isinstance(traj, str):
            traj = json.loads(traj)
        tasks.append(
            MCPAtlasTask(
                task_id=ex.get("TASK", ""),
                prompt=ex.get("PROMPT", ""),
                enabled_tools=tools,
                gtfa_claims=claims,
                trajectory=traj,
            )
        )
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


async def list_openai_tools(
    session: aiohttp.ClientSession, mcp_url: str, enabled: Sequence[str]
) -> List[Dict[str, Any]]:
    async with session.get(f"{mcp_url}/tools/list") as resp:
        data = await resp.json()
    all_tools = data.get("tools", [])
    enabled_set = set(enabled)
    result = []
    for t in all_tools:
        tool_name = t.get("name", "")
        if tool_name in enabled_set or f"{tool_name}" in enabled_set:
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {}),
                },
            }
            result.append(schema)
    return result


async def mcp_call_tool(
    session: aiohttp.ClientSession, mcp_url: str, name: str, args: Dict
) -> List[Dict[str, Any]]:
    payload = {"name": name, "arguments": args}
    async with session.post(
        f"{mcp_url}/tools/call", json=payload, timeout=aiohttp.ClientTimeout(total=120)
    ) as resp:
        data = await resp.json()
    return data.get("content", [])


def flatten_tool_payload(payload: List[Dict[str, Any]]) -> str:
    parts = []
    for item in payload:
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        else:
            parts.append(json.dumps(item))
    return "\n".join(parts)


class MCPAtlasRunner:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_id: str = "default",
        mcp_server_url: str = "http://localhost:1984",
        max_turns: int = 20,
        max_tokens: int = 16384,
        temperature: float = 0.0,
    ):
        self.client = StreamingChatClient(base_url=base_url)
        self.model_id = model_id
        self.mcp_server_url = mcp_server_url
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature

        server_model = self.client.get_server_model_id()
        if server_model and server_model != model_id:
            logger.info("Server model: %s", server_model)
            self.model_id = server_model

    def run(self, tasks: List[MCPAtlasTask]) -> List[MCPAtlasResult]:
        results = []
        for i, task in enumerate(tasks):
            logger.info("Task %d/%d: %s", i + 1, len(tasks), task.task_id)
            print(f"\n  task={task.task_id}  prompt={task.prompt[:60]}...")
            result = self._run_task(task)
            results.append(result)
            print(
                f"    tools={result.tool_calls}  "
                f"in_tok={result.input_tokens}  out_tok={result.output_tokens}  "
                f"ttft={result.ttft_ms:.1f}ms  tpot={result.tpot_ms_avg:.1f}ms  "
                f"latency={result.latency_ms:.1f}ms"
            )
        return results

    def _run_task(self, task: MCPAtlasTask) -> MCPAtlasResult:
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._async_run_task(task))
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result(timeout=600)

    async def _async_run_task(self, task: MCPAtlasTask) -> MCPAtlasResult:
        async with aiohttp.ClientSession() as session:
            tools = await list_openai_tools(
                session, self.mcp_server_url, task.enabled_tools
            )

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.prompt},
            ]

            errors: List[str] = []
            total_input = 0
            total_output = 0
            total_tool_calls = 0
            first_ttft = 0.0
            all_tpot: List[float] = []
            t_start = time.perf_counter()

            for turn in range(self.max_turns):
                payload = {
                    "model": self.model_id,
                    "messages": messages,
                    "tools": tools if tools else None,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                }
                try:
                    async with session.post(
                        f"{self.client.base_url}/v1/chat/completions",
                        json={k: v for k, v in payload.items() if v is not None},
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        result = await resp.json()
                except Exception as exc:
                    errors.append(str(exc))
                    break

                usage = result.get("usage", {})
                total_input += int(usage.get("prompt_tokens", 0))
                total_output += int(usage.get("completion_tokens", 0))

                choices = result.get("choices", [])
                if not choices:
                    break

                assistant = choices[0].get("message", {})
                messages.append(assistant)

                tool_calls = assistant.get("tool_calls", [])
                if not tool_calls:
                    break

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = (
                            json.loads(raw_args)
                            if isinstance(raw_args, str)
                            else raw_args
                        )
                    except json.JSONDecodeError:
                        args = {}

                    total_tool_calls += 1
                    print(f"      turn={turn}  → {name}({json.dumps(args)[:60]})")

                    try:
                        tool_result = await mcp_call_tool(
                            session, self.mcp_server_url, name, args
                        )
                    except Exception as exc:
                        errors.append(f"{name}: {exc}")
                        tool_result = [{"type": "text", "text": f"ERROR: {exc}"}]

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": flatten_tool_payload(tool_result),
                        }
                    )

            elapsed_ms = (time.perf_counter() - t_start) * 1000

            final_text = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_text = THINK_RE.sub("", msg["content"]).strip()
                    break

            return MCPAtlasResult(
                task_id=task.task_id,
                response=final_text,
                tool_calls=total_tool_calls,
                input_tokens=total_input,
                output_tokens=total_output,
                latency_ms=elapsed_ms,
                ttft_ms=first_ttft,
                tpot_ms_avg=0.0,
                errors=errors,
            )
