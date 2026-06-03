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


_HARMONY_STOP_CACHE: Optional[List[int]] = None


def _harmony_stop_token_ids() -> List[int]:
    global _HARMONY_STOP_CACHE
    if _HARMONY_STOP_CACHE is None:
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        _HARMONY_STOP_CACHE = list(map(int, enc.stop_tokens_for_assistant_actions()))
    return _HARMONY_STOP_CACHE


async def _stream_chat(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_call_fragments: Dict[int, Dict[str, str]] = {}
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    cached_tokens = 0
    sglang_style = False
    ttft_ms = 0.0
    itl_ms: List[float] = []
    t_start = time.perf_counter()
    most_recent = t_start

    async with session.post(
        url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
        done = False
        async for chunk_bytes in resp.content:
            if done:
                break
            for raw_line in chunk_bytes.decode("utf-8").split("\n"):
                raw_line = raw_line.strip()
                if not raw_line or raw_line.startswith(":"):
                    continue
                raw_line = raw_line.removeprefix("data: ").removeprefix("data:")
                if raw_line == "[DONE]":
                    done = True
                    break
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                ts = time.perf_counter()
                usage = data.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
                    completion_tokens = int(usage.get("completion_tokens", 0))
                    details = usage.get("completion_tokens_details") or {}
                    reasoning_tokens = int(details.get("reasoning_tokens") or 0)
                    top_r = usage.get("reasoning_tokens")
                    if top_r is not None and reasoning_tokens == 0:
                        reasoning_tokens = int(top_r or 0)
                        sglang_style = True
                    pdet = usage.get("prompt_tokens_details") or {}
                    cached_tokens = int(pdet.get("cached_tokens") or 0)
                    most_recent = ts
                    continue

                choices = data.get("choices", [])
                if not choices:
                    most_recent = ts
                    continue
                delta = choices[0].get("delta", {}) or {}
                has_output = False

                cp = delta.get("content")
                if cp:
                    content_parts.append(cp)
                    has_output = True

                rp = delta.get("reasoning_content") or delta.get("reasoning")
                if rp:
                    reasoning_parts.append(rp)
                    has_output = True

                tcs = delta.get("tool_calls")
                if tcs:
                    for tc in tcs:
                        idx = tc.get("index", 0)
                        slot = tool_call_fragments.setdefault(
                            idx,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                            has_output = True
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                            has_output = True

                if ttft_ms == 0.0:
                    if has_output:
                        ttft_ms = (ts - t_start) * 1000
                else:
                    itl_ms.append((ts - most_recent) * 1000)
                most_recent = ts

    tool_calls = [
        {
            "id": frag["id"] or f"call_{i}",
            "type": "function",
            "function": {"name": frag["name"], "arguments": frag["arguments"]},
        }
        for i, frag in sorted(tool_call_fragments.items())
        if frag["name"]
    ]
    if reasoning_tokens == 0 and reasoning_parts:
        r_text = "".join(reasoning_parts)
        try:
            import tiktoken
            reasoning_tokens = len(tiktoken.get_encoding("o200k_harmony").encode(r_text))
            sglang_style = True
        except Exception:
            reasoning_tokens = max(1, len(r_text) // 4)
            sglang_style = True
    if sglang_style and completion_tokens >= reasoning_tokens:
        total_output_tokens = completion_tokens
        completion_tokens = total_output_tokens - reasoning_tokens
    else:
        total_output_tokens = completion_tokens + reasoning_tokens
    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_calls": tool_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "total_output_tokens": total_output_tokens,
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
    }


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
    num_turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    latency_ms: float
    ttft_ms: float
    tpot_ms_avg: float
    errors: List[str] = field(default_factory=list)


_FREE_SERVERS = {
    "arxiv", "brave-search", "calculator", "cli-mcp-server",
    "clinicaltrialsgov-mcp-server", "context7", "ddg-search",
    "desktop-commander", "fetch", "filesystem", "git", "github",
    "mcp-code-executor", "mcp-server-code-runner", "memory",
    "met-museum", "open-library", "osm-mcp-server", "pubmed",
    "weather", "whois", "wikipedia",
}


def _server_of(tool_name: str) -> str:
    for s in sorted(_FREE_SERVERS, key=len, reverse=True):
        if tool_name.startswith(s):
            return s
    return tool_name.split("_")[0]


def load_mcpatlas_tasks(limit: int = 0, free_only: bool = False) -> List[MCPAtlasTask]:
    ds = load_dataset("ScaleAI/mcp-atlas", split="train")
    tasks = []
    for ex in ds:
        tools = ex.get("ENABLED_TOOLS", [])
        if isinstance(tools, str):
            tools = json.loads(tools)
        if free_only:
            names = [t if isinstance(t, str) else t.get("name", "") for t in tools]
            servers = {_server_of(t) for t in names if t}
            if not servers.issubset(_FREE_SERVERS):
                continue
        claims = ex.get("GTFA_CLAIMS", [])
        if isinstance(claims, str):
            try:
                claims = json.loads(claims)
            except (json.JSONDecodeError, ValueError):
                claims = [claims] if claims.strip() else []
        traj = ex.get("TRAJECTORY", [])
        if isinstance(traj, str):
            try:
                traj = json.loads(traj)
            except (json.JSONDecodeError, ValueError):
                traj = []
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
    async with session.post(f"{mcp_url}/list-tools") as resp:
        data = await resp.json()
    all_tools = data if isinstance(data, list) else data.get("tools", [])
    enabled_set = {
        e if isinstance(e, str) else (e.get("name", "") if isinstance(e, dict) else "")
        for e in enabled
    }
    result = []
    for t in all_tools:
        tool_name = t.get("name", "")
        if tool_name in enabled_set:
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
    payload = {"tool_name": name, "tool_args": args}
    async with session.post(
        f"{mcp_url}/call-tool", json=payload, timeout=aiohttp.ClientTimeout(total=120)
    ) as resp:
        data = await resp.json()
    if isinstance(data, list):
        return data
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
        per_turn_stats_dir: Optional[str] = None,
    ):
        self.client = StreamingChatClient(base_url=base_url)
        self.model_id = model_id
        self.mcp_server_url = mcp_server_url
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.per_turn_stats_dir = per_turn_stats_dir

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
            total_completion = 0
            total_reasoning = 0
            total_cached = 0
            total_tool_calls = 0
            num_turns = 0
            first_ttft = 0.0
            all_itl: List[float] = []
            t_start = time.perf_counter()

            for turn in range(self.max_turns):
                num_turns += 1
                payload = {
                    "model": self.model_id,
                    "messages": messages,
                    "tools": tools if tools else None,
                    "temperature": self.temperature,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "stop_token_ids": _harmony_stop_token_ids(),
                }
                try:
                    turn_data = await _stream_chat(
                        session,
                        f"{self.client.base_url}/v1/chat/completions",
                        {k: v for k, v in payload.items() if v is not None},
                    )
                except Exception as exc:
                    errors.append(str(exc))
                    break

                total_input += turn_data["prompt_tokens"]
                total_completion += turn_data["completion_tokens"]
                total_reasoning += turn_data["reasoning_tokens"]
                total_cached += turn_data["cached_tokens"]
                if turn_data["ttft_ms"] > 0 and first_ttft == 0.0:
                    first_ttft = turn_data["ttft_ms"]
                all_itl.extend(turn_data["itl_ms"])

                if self.per_turn_stats_dir:
                    import os as _os
                    _os.makedirs(self.per_turn_stats_dir, exist_ok=True)
                    _t_out = turn_data["total_output_tokens"]
                    _itl = turn_data["itl_ms"]
                    _tpot_tok = (sum(_itl) / _t_out) if _t_out > 0 else 0.0
                    _tpot_chunk = (sum(_itl) / len(_itl)) if _itl else 0.0
                    _row = {
                        "turn": turn,
                        "ttft_ms": turn_data["ttft_ms"],
                        "tpot_ms": _tpot_tok,
                        "tpot_chunk_ms": _tpot_chunk,
                        "chunks": len(_itl) + 1,
                        "prompt_tokens": turn_data["prompt_tokens"],
                        "completion_tokens": turn_data["completion_tokens"],
                        "reasoning_tokens": turn_data["reasoning_tokens"],
                        "cached_tokens": turn_data["cached_tokens"],
                        "total_output_tokens": _t_out,
                    }
                    with open(f"{self.per_turn_stats_dir}/{task.task_id}.jsonl", "a") as _f:
                        _f.write(json.dumps(_row) + "\n")

                assistant = {
                    "role": "assistant",
                    "content": turn_data["content"] or None,
                }
                if turn_data["reasoning_content"]:
                    assistant["reasoning_content"] = turn_data["reasoning_content"]
                if turn_data["tool_calls"]:
                    assistant["tool_calls"] = turn_data["tool_calls"]
                messages.append(assistant)

                tool_calls = turn_data["tool_calls"]
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
            if not final_text:
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("reasoning_content"):
                        final_text = msg["reasoning_content"].strip()
                        break

            total_output = total_completion + total_reasoning
            tpot_per_token = (sum(all_itl) / total_output) if total_output > 0 else 0.0
            return MCPAtlasResult(
                task_id=task.task_id,
                response=final_text,
                num_turns=num_turns,
                tool_calls=total_tool_calls,
                input_tokens=total_input,
                output_tokens=total_output,
                completion_tokens=total_completion,
                reasoning_tokens=total_reasoning,
                cached_tokens=total_cached,
                latency_ms=elapsed_ms,
                ttft_ms=first_ttft,
                tpot_ms_avg=tpot_per_token,
                errors=errors,
            )
