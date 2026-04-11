"""Streaming chat client with true TTFT and TPOT measurement.

Uses aiohttp for real SSE streaming (same approach as vLLM's own
benchmark_serving.py). urllib buffers the entire chunked response,
making all token timestamps identical. aiohttp's
``async for chunk in response.content`` yields each SSE event as it
arrives from the TCP stream.
"""

import asyncio
import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import aiohttp
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
    Conversation,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
logger = logging.getLogger(__name__)


@dataclass
class StreamingChatResponse:
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    ttft_ms: float
    tpot_ms_avg: float
    tpot_ms_p99: float
    model: str
    tool_call_count: int = 0
    itl: List[float] = field(default_factory=list)
    token_timestamps: List[float] = field(default_factory=list)
    raw_chunks: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def _compute_percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * percentile / 100.0)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class StreamingChatClient:
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._server_model_id: Optional[str] = None

    def get_server_model_id(self) -> Optional[str]:
        if self._server_model_id is not None:
            return self._server_model_id
        try:
            url = f"{self.base_url}/v1/models"
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                self._server_model_id = models[0].get("id", None)
            return self._server_model_id
        except Exception:
            return None

    def scrape_server_metrics(self) -> Dict[str, float]:
        """Scrape vLLM Prometheus /metrics for server-side TTFT/TPOT."""
        try:
            url = f"{self.base_url}/metrics"
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=5)
            text = resp.read().decode("utf-8")
        except Exception:
            return {}

        result: Dict[str, float] = {}
        for line in text.split("\n"):
            if line.startswith("#"):
                continue
            if "time_to_first_token_seconds_sum" in line:
                result["ttft_sum"] = float(line.split()[-1])
            elif "time_to_first_token_seconds_count" in line:
                result["ttft_count"] = float(line.split()[-1])
            elif "time_per_output_token_seconds_sum" in line:
                result["tpot_sum"] = float(line.split()[-1])
            elif "time_per_output_token_seconds_count" in line:
                result["tpot_count"] = float(line.split()[-1])
        return result

    def compute_server_tpot(
        self, before: Dict[str, float], after: Dict[str, float]
    ) -> tuple:
        """Compute avg TTFT/TPOT from delta of two /metrics scrapes."""
        ttft_avg = 0.0
        tpot_avg = 0.0
        d_ttft_sum = after.get("ttft_sum", 0) - before.get("ttft_sum", 0)
        d_ttft_count = after.get("ttft_count", 0) - before.get("ttft_count", 0)
        d_tpot_sum = after.get("tpot_sum", 0) - before.get("tpot_sum", 0)
        d_tpot_count = after.get("tpot_count", 0) - before.get("tpot_count", 0)
        if d_ttft_count > 0:
            ttft_avg = (d_ttft_sum / d_ttft_count) * 1000
        if d_tpot_count > 0:
            tpot_avg = (d_tpot_sum / d_tpot_count) * 1000
        return ttft_avg, tpot_avg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: int = 600,
        stop_token_ids: Optional[List[int]] = None,
    ) -> StreamingChatResponse:
        from concurrent.futures import ThreadPoolExecutor

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._async_chat(
                        messages,
                        model,
                        temperature,
                        max_tokens,
                        tools,
                        timeout,
                        stop_token_ids,
                    )
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result(timeout=timeout + 30)

    async def _async_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]],
        timeout: int,
        stop_token_ids: Optional[List[int]],
    ) -> StreamingChatResponse:
        url = f"{self.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        if stop_token_ids:
            payload["stop_token_ids"] = stop_token_ids

        content_parts: List[str] = []
        raw_chunks: List[Dict[str, Any]] = []
        itl: List[float] = []
        tool_call_fragments: Dict[int, Dict[str, str]] = {}

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        st = time.perf_counter()
        resp_model = model
        ttft = 0.0
        most_recent_timestamp = 0.0

        max_retries = 3
        retryable_codes = {429, 502, 503}

        for attempt in range(max_retries + 1):
            st = time.perf_counter()

            try:
                async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
                    async with session.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.status != 200:
                            error_body = await response.text()
                            if (
                                response.status in retryable_codes
                                and attempt < max_retries
                            ):
                                wait = 5 * (3**attempt)
                                logger.warning(
                                    f"HTTP {response.status}, retry {attempt + 1}/{max_retries} in {wait}s"
                                )
                                await asyncio.sleep(wait)
                                continue
                            return StreamingChatResponse(
                                content="",
                                input_tokens=0,
                                output_tokens=0,
                                total_tokens=0,
                                latency_ms=(time.perf_counter() - st) * 1000,
                                ttft_ms=0.0,
                                tpot_ms_avg=0.0,
                                tpot_ms_p99=0.0,
                                model=model,
                                error=f"HTTP {response.status}: {error_body[:500]}",
                            )

                        done = False
                        async for chunk_bytes in response.content:
                            if done:
                                break
                            for raw_line in chunk_bytes.decode("utf-8").split("\n"):
                                raw_line = raw_line.strip()
                                if not raw_line or raw_line.startswith(":"):
                                    continue
                                raw_line = raw_line.removeprefix("data: ").removeprefix(
                                    "data:"
                                )
                                if raw_line == "[DONE]":
                                    done = True
                                    break
                                try:
                                    data = json.loads(raw_line)
                                except json.JSONDecodeError:
                                    continue

                                timestamp = time.perf_counter()
                                raw_chunks.append(data)
                                resp_model = data.get("model", resp_model)

                                usage = data.get("usage")
                                if usage:
                                    input_tokens = int(usage.get("prompt_tokens", 0))
                                    output_tokens = int(
                                        usage.get("completion_tokens", 0)
                                    )
                                    total_tokens = int(
                                        usage.get(
                                            "total_tokens", input_tokens + output_tokens
                                        )
                                    )
                                    most_recent_timestamp = timestamp
                                    continue

                                choices = data.get("choices", [])
                                if not choices:
                                    most_recent_timestamp = timestamp
                                    continue

                                delta = choices[0].get("delta", {})
                                has_output = False

                                content_piece = delta.get("content")
                                if content_piece:
                                    content_parts.append(content_piece)
                                    has_output = True

                                if delta.get("reasoning_content") or delta.get(
                                    "reasoning"
                                ):
                                    has_output = True

                                tc_deltas = delta.get("tool_calls")
                                if tc_deltas:
                                    for tc in tc_deltas:
                                        idx = tc.get("index", 0)
                                        if idx not in tool_call_fragments:
                                            tool_call_fragments[idx] = {
                                                "name": "",
                                                "arguments": "",
                                            }
                                        fn = tc.get("function", {})
                                        if fn.get("name"):
                                            tool_call_fragments[idx]["name"] = fn[
                                                "name"
                                            ]
                                            has_output = True
                                        if fn.get("arguments"):
                                            tool_call_fragments[idx]["arguments"] += fn[
                                                "arguments"
                                            ]
                                            has_output = True

                                if ttft == 0.0:
                                    if has_output:
                                        ttft = timestamp - st
                                else:
                                    itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                break

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < max_retries:
                    wait = 5 * (3**attempt)
                    logger.warning(
                        f"Connection error: {exc}, retry {attempt + 1}/{max_retries} in {wait}s"
                    )
                    await asyncio.sleep(wait)
                    continue
                return StreamingChatResponse(
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    latency_ms=(time.perf_counter() - st) * 1000,
                    ttft_ms=0.0,
                    tpot_ms_avg=0.0,
                    tpot_ms_p99=0.0,
                    model=model,
                    error=f"Connection failed after {max_retries} retries: {exc}",
                )
            except Exception as exc:
                t_end = time.perf_counter()
                return StreamingChatResponse(
                    content="".join(content_parts),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=(t_end - st) * 1000,
                    ttft_ms=ttft * 1000,
                    tpot_ms_avg=0.0,
                    tpot_ms_p99=0.0,
                    model=resp_model,
                    tool_call_count=len(tool_call_fragments),
                    itl=[x * 1000 for x in itl],
                    raw_chunks=raw_chunks,
                    error=str(exc),
                )

        latency = (
            most_recent_timestamp - st
            if most_recent_timestamp > st
            else time.perf_counter() - st
        )

        tpot_avg = _mean(itl) * 1000 if itl else 0.0
        tpot_p99 = _compute_percentile([x * 1000 for x in itl], 99)

        # Fallback: use formula if aiohttp still batched (shouldn't happen)
        if tpot_avg == 0.0 and output_tokens > 1 and ttft > 0:
            decode_s = latency - ttft
            if decode_s > 0:
                tpot_avg = (decode_s / (output_tokens - 1)) * 1000
                tpot_p99 = tpot_avg

        tool_call_count = len(tool_call_fragments)
        finish_reason = ""
        if raw_chunks:
            last_choices = raw_chunks[-1].get("choices", [])
            if last_choices:
                finish_reason = last_choices[0].get("finish_reason", "") or ""
        if finish_reason == "tool_calls" and tool_call_count == 0:
            tool_call_count = 1

        return StreamingChatResponse(
            content="".join(content_parts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency * 1000,
            ttft_ms=ttft * 1000,
            tpot_ms_avg=tpot_avg,
            tpot_ms_p99=tpot_p99,
            model=resp_model,
            tool_call_count=tool_call_count,
            itl=[x * 1000 for x in itl],
            raw_chunks=raw_chunks,
        )

    def chat_batch(
        self,
        messages_list: List[List[Dict[str, Any]]],
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        tools: Optional[List[Dict[str, Any]]] = None,
        concurrency: int = 1,
        timeout: int = 600,
        stop_token_ids: Optional[List[int]] = None,
    ) -> List[StreamingChatResponse]:
        results: List[Optional[StreamingChatResponse]] = [None] * len(messages_list)

        def _run(idx: int, msgs: List[Dict[str, Any]]) -> None:
            results[idx] = self.chat(
                messages=msgs,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                timeout=timeout,
                stop_token_ids=stop_token_ids,
            )

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_run, i, msgs): i for i, msgs in enumerate(messages_list)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    results[idx] = StreamingChatResponse(
                        content="",
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        latency_ms=0.0,
                        ttft_ms=0.0,
                        tpot_ms_avg=0.0,
                        tpot_ms_p99=0.0,
                        model=model,
                        error=str(exc),
                    )

        return [r for r in results if r is not None]

    def completion_from_prompt_ids(
        self,
        prompt_token_ids: List[int],
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        timeout: int = 600,
        stop_token_ids: Optional[List[int]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> StreamingChatResponse:
        from concurrent.futures import ThreadPoolExecutor

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._async_completion_from_prompt_ids(  # pyright: ignore[reportAttributeAccessIssue]
                        prompt_token_ids=prompt_token_ids,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        stop_token_ids=stop_token_ids,
                        extra_body=extra_body,
                    )
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result(timeout=timeout + 30)
