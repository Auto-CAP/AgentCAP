"""Streaming OpenAI-compatible chat client with true TTFT and TPOT measurement.

Uses SSE (Server-Sent Events) streaming to measure:
- TTFT: Time To First Token (request sent -> first content chunk received)
- TPOT: Time Per Output Token (inter-token intervals, avg and p99)

Only uses stdlib — no external dependencies.
"""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StreamingChatResponse:
    """Response from a streaming chat completion request.

    Extends the concept of ChatResponse with granular timing metrics
    obtained from SSE streaming.
    """

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
    token_timestamps: List[float] = field(default_factory=list)
    raw_chunks: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def _compute_percentile(values: List[float], percentile: float) -> float:
    """Compute the given percentile from a list of values.

    Args:
        values: List of numeric values.
        percentile: Percentile to compute (0-100).

    Returns:
        The percentile value, or 0.0 if the list is empty.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * percentile / 100.0)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def _mean(values: List[float]) -> float:
    """Compute mean of a list, returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


class StreamingChatClient:
    """OpenAI-compatible streaming chat client with TTFT/TPOT measurement.

    Sends requests with ``stream=True`` and parses SSE chunks to measure
    the exact time-to-first-token and per-token generation latency.

    Usage::

        client = StreamingChatClient("http://localhost:8000")
        resp = client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-oss-120b",
        )
        print(f"TTFT={resp.ttft_ms:.1f}ms  TPOT_avg={resp.tpot_ms_avg:.1f}ms")
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._server_model_id: Optional[str] = None

    def get_server_model_id(self) -> Optional[str]:
        """Query /v1/models to get the model ID the server actually loaded."""
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: int = 600,
        stop_token_ids: Optional[List[int]] = None,
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

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        content_parts: List[str] = []
        raw_chunks: List[Dict[str, Any]] = []
        token_timestamps: List[float] = []
        tool_call_fragments: Dict[int, Dict[str, str]] = {}

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        resp_model = model
        first_token_time: Optional[float] = None

        t_start = time.perf_counter()

        try:
            response = urllib.request.urlopen(req, timeout=timeout)

            for line_bytes in response:
                line = line_bytes.decode("utf-8").rstrip("\n").rstrip("\r")
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[len("data: ") :]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                raw_chunks.append(chunk)
                resp_model = chunk.get("model", resp_model)

                # Extract usage (vLLM sends in last chunk)
                usage = chunk.get("usage")
                if usage:
                    input_tokens = int(usage.get("prompt_tokens", 0))
                    output_tokens = int(usage.get("completion_tokens", 0))
                    total_tokens = int(
                        usage.get("total_tokens", input_tokens + output_tokens)
                    )

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                # --- Content tokens ---
                content_piece = delta.get("content")
                if content_piece:
                    now = time.perf_counter()
                    token_timestamps.append(now - t_start)
                    if first_token_time is None:
                        first_token_time = now
                    content_parts.append(content_piece)

                # --- Tool call deltas ---
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
                            tool_call_fragments[idx]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tool_call_fragments[idx]["arguments"] += fn["arguments"]

            response.close()

        except Exception as exc:
            t_end = time.perf_counter()
            return StreamingChatResponse(
                content="".join(content_parts),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=(t_end - t_start) * 1000,
                ttft_ms=(
                    (first_token_time - t_start) * 1000
                    if first_token_time
                    else (t_end - t_start) * 1000
                ),
                tpot_ms_avg=0.0,
                tpot_ms_p99=0.0,
                model=resp_model,
                tool_call_count=len(tool_call_fragments),
                token_timestamps=token_timestamps,
                raw_chunks=raw_chunks,
                error=str(exc),
            )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        ttft_ms = (
            (first_token_time - t_start) * 1000 if first_token_time else latency_ms
        )

        # Compute TPOT from inter-token intervals
        intervals_ms: List[float] = []
        if len(token_timestamps) >= 2:
            for i in range(1, len(token_timestamps)):
                interval = (token_timestamps[i] - token_timestamps[i - 1]) * 1000
                intervals_ms.append(interval)

        tpot_avg = _mean(intervals_ms)
        tpot_p99 = _compute_percentile(intervals_ms, 99)

        # Infer tool call count
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
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms_avg=tpot_avg,
            tpot_ms_p99=tpot_p99,
            model=resp_model,
            tool_call_count=tool_call_count,
            token_timestamps=token_timestamps,
            raw_chunks=raw_chunks,
        )

    def chat_batch(
        self,
        messages_list: List[List[Dict[str, Any]]],
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 4096,
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
