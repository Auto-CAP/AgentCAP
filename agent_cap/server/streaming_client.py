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
from typing import Any, Dict, Iterator, List, Optional


def _iter_sse_lines(response) -> Iterator[str]:
    """Yield SSE lines in real-time using unbuffered read1().

    Python's HTTPResponse wraps the socket in BufferedIOBase. Both
    __iter__ and readline() can over-buffer, causing all SSE events
    to arrive at once.  read1() returns data as soon as the OS has
    any bytes available (like C's read(2)), giving us true per-token
    timestamps for TTFT/TPOT.
    """
    buf = b""
    read = getattr(response, "read1", None) or response.read
    while True:
        chunk = read(4096)
        if not chunk:
            if buf:
                yield buf.decode("utf-8", errors="replace").rstrip("\r\n")
            break
        buf += chunk
        while b"\n" in buf:
            line_bytes, buf = buf.split(b"\n", 1)
            yield line_bytes.decode("utf-8", errors="replace").rstrip("\r")


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

    def __init__(self, base_url: str = "http://localhost:30000") -> None:
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
        stream: bool = True,
    ) -> StreamingChatResponse:
        if not stream:
            return self._chat_non_stream(
                messages, model, temperature, max_tokens, tools, timeout
            )

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

            for line in _iter_sse_lines(response):
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

                content_piece = delta.get("content")
                if content_piece:
                    now = time.perf_counter()
                    token_timestamps.append(now - t_start)
                    if first_token_time is None:
                        first_token_time = now
                    content_parts.append(content_piece)

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
                            now = time.perf_counter()
                            token_timestamps.append(now - t_start)
                            if first_token_time is None:
                                first_token_time = now

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

        # Compute TPOT from inter-chunk intervals, with fallback estimation
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

    def chat_responses_api(
        self,
        input_items: List[Dict[str, Any]],
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: int = 600,
    ) -> StreamingChatResponse:
        """Streaming request via /v1/responses (OpenAI Responses API).

        Avoids the openai_harmony tool-call parser used by
        /v1/chat/completions.  Tool definitions use the flat Responses
        format: ``{type, name, description, parameters}`` (no nested
        ``function`` key).  Multi-turn tool results are passed as
        ``{type: "function_call_output", call_id, output}`` items.
        """
        url = f"{self.base_url}/v1/responses"

        # Convert chat-completions tool format → responses flat format
        resp_tools = None
        if tools:
            resp_tools = []
            for t in tools:
                fn = t.get("function", t)
                resp_tools.append(
                    {
                        "type": "function",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                )

        payload: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": True,
            "store": False,
        }
        if max_tokens:
            payload["max_output_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if resp_tools:
            payload["tools"] = resp_tools

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        content_parts: List[str] = []
        token_timestamps: List[float] = []
        raw_events: List[Dict[str, Any]] = []
        tool_calls_collected: List[Dict[str, Any]] = []
        tc_fragments: Dict[str, Dict[str, str]] = {}  # keyed by call_id

        input_tokens = 0
        output_tokens = 0
        resp_model = model
        first_token_time: Optional[float] = None

        t_start = time.perf_counter()

        try:
            response = urllib.request.urlopen(req, timeout=timeout)
            event_type = ""

            for line in _iter_sse_lines(response):
                if line.startswith("event: "):
                    event_type = line[len("event: ") :]
                    continue
                if not line.startswith("data: "):
                    continue

                data_str = line[len("data: ") :]
                try:
                    evt = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                raw_events.append({"event": event_type, "data": evt})

                if event_type == "response.output_text.delta":
                    delta_text = evt.get("delta", "")
                    if delta_text:
                        now = time.perf_counter()
                        token_timestamps.append(now - t_start)
                        if first_token_time is None:
                            first_token_time = now
                        content_parts.append(delta_text)

                elif event_type == "response.output_item.added":
                    item = evt.get("item", {})
                    if item.get("type") == "function_call":
                        cid = item.get("call_id", item.get("id", ""))
                        tc_fragments[cid] = {
                            "id": cid,
                            "name": item.get("name", ""),
                            "arguments": "",
                        }

                elif event_type == "response.function_call_arguments.delta":
                    cid = evt.get("call_id", evt.get("item_id", ""))
                    if cid in tc_fragments:
                        delta_arg = evt.get("delta", "")
                        tc_fragments[cid]["arguments"] += delta_arg
                        if delta_arg:
                            now = time.perf_counter()
                            token_timestamps.append(now - t_start)
                            if first_token_time is None:
                                first_token_time = now

                elif event_type == "response.output_item.done":
                    item = evt.get("item", {})
                    if item.get("type") == "function_call":
                        cid = item.get("call_id", item.get("id", ""))
                        tc = tc_fragments.pop(cid, None)
                        if tc:
                            if not tc["arguments"]:
                                tc["arguments"] = item.get("arguments", "")
                            if not tc["name"]:
                                tc["name"] = item.get("name", "")
                            tool_calls_collected.append(tc)

                elif event_type == "response.completed":
                    resp_data = evt.get("response", {})
                    usage = resp_data.get("usage", {})
                    if usage:
                        input_tokens = int(
                            usage.get("input_tokens", usage.get("prompt_tokens", 0))
                        )
                        output_tokens = int(
                            usage.get(
                                "output_tokens", usage.get("completion_tokens", 0)
                            )
                        )
                    resp_model = resp_data.get("model", resp_model)

            response.close()

        except Exception as exc:
            t_end = time.perf_counter()
            return StreamingChatResponse(
                content="".join(content_parts),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_ms=(t_end - t_start) * 1000,
                ttft_ms=(
                    (first_token_time - t_start) * 1000 if first_token_time else 0.0
                ),
                tpot_ms_avg=0.0,
                tpot_ms_p99=0.0,
                model=resp_model,
                tool_call_count=len(tool_calls_collected),
                raw_chunks=raw_events,
                error=str(exc),
            )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        ttft_ms = (
            (first_token_time - t_start) * 1000 if first_token_time else latency_ms
        )

        intervals_ms: List[float] = []
        if len(token_timestamps) >= 2:
            for i in range(1, len(token_timestamps)):
                intervals_ms.append(
                    (token_timestamps[i] - token_timestamps[i - 1]) * 1000
                )

        tpot_avg = _mean(intervals_ms)
        tpot_p99 = _compute_percentile(intervals_ms, 99)

        return StreamingChatResponse(
            content="".join(content_parts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms_avg=tpot_avg,
            tpot_ms_p99=tpot_p99,
            model=resp_model,
            tool_call_count=len(tool_calls_collected),
            token_timestamps=token_timestamps,
            raw_chunks=raw_events,
        )

    def _chat_non_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]],
        timeout: int,
    ) -> StreamingChatResponse:
        url = f"{self.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        try:
            response = urllib.request.urlopen(req, timeout=timeout)
            body = json.loads(response.read().decode("utf-8"))
            response.close()
        except Exception as exc:
            t_end = time.perf_counter()
            return StreamingChatResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=(t_end - t_start) * 1000,
                ttft_ms=0.0,
                tpot_ms_avg=0.0,
                tpot_ms_p99=0.0,
                model=model,
                error=str(exc),
            )
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        usage = body.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))

        tool_call_count = 0
        tc_list = message.get("tool_calls")
        if tc_list:
            tool_call_count = len(tc_list)

        tpot_avg = 0.0
        if output_tokens > 1:
            tpot_avg = latency_ms / output_tokens

        return StreamingChatResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            ttft_ms=latency_ms,
            tpot_ms_avg=tpot_avg,
            tpot_ms_p99=tpot_avg,
            model=body.get("model", model),
            tool_call_count=tool_call_count,
            raw_chunks=[body],
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
        stream: bool = True,
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
                stream=stream,
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
