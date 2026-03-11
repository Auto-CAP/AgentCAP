from dataclasses import dataclass
from typing import Any, Dict, List
import json
import time
import urllib.request


@dataclass
class ChatResponse:
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    ttft_ms: float
    model: str
    raw_response: Dict[str, Any]
    tool_call_count: int = 0


class ChatClient:
    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict],
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read()
        end = time.perf_counter()

        raw = json.loads(body.decode("utf-8"))
        usage = raw.get("usage", {})
        choices = raw.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        tool_calls = message.get("tool_calls", [])
        tool_call_count = len(tool_calls) if tool_calls else 0
        finish_reason = choices[0].get("finish_reason", "") if choices else ""
        if finish_reason == "tool_calls" and tool_call_count == 0:
            tool_call_count = 1
        content = message.get("content", "")
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        latency_ms = (end - start) * 1000

        return ChatResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            ttft_ms=latency_ms,
            model=raw.get("model", model),
            raw_response=raw,
            tool_call_count=tool_call_count,
        )
