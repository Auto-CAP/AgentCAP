from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


@dataclass
class ChatCompletionTimedResult:
    response_json: Dict[str, Any]
    ttft_seconds: float
    decode_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    is_streaming: bool = False


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


_GPT_OSS_STOP_TOKENS: Optional[List[int]] = None


def _get_gpt_oss_extra() -> Dict[str, Any]:
    global _GPT_OSS_STOP_TOKENS
    if _GPT_OSS_STOP_TOKENS is None:
        try:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding

            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            _GPT_OSS_STOP_TOKENS = encoding.stop_tokens_for_assistant_actions()
        except ImportError:
            _GPT_OSS_STOP_TOKENS = [200002, 200012]
    return {"stop_token_ids": _GPT_OSS_STOP_TOKENS}


def _extract_cached_tokens(usage: Dict[str, Any]) -> int:
    cached_tokens = 0
    ptd = usage.get("prompt_tokens_details") or {}
    cached_tokens = _to_int(ptd.get("cached_tokens", 0))
    if cached_tokens == 0:
        cached_tokens = _to_int(usage.get("prompt_cache_hit_tokens", 0))
    if cached_tokens == 0:
        cached_tokens = _to_int(usage.get("cache_read_input_tokens", 0))
    return cached_tokens


def _extract_thinking_content(message: Dict[str, Any]) -> str:
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        return reasoning_content

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    if reasoning is not None:
        try:
            return json.dumps(reasoning, ensure_ascii=False)
        except Exception:
            return str(reasoning)

    content = message.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).lower()
            if block_type not in {"thinking", "reasoning"}:
                continue
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            inner_content = block.get("content")
            if isinstance(inner_content, str) and inner_content:
                parts.append(inner_content)
        return "\n".join(parts)

    return ""


def _load_schema_patches() -> Dict[str, Dict[str, Any]]:
    patch_path = Path(__file__).parent / "tool_schema_patches.json"
    if patch_path.exists():
        with open(patch_path) as f:
            return json.load(f)
    return {}


_SCHEMA_PATCHES = _load_schema_patches()


def _fix_tool_schema(schema: Any, tool_name: str = "") -> Dict[str, Any]:
    if not isinstance(schema, dict):
        schema = {}
    if schema.get("type") is None:
        schema["type"] = "object"
    if "properties" not in schema or not schema["properties"]:
        patch = _SCHEMA_PATCHES.get(tool_name)
        if patch:
            return patch
        schema["properties"] = {}
    return schema


def _clean_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    tool_schemas: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    schema = tool_schemas.get(tool_name)
    if not schema:
        return args
    properties = schema.get("properties", {})
    if not properties:
        return args
    required = set(schema.get("required", []))
    allowed = set(properties.keys())
    extra_keys = set(args.keys()) - allowed
    cleaned = {k: v for k, v in args.items() if k in allowed}
    for req in required:
        if req not in cleaned:
            for ek in extra_keys:
                if isinstance(args[ek], str) and args[ek]:
                    cleaned[req] = args[ek]
                    extra_keys.discard(ek)
                    break
            else:
                prop_type = properties.get(req, {}).get("type", "string")
                if prop_type == "string":
                    cleaned[req] = ""
                elif prop_type in ("number", "integer"):
                    cleaned[req] = 0
                elif prop_type == "boolean":
                    cleaned[req] = False
                elif prop_type == "array":
                    cleaned[req] = []
                else:
                    cleaned[req] = ""
    return cleaned


async def chat_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    max_tokens: int,
    temperature: float = 0.0,
    openrouter_provider: str = "",
) -> Dict[str, Any]:
    headers = {}
    if api_key and api_key != "dummy":
        headers["Authorization"] = f"Bearer {api_key}"

    is_openai = "api.openai.com" in base_url
    is_openrouter = "openrouter.ai" in base_url
    is_gpt_oss = "gpt-oss" in model.lower() or "harmony" in model.lower()
    provider_text = openrouter_provider.lower()
    needs_temp_1 = (
        "kimi" in model.lower()
        or "moonshot" in base_url.lower()
        or "moonshot" in provider_text
    )
    token_key = "max_completion_tokens" if is_openai else "max_tokens"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 1.0 if needs_temp_1 else temperature,
        token_key: max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if is_openrouter and openrouter_provider:
        payload["provider"] = {
            "order": [openrouter_provider],
            "allow_fallbacks": False,
        }
    if is_gpt_oss and not tools:
        payload.update(_get_gpt_oss_extra())

    async with session.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"chat failed ({resp.status}): {body}")
        result = await resp.json()

    usage = result.get("usage") or {}
    cached_tokens = _extract_cached_tokens(usage)
    if isinstance(result.get("usage"), dict):
        result["usage"]["cached_tokens"] = cached_tokens
    else:
        result["usage"] = {"cached_tokens": cached_tokens}
    return result


async def chat_completion_streaming(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    max_tokens: int,
    temperature: float = 0.0,
    openrouter_provider: str = "",
) -> ChatCompletionTimedResult:
    headers = {}
    if api_key and api_key != "dummy":
        headers["Authorization"] = f"Bearer {api_key}"

    is_openai = "api.openai.com" in base_url
    is_openrouter = "openrouter.ai" in base_url
    is_gpt_oss = "gpt-oss" in model.lower() or "harmony" in model.lower()
    provider_text = openrouter_provider.lower()
    needs_temp_1 = (
        "kimi" in model.lower()
        or "moonshot" in base_url.lower()
        or "moonshot" in provider_text
    )
    token_key = "max_completion_tokens" if is_openai else "max_tokens"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 1.0 if needs_temp_1 else temperature,
        token_key: max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
    if is_openrouter and openrouter_provider:
        payload["provider"] = {
            "order": [openrouter_provider],
            "allow_fallbacks": False,
        }
    if is_gpt_oss and not tools:
        payload.update(_get_gpt_oss_extra())

    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    t_last_token = t_start

    collected_content = ""
    collected_reasoning_content = ""
    collected_tool_calls: Dict[int, Dict[str, Any]] = {}
    finish_reason = None
    usage: Dict[str, Any] = {}

    async with session.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"chat failed ({resp.status}): {body}")

        async for raw_line in resp.content:
            text = raw_line.decode("utf-8").strip()
            if not text or not text.startswith("data:"):
                continue
            data_str = text[5:].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            now = time.perf_counter()
            choices = chunk.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}

                content_delta = delta.get("content")
                if content_delta:
                    if t_first_token is None:
                        t_first_token = now
                    t_last_token = now
                    collected_content += content_delta

                reasoning_delta = delta.get("reasoning_content")
                if isinstance(reasoning_delta, str) and reasoning_delta:
                    if t_first_token is None:
                        t_first_token = now
                    t_last_token = now
                    collected_reasoning_content += reasoning_delta

                if not reasoning_delta:
                    alt_reasoning_delta = delta.get("reasoning")
                    if isinstance(alt_reasoning_delta, str) and alt_reasoning_delta:
                        if t_first_token is None:
                            t_first_token = now
                        t_last_token = now
                        collected_reasoning_content += alt_reasoning_delta

                tool_call_deltas = delta.get("tool_calls") or []
                if tool_call_deltas:
                    if t_first_token is None:
                        t_first_token = now
                    t_last_token = now
                    for tc_delta in tool_call_deltas:
                        idx = int(tc_delta.get("index", 0))
                        if idx not in collected_tool_calls:
                            collected_tool_calls[idx] = {
                                "id": tc_delta.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.get("id"):
                            collected_tool_calls[idx]["id"] = tc_delta["id"]
                        fn = tc_delta.get("function") or {}
                        if fn.get("name"):
                            collected_tool_calls[idx]["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            collected_tool_calls[idx]["function"]["arguments"] += fn[
                                "arguments"
                            ]

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

            if chunk.get("usage"):
                usage = chunk["usage"]

    if t_first_token is None:
        t_first_token = t_start

    ttft = t_first_token - t_start
    decode_time = t_last_token - t_first_token

    assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": collected_content or None,
    }
    if collected_reasoning_content:
        assistant_message["reasoning_content"] = collected_reasoning_content
    if collected_tool_calls:
        assistant_message["tool_calls"] = [
            collected_tool_calls[i] for i in sorted(collected_tool_calls.keys())
        ]

    response_json = {
        "choices": [{"message": assistant_message, "finish_reason": finish_reason}],
        "usage": usage,
    }

    cached_tokens = _extract_cached_tokens(usage)
    if isinstance(response_json.get("usage"), dict):
        response_json["usage"]["cached_tokens"] = cached_tokens

    return ChatCompletionTimedResult(
        response_json=response_json,
        ttft_seconds=ttft,
        decode_seconds=decode_time,
        input_tokens=_to_int(usage.get("prompt_tokens", 0)),
        output_tokens=_to_int(usage.get("completion_tokens", 0)),
        cached_tokens=cached_tokens,
        is_streaming=True,
    )


async def _chat_with_fallback(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    max_tokens: int,
    temperature: float,
    openrouter_provider: str,
    use_streaming: bool,
    errors: List[str],
) -> ChatCompletionTimedResult:
    if use_streaming:
        try:
            return await chat_completion_streaming(
                session=session,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                openrouter_provider=openrouter_provider,
            )
        except Exception as exc:
            errors.append(f"streaming fallback: {exc}")

    t0 = time.perf_counter()
    response_json = await chat_completion(
        session=session,
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        openrouter_provider=openrouter_provider,
    )
    elapsed = time.perf_counter() - t0
    usage = response_json.get("usage") or {}
    cached_tokens = _to_int(usage.get("cached_tokens", 0))
    if cached_tokens == 0:
        cached_tokens = _extract_cached_tokens(usage)
    return ChatCompletionTimedResult(
        response_json=response_json,
        ttft_seconds=elapsed,
        decode_seconds=0.0,
        input_tokens=_to_int(usage.get("prompt_tokens", 0)),
        output_tokens=_to_int(usage.get("completion_tokens", 0)),
        cached_tokens=cached_tokens,
        is_streaming=False,
    )


__all__ = [
    "ChatCompletionTimedResult",
    "chat_completion",
    "chat_completion_streaming",
    "_chat_with_fallback",
    "_to_int",
    "_get_gpt_oss_extra",
    "_extract_cached_tokens",
    "_extract_thinking_content",
    "_load_schema_patches",
    "_SCHEMA_PATCHES",
    "_fix_tool_schema",
    "_clean_tool_args",
]
