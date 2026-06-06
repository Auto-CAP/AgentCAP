"""Drop-in aiohttp streaming wrapper for SWE-agent's litellm.completion call.

SWE-agent uses litellm.completion which defaults to non-streaming and (for
the openai/ prefix on local vLLM/SGLang) drops vendor fields like
delta.reasoning and prompt_tokens_details.cached_tokens. This wrapper
replaces that single call with a streaming aiohttp POST to /v1/chat/completions,
captures TTFT / per-token TPOT / token usage (visible/reasoning/cached) per
LLM call, writes them to SWEAGENT_STREAM_STATS_PATH (the agent_cap.agents
sweagent strategy already aggregates that file into the per-task RunResult),
and builds a litellm.ModelResponse so downstream code in
sweagent.agent.models is untouched.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp


def _o200k_count(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_harmony").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _strip_model_prefix(model: str) -> str:
    for p in ("openai/", "hosted_vllm/", "vllm/", "sglang/"):
        if model.startswith(p):
            return model[len(p):]
    return model


def _build_litellm_response(
    *,
    model: str,
    content: str,
    reasoning_content: str,
    tool_calls: List[Dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    cached_tokens: int,
    finish_reason: str,
):
    import litellm
    from litellm.types.utils import (
        ModelResponse,
        Choices,
        Message,
        Usage,
    )

    msg_kwargs: Dict[str, Any] = {"role": "assistant"}
    if content:
        msg_kwargs["content"] = content
    if reasoning_content:
        msg_kwargs["reasoning_content"] = reasoning_content
    if tool_calls:
        from litellm.types.utils import ChatCompletionMessageToolCall, Function as LiteLLMFunction
        ltc = []
        for tc in tool_calls:
            ltc.append(
                ChatCompletionMessageToolCall(
                    id=tc.get("id") or f"call_{len(ltc)}",
                    type="function",
                    function=LiteLLMFunction(
                        name=(tc.get("function") or {}).get("name", ""),
                        arguments=(tc.get("function") or {}).get("arguments", ""),
                    ),
                )
            )
        msg_kwargs["tool_calls"] = ltc

    message = Message(**msg_kwargs)
    choice = Choices(finish_reason=finish_reason or "stop", index=0, message=message)
    usage = Usage(
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens + reasoning_tokens),
        total_tokens=int(prompt_tokens + completion_tokens + reasoning_tokens),
    )
    try:
        from litellm.types.utils import (
            CompletionTokensDetailsWrapper as _CTD,
            PromptTokensDetailsWrapper as _PTD,
        )
        usage.completion_tokens_details = _CTD(reasoning_tokens=int(reasoning_tokens))
        if cached_tokens:
            usage.prompt_tokens_details = _PTD(cached_tokens=int(cached_tokens))
    except Exception:
        pass

    return ModelResponse(
        id=f"chatcmpl-aiohttp-{int(time.time()*1000)}",
        created=int(time.time()),
        model=model,
        object="chat.completion",
        choices=[choice],
        usage=usage,
    )


async def _astream(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_total: int,
) -> Dict[str, Any]:
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_call_frags: Dict[int, Dict[str, Any]] = {}
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    cached_tokens = 0
    sglang_style = False
    finish_reason = "stop"

    ttft_ms = 0.0
    itl_ms: List[float] = []
    t_start = time.perf_counter()
    last_ts = t_start

    timeout = aiohttp.ClientTimeout(total=timeout_total)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"chat failed ({resp.status}): {body[:500]}")
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
                        ctd = usage.get("completion_tokens_details") or {}
                        reasoning_tokens = int(ctd.get("reasoning_tokens") or 0)
                        top_r = usage.get("reasoning_tokens")
                        if top_r is not None and reasoning_tokens == 0:
                            reasoning_tokens = int(top_r or 0)
                            sglang_style = True
                        pd = usage.get("prompt_tokens_details") or {}
                        cached_tokens = int(pd.get("cached_tokens") or 0)
                        last_ts = ts
                        continue

                    choices = data.get("choices") or []
                    if not choices:
                        last_ts = ts
                        continue
                    ch = choices[0]
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    has_out = False

                    cp = delta.get("content")
                    if cp:
                        content_parts.append(cp)
                        has_out = True
                    rp = delta.get("reasoning_content") or delta.get("reasoning")
                    if rp:
                        reasoning_parts.append(rp)
                        has_out = True
                    tcs = delta.get("tool_calls")
                    if tcs:
                        for tc in tcs:
                            idx = tc.get("index", 0)
                            frag = tool_call_frags.setdefault(
                                idx, {"id": "", "function": {"name": "", "arguments": ""}}
                            )
                            if tc.get("id"):
                                frag["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                frag["function"]["name"] = fn["name"]
                                has_out = True
                            if fn.get("arguments"):
                                frag["function"]["arguments"] += fn["arguments"]
                                has_out = True

                    if ttft_ms == 0.0 and has_out:
                        ttft_ms = (ts - t_start) * 1000
                    elif ttft_ms > 0.0:
                        itl_ms.append((ts - last_ts) * 1000)
                    last_ts = ts

    content_text = "".join(content_parts)
    reasoning_text = "".join(reasoning_parts)
    for frag in tool_call_frags.values():
        args = frag["function"].get("arguments") or ""
        if args:
            try:
                json.loads(args)
            except json.JSONDecodeError as e:
                if "Extra data" in str(e) and getattr(e, "pos", None):
                    frag["function"]["arguments"] = args[: e.pos]
                else:
                    depth = 0
                    first_end = -1
                    for i, ch in enumerate(args):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                first_end = i + 1
                                break
                    if first_end > 0:
                        frag["function"]["arguments"] = args[:first_end]
    tool_calls = [
        {"id": frag["id"], "type": "function", "function": frag["function"]}
        for frag in tool_call_frags.values()
        if frag["function"]["name"]
    ]

    if reasoning_tokens == 0 and reasoning_text:
        reasoning_tokens = _o200k_count(reasoning_text)
        sglang_style = True
    if sglang_style and completion_tokens >= reasoning_tokens:
        total_out = completion_tokens
        visible = completion_tokens - reasoning_tokens
    else:
        visible = completion_tokens
        total_out = completion_tokens + reasoning_tokens

    tpot_ms = (sum(itl_ms) / total_out) if total_out > 0 else 0.0

    return {
        "content": content_text,
        "reasoning_content": reasoning_text,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens_visible": visible,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "total_output_tokens": total_out,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tpot_chunk_ms": (sum(itl_ms) / len(itl_ms)) if itl_ms else 0.0,
        "chunks": len(itl_ms) + 1,
    }


def completion_streaming(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    top_p: float = 1.0,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout_total: int = 1800,
    **_ignored: Any,
):
    if not api_base:
        raise RuntimeError(
            "agent_cap.sweagent_streaming requires api_base; pass it via "
            "--agent.model.api_base."
        )
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key not in ("dummy", ""):
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "model": _strip_model_prefix(model),
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
    if extra_body:
        for k, v in extra_body.items():
            payload[k] = v

    result = asyncio.run(_astream(url, headers, payload, timeout_total))

    sink = os.environ.get("SWEAGENT_STREAM_STATS_PATH")
    if sink:
        try:
            with open(sink, "a") as f:
                f.write(json.dumps({
                    "ttft_ms": result["ttft_ms"],
                    "tpot_ms": result["tpot_ms"],
                    "tpot_chunk_ms": result["tpot_chunk_ms"],
                    "chunks": result["chunks"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens_visible"],
                    "reasoning_tokens": result["reasoning_tokens"],
                    "cached_tokens": result["cached_tokens"],
                    "total_output_tokens": result["total_output_tokens"],
                }) + "\n")
        except Exception:
            pass

    return _build_litellm_response(
        model=model,
        content=result["content"],
        reasoning_content=result["reasoning_content"],
        tool_calls=result["tool_calls"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens_visible"],
        reasoning_tokens=result["reasoning_tokens"],
        cached_tokens=result["cached_tokens"],
        finish_reason=result["finish_reason"],
    )


__all__ = ["completion_streaming"]
