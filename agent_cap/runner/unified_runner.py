from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Sequence, TextIO, Union

import aiohttp
from agent_cap.runner.tool_backends import (
    MCPToolBackend,
    SWEBenchToolBackend,
    ToolBackend,
)

try:
    import psutil
except ImportError:
    psutil = None

from agent_cap.server.gpu_monitor import GPUMetricsSummary, GPUMonitor


@dataclass
class UnifiedTask:
    task_id: str
    task_name: str
    messages: List[Dict[str, Any]]
    eval_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UnifiedTask":
        task_id = raw.get("task_id") or raw.get("id") or ""
        task_name = raw.get("task_name") or raw.get("name") or ""
        messages = raw.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        eval_config = raw.get("eval_config")
        if not isinstance(eval_config, dict):
            eval_config = None
        return cls(
            task_id=str(task_id),
            task_name=str(task_name),
            messages=[m for m in messages if isinstance(m, dict)],
            eval_config=eval_config,
        )


@dataclass
class UnifiedConfig:
    model_name: str
    serving_engine: str
    base_url: str
    dataset: str
    mcp_server_url: str
    backend: str = "mcp"
    swebench_runtime: str = "docker"
    api_key: str = "dummy"
    api_provider: str = ""
    openrouter_provider_pin: str = ""
    openrouter_provider: str = ""
    is_local: bool = True
    precision: str = "bfloat16"
    max_turns: int = 15
    max_tokens: int = 8192
    temperature: float = 0.0
    enabled_tools: Sequence[str] = field(default_factory=list)
    output_root: Path = Path("results")
    use_streaming: bool = True


@dataclass
class ChatCompletionTimedResult:
    response_json: Dict[str, Any]
    ttft_seconds: float
    decode_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    is_streaming: bool = False


@dataclass
class ExampleResult:
    example_index: int
    task_id: str
    task_name: str
    total_input_tokens: int
    total_output_tokens: int
    tool_call_count: int
    num_requests: int
    e2e_latency_s: float
    avg_input_tokens_per_request: float
    avg_output_tokens_per_request: float
    max_input_tokens_per_request: int
    total_prefill_time_s: float
    total_decode_time_s: float
    output_text: str
    total_cached_tokens: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class UnifiedRunResult:
    output_dir: Path
    suffix: str
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    example_results: List[ExampleResult]


def _run_command(args: Sequence[str], timeout_s: float = 2.0) -> str:
    try:
        result = subprocess.run(
            list(args),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def collect_hardware_info() -> Dict[str, Any]:
    gpu_type = "unknown"
    num_gpus = 0
    cpu_type = "unknown"
    num_cpus = int(os.cpu_count() or 0)

    gpu_name_raw = _run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        timeout_s=3.0,
    )
    if gpu_name_raw:
        lines = [line.strip() for line in gpu_name_raw.splitlines() if line.strip()]
        if lines:
            gpu_type = lines[0]

    gpu_list_raw = _run_command(["nvidia-smi", "--list-gpus"], timeout_s=3.0)
    if gpu_list_raw:
        num_gpus = len([line for line in gpu_list_raw.splitlines() if line.strip()])

    lscpu_raw = _run_command(["lscpu"], timeout_s=3.0)
    if lscpu_raw:
        for line in lscpu_raw.splitlines():
            if line.startswith("Model name:"):
                cpu_type = line.split(":", 1)[1].strip() if ":" in line else cpu_type
                break

    return {
        "gpu_type": gpu_type,
        "num_gpus": num_gpus,
        "cpu_type": cpu_type,
        "num_cpus": num_cpus,
    }


def flatten_tool_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("content"), list):
            return flatten_tool_payload(payload["content"])
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(payload, list):
        parts: List[str] = []
        for item in payload:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return str(payload)


def count_tool_calls(messages: Sequence[Dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            total += len(message.get("tool_calls") or [])
    return total


def extract_final_assistant_text(messages: Sequence[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message.get("content", ""))
    return ""


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
    if is_gpt_oss:
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
    if is_gpt_oss:
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


async def run_single_example(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    task: UnifiedTask,
    tools: List[Dict[str, Any]],
    backend: ToolBackend,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    openrouter_provider: str,
    example_index: int,
    request_details_file: IO[str],
    use_streaming: bool = True,
    traj_dir: Optional[Path] = None,
) -> ExampleResult:
    messages = [dict(m) for m in task.messages]
    tool_schemas: Dict[str, Dict[str, Any]] = {}
    for t in tools:
        fn = t.get("function", {})
        if fn.get("name"):
            tool_schemas[fn["name"]] = fn.get("parameters", {})
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    total_prefill_time = 0.0
    total_decode_time = 0.0
    request_index = 0
    per_request_input_tokens: List[int] = []
    errors: List[str] = []
    task_dir: Optional[Path] = None
    if traj_dir:
        task_dir = traj_dir / f"task_{example_index:03d}"
        task_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    executed_turns = 0

    for turn_index in range(max_turns):
        executed_turns = turn_index + 1
        if task_dir is not None:
            request_data = {
                "turn": turn_index,
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "messages": messages,
                "tools_count": len(tools) if tools else 0,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            (task_dir / f"turn_{turn_index:03d}_request.json").write_text(
                json.dumps(request_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        try:
            timed = await _chat_with_fallback(
                session=session,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                openrouter_provider=openrouter_provider,
                use_streaming=use_streaming,
                errors=errors,
            )
        except Exception as exc:
            errors.append(f"llm_request_failed: {exc}")
            break

        result = timed.response_json
        usage = result.get("usage") or {}
        in_tok = _to_int(usage.get("prompt_tokens", timed.input_tokens))
        out_tok = _to_int(usage.get("completion_tokens", timed.output_tokens))
        cached_tok = _to_int(usage.get("cached_tokens", timed.cached_tokens))
        if cached_tok == 0:
            cached_tok = _extract_cached_tokens(usage)

        total_input_tokens += in_tok
        total_output_tokens += out_tok
        total_cached_tokens += cached_tok
        total_prefill_time += timed.ttft_seconds
        total_decode_time += timed.decode_seconds
        per_request_input_tokens.append(in_tok)

        choices = result.get("choices") or []
        if not choices:
            errors.append("model returned empty choices")
            break
        assistant = choices[0].get("message") or {}
        if "role" not in assistant:
            assistant["role"] = "assistant"
        tool_calls = assistant.get("tool_calls") or []

        if task_dir is not None:
            response_data = {
                "turn": turn_index,
                "timestamp": datetime.now().isoformat(),
                "raw_response": result,
                "thinking_content": _extract_thinking_content(assistant),
                "content": assistant.get("content", ""),
                "tool_calls": tool_calls,
                "usage": usage,
                "cached_tokens": cached_tok,
                "ttft_s": timed.ttft_seconds if timed.is_streaming else 0.0,
                "decode_s": timed.decode_seconds if timed.is_streaming else 0.0,
            }
            (task_dir / f"turn_{turn_index:03d}_response.json").write_text(
                json.dumps(response_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        tpot = timed.decode_seconds / out_tok if out_tok > 0 else 0.0
        throughput = out_tok / timed.decode_seconds if timed.decode_seconds > 0 else 0.0
        request_detail = {
            "example_index": example_index,
            "request_index": request_index,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cached_tokens": cached_tok,
            "prefill_time_s": round(timed.ttft_seconds, 6),
            "decode_time_s": round(timed.decode_seconds, 6),
            "tpot_s": round(tpot, 6),
            "output_throughput_tok_s": round(throughput, 2),
            "has_tool_calls": len(tool_calls) > 0,
            "num_tool_calls": len(tool_calls),
        }
        request_details_file.write(
            json.dumps(request_detail, ensure_ascii=False) + "\n"
        )
        request_details_file.flush()
        request_index += 1

        messages.append(assistant)

        if not tool_calls:
            break

        tool_calls_log: List[Dict[str, Any]] = []
        tool_results_log: List[Dict[str, Any]] = []
        for tc in tool_calls:
            function = tc.get("function") or {}
            name = str(function.get("name", ""))
            raw_args = function.get("arguments", "{}")
            try:
                parsed_args = (
                    json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                )
            except json.JSONDecodeError:
                parsed_args = {}
            cleaned_args = parsed_args
            if isinstance(parsed_args, dict):
                cleaned_args = _clean_tool_args(name, parsed_args, tool_schemas)

            tool_calls_log.append(
                {
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "arguments_raw": raw_args,
                    "arguments_parsed": parsed_args,
                    "arguments_cleaned": cleaned_args,
                }
            )

            tool_start = time.perf_counter()
            is_error = False
            error_msg = None
            try:
                tool_result = await backend.call_tool(name, cleaned_args)
            except Exception as exc:
                is_error = True
                error_msg = str(exc)
                errors.append(f"{name}: {exc}")
                tool_result = [{"type": "text", "text": f"ERROR: {exc}"}]

            tool_latency = time.perf_counter() - tool_start
            flattened_result = flatten_tool_payload(tool_result)
            tool_results_log.append(
                {
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "success": not is_error,
                    "result": flattened_result,
                    "error": error_msg,
                    "latency_s": round(tool_latency, 4),
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": flattened_result,
                }
            )

        if task_dir is not None:
            (task_dir / f"turn_{turn_index:03d}_tool_calls.json").write_text(
                json.dumps(tool_calls_log, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (task_dir / f"turn_{turn_index:03d}_tool_results.json").write_text(
                json.dumps(tool_results_log, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    elapsed = time.perf_counter() - start
    final_response = extract_final_assistant_text(messages)
    tool_call_count = count_tool_calls(messages)

    if task_dir is not None:
        summary = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "model": model,
            "total_turns": executed_turns,
            "total_tool_calls": tool_call_count,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_tokens": total_cached_tokens,
            "e2e_latency_s": elapsed,
            "errors": errors,
            "final_output": final_response,
            "tools": tools,
        }
        (task_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return ExampleResult(
        example_index=example_index,
        task_id=task.task_id,
        task_name=task.task_name,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cached_tokens=total_cached_tokens,
        tool_call_count=tool_call_count,
        num_requests=request_index,
        e2e_latency_s=elapsed,
        avg_input_tokens_per_request=(
            total_input_tokens / request_index if request_index > 0 else 0.0
        ),
        avg_output_tokens_per_request=(
            total_output_tokens / request_index if request_index > 0 else 0.0
        ),
        max_input_tokens_per_request=(
            max(per_request_input_tokens) if per_request_input_tokens else 0
        ),
        total_prefill_time_s=total_prefill_time,
        total_decode_time_s=total_decode_time,
        output_text=final_response,
        errors=errors,
    )


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    frac = rank - low
    return sorted_vals[low] * (1.0 - frac) + sorted_vals[high] * frac


def _read_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def compute_aggregated_metrics(
    results: Sequence[ExampleResult],
    gpu_stats: GPUMetricsSummary,
    wall_time: float,
    hw_info: Dict[str, Any],
    detailed_results_path: Optional[Path] = None,
    cpu_samples: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    request_rows = _read_jsonl(detailed_results_path)

    e2e_latencies = [r.e2e_latency_s for r in results]
    prefill_times = [float(r.get("prefill_time_s", 0.0)) for r in request_rows]
    decode_times = [float(r.get("decode_time_s", 0.0)) for r in request_rows]
    tpot_vals = [
        float(r.get("tpot_s", 0.0)) for r in request_rows if r.get("tpot_s") is not None
    ]
    throughput_vals = [
        float(r.get("output_throughput_tok_s", 0.0))
        for r in request_rows
        if r.get("output_throughput_tok_s") is not None
    ]

    total_examples = len(results)
    total_requests = sum(r.num_requests for r in results)
    total_input_tokens = sum(r.total_input_tokens for r in results)
    total_output_tokens = sum(r.total_output_tokens for r in results)
    total_cached_tokens = sum(r.total_cached_tokens for r in results)
    total_tool_calls = sum(r.tool_call_count for r in results)
    error_examples = sum(1 for r in results if r.errors)
    completed_examples = total_examples - error_examples

    cpu_vals = [float(v) for v in (cpu_samples or [])]

    return {
        "performance": {
            "e2e_s": wall_time,
            "avg_e2e_latency_s": _safe_mean(e2e_latencies),
            "p50_e2e_latency_s": _percentile(e2e_latencies, 50),
            "p99_e2e_latency_s": _percentile(e2e_latencies, 99),
            "examples_per_second": (
                (total_examples / wall_time)
                if wall_time > 0 and total_examples > 0
                else 0.0
            ),
            "ttft": _safe_mean(prefill_times),
            "p99_ttft": _percentile(prefill_times, 99),
            "tpot": _safe_mean(tpot_vals),
            "p99_tpot": _percentile(tpot_vals, 99),
            "decode_time_s": _safe_mean(decode_times),
            "p99_decode_time_s": _percentile(decode_times, 99),
            "output_throughput_tok_s": _safe_mean(throughput_vals),
        },
        "agentic": {
            "avg_total_input_tokens": _safe_mean(
                [float(r.total_input_tokens) for r in results]
            ),
            "avg_total_output_tokens": _safe_mean(
                [float(r.total_output_tokens) for r in results]
            ),
            "avg_tool_call_count": _safe_mean(
                [float(r.tool_call_count) for r in results]
            ),
            "avg_num_requests": _safe_mean([float(r.num_requests) for r in results]),
            "avg_input_tokens_per_request": _safe_mean(
                [float(r.avg_input_tokens_per_request) for r in results]
            ),
            "avg_output_tokens_per_request": _safe_mean(
                [float(r.avg_output_tokens_per_request) for r in results]
            ),
            "avg_max_input_tokens_per_request": _safe_mean(
                [float(r.max_input_tokens_per_request) for r in results]
            ),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_tokens": total_cached_tokens,
            "avg_cache_hit_rate": _safe_mean(
                [
                    (r.total_cached_tokens / r.total_input_tokens)
                    for r in results
                    if r.total_input_tokens > 0
                ]
            ),
            "total_requests": total_requests,
            "total_tool_calls": total_tool_calls,
        },
        "quality": {
            "total_examples": total_examples,
            "completed": completed_examples,
            "errors": error_examples,
        },
        "hardware": {
            "gpu_type": hw_info.get("gpu_type", "unknown"),
            "num_gpus": int(hw_info.get("num_gpus", 0) or 0),
            "avg_gpu_utilization_pct": float(gpu_stats.avg_gpu_util_pct),
            "peak_gpu_memory_used_mb": float(gpu_stats.peak_memory_used_mb),
            "avg_cpu_utilization_pct": _safe_mean(cpu_vals),
        },
    }


async def run_experiment(
    config: UnifiedConfig,
    tasks: Sequence[Union[UnifiedTask, Dict[str, Any]]],
) -> UnifiedRunResult:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output_root) / config.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{config.dataset}_{timestamp}"
    traj_dir = out_dir / f"trajectories_{suffix}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / f"metadata_{suffix}.json"
    metrics_path = out_dir / f"metrics_{suffix}.json"
    detailed_results_path = out_dir / f"detailed_results_{suffix}.jsonl"
    output_data_path = out_dir / f"output_data_{suffix}.jsonl"

    normalized_tasks: List[UnifiedTask] = [
        task if isinstance(task, UnifiedTask) else UnifiedTask.from_dict(task)
        for task in tasks
    ]

    hw = collect_hardware_info()
    metadata = {
        "hardware": hw,
        "model_config": {
            "model_name": config.model_name,
            "precision": config.precision,
        },
        "system_environment": {
            "inference_engine": config.serving_engine,
            "base_url": config.base_url,
            "is_local": config.is_local,
            "mcp_server_url": config.mcp_server_url,
            "backend": config.backend,
            "swebench_runtime": config.swebench_runtime,
            "dataset": config.dataset,
            "num_examples": len(normalized_tasks),
            "max_turns": config.max_turns,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "timestamp": timestamp,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    gpu_monitor = GPUMonitor(interval=1.0)
    gpu_monitor.start()

    all_example_results: List[ExampleResult] = []
    cpu_samples: List[float] = []
    wall_start = time.perf_counter()
    wall_end = wall_start

    try:
        async with aiohttp.ClientSession() as session:
            backend_name = (config.backend or "mcp").lower().replace("_", "-")
            runtime_name = (
                (config.swebench_runtime or "docker").lower().replace("_", "-")
            )
            if backend_name == "mcp":
                backend: ToolBackend = MCPToolBackend(
                    session=session,
                    mcp_server_url=config.mcp_server_url,
                    enabled_tools=config.enabled_tools,
                )
            elif backend_name in ("swebench", "swe-bench"):
                if runtime_name not in ("docker", "modal"):
                    raise ValueError(
                        f"Unknown swebench runtime: {config.swebench_runtime}. Supported: docker, modal"
                    )
                backend = SWEBenchToolBackend(runtime=runtime_name)
            elif backend_name in ("swebench-docker", "swe-bench-docker"):
                backend = SWEBenchToolBackend(runtime="docker")
            elif backend_name in ("swebench-modal", "swe-bench-modal"):
                backend = SWEBenchToolBackend(runtime="modal")
            else:
                raise ValueError(
                    f"Unknown backend: {config.backend}. Supported: mcp, swebench-docker, swebench-modal"
                )

            if backend_name == "mcp":
                setup_ok = await backend.setup({})
                if not setup_ok:
                    raise RuntimeError("MCP backend setup failed")
                tools = await backend.list_tools()
            else:
                tools = []

            with (
                detailed_results_path.open("w", encoding="utf-8") as req_f,
                output_data_path.open("w", encoding="utf-8") as out_f,
            ):
                wall_start = time.perf_counter()
                for i, task in enumerate(normalized_tasks):
                    if backend_name != "mcp":
                        task_config = task.eval_config or {}
                        task_setup_ok = await backend.setup(task_config)
                        if not task_setup_ok:
                            result = ExampleResult(
                                example_index=i,
                                task_id=task.task_id,
                                task_name=task.task_name,
                                total_input_tokens=0,
                                total_output_tokens=0,
                                total_cached_tokens=0,
                                tool_call_count=0,
                                num_requests=0,
                                e2e_latency_s=0.0,
                                avg_input_tokens_per_request=0.0,
                                avg_output_tokens_per_request=0.0,
                                max_input_tokens_per_request=0,
                                total_prefill_time_s=0.0,
                                total_decode_time_s=0.0,
                                output_text="",
                                errors=["backend_setup_failed"],
                            )
                            output_data = {
                                "index": i,
                                "task_id": result.task_id,
                                "input_tokens": result.total_input_tokens,
                                "output_tokens": result.total_output_tokens,
                                "tool_call_count": result.tool_call_count,
                                "num_requests": result.num_requests,
                                "e2e_latency_s": result.e2e_latency_s,
                                "output_text": result.output_text,
                                "errors": result.errors,
                            }
                            out_f.write(
                                json.dumps(output_data, ensure_ascii=False) + "\n"
                            )
                            out_f.flush()
                            all_example_results.append(result)
                            if psutil is not None:
                                try:
                                    cpu_samples.append(
                                        float(psutil.cpu_percent(interval=None))
                                    )
                                except Exception:
                                    pass
                            await backend.teardown()
                            continue
                        tools = await backend.list_tools()

                    openrouter_provider = (
                        config.openrouter_provider or config.openrouter_provider_pin
                    )
                    try:
                        result = await run_single_example(
                            session=session,
                            base_url=config.base_url,
                            api_key=config.api_key,
                            model=config.model_name,
                            task=task,
                            tools=tools,
                            backend=backend,
                            max_turns=config.max_turns,
                            max_tokens=config.max_tokens,
                            temperature=config.temperature,
                            openrouter_provider=openrouter_provider,
                            example_index=i,
                            request_details_file=req_f,
                            use_streaming=config.use_streaming,
                            traj_dir=traj_dir,
                        )
                    finally:
                        if backend_name != "mcp":
                            try:
                                patch = await backend.get_patch()
                                if patch:
                                    task_dir = traj_dir / f"task_{i:03d}"
                                    task_dir.mkdir(parents=True, exist_ok=True)
                                    (task_dir / "patch.diff").write_text(
                                        patch, encoding="utf-8"
                                    )
                            except Exception:
                                pass
                            await backend.teardown()

                    output_data = {
                        "index": i,
                        "task_id": result.task_id,
                        "input_tokens": result.total_input_tokens,
                        "output_tokens": result.total_output_tokens,
                        "tool_call_count": result.tool_call_count,
                        "num_requests": result.num_requests,
                        "e2e_latency_s": result.e2e_latency_s,
                        "output_text": result.output_text,
                        "errors": result.errors,
                    }
                    out_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    out_f.flush()
                    all_example_results.append(result)

                    if psutil is not None:
                        try:
                            cpu_samples.append(float(psutil.cpu_percent(interval=None)))
                        except Exception:
                            pass
                wall_end = time.perf_counter()

            if backend_name == "mcp":
                await backend.teardown()
    finally:
        gpu_stats = gpu_monitor.stop()

    metrics = compute_aggregated_metrics(
        results=all_example_results,
        gpu_stats=gpu_stats,
        wall_time=max(0.0, wall_end - wall_start),
        hw_info=hw,
        detailed_results_path=detailed_results_path,
        cpu_samples=cpu_samples,
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    return UnifiedRunResult(
        output_dir=out_dir,
        suffix=suffix,
        metadata=metadata,
        metrics=metrics,
        example_results=all_example_results,
    )


def _load_dataset_tasks(dataset_name: str, limit: int = 0) -> List[UnifiedTask]:
    name = dataset_name.lower().replace("-", "_").replace(" ", "_")

    if name in ("mcp_atlas", "mcpatlas"):
        from datasets import load_dataset as hf_load

        ds = hf_load("ScaleAI/mcp-atlas", split="train")
        tasks: List[UnifiedTask] = []
        for ex in ds:
            if not isinstance(ex, dict):
                continue
            tasks.append(
                UnifiedTask(
                    task_id=ex.get("TASK", ""),
                    task_name=ex.get("PROMPT", "")[:80],
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a factual, tool-aware assistant connected to a variety of tools. Use the available tools to answer the user query. Do not ask the user for clarification; fully complete the task using the information provided.",
                        },
                        {"role": "user", "content": ex.get("PROMPT", "")},
                    ],
                )
            )
        if limit > 0:
            tasks = tasks[:limit]
        return tasks

    if name in ("swebench_pro", "swe_bench_pro", "swebench", "swe_bench"):
        from datasets import load_dataset as hf_load

        ds = hf_load("ScaleAI/SWE-bench_Pro", split="test")
        tasks = []
        for ex in ds:
            if not isinstance(ex, dict):
                continue
            instance_id = ex.get("instance_id", "")
            repo = ex.get("repo", "")
            problem = ex.get("problem_statement", "")
            prompt = (
                f"You are working on {repo}. Fix this issue:\n\n{problem}\n\n"
                "Use the available tools (read_file, write_file, run_shell, search_code) "
                "to explore the codebase and make the fix."
            )
            tasks.append(
                UnifiedTask(
                    task_id=instance_id,
                    task_name=f"[{repo}] {problem[:60]}",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a software engineer. Use the available tools to read files, search code, run commands, and write fixes. Make minimal changes to fix the issue.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    eval_config={
                        "instance_id": instance_id,
                        "repo": repo,
                        "base_commit": ex.get("base_commit", ""),
                        "dockerhub_tag": ex.get("dockerhub_tag", ""),
                        "test_patch": ex.get("test_patch", ""),
                        "fail_to_pass": ex.get("fail_to_pass", ""),
                        "patch": ex.get("patch", ""),
                    },
                )
            )
        if limit > 0:
            tasks = tasks[:limit]
        return tasks

    raise ValueError(
        f"Unknown dataset: {dataset_name}. Supported: mcp-atlas, swe-bench-pro"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AgentCAP Unified Single-Agent Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All CLI args:
  python -m agent_cap.runner.unified_runner \\
    --model-name Qwen/Qwen3-30B-A3B \\
    --dataset mcp-atlas \\
    --base-url http://localhost:30000 \\
    --mcp-server-url http://localhost:1984

  # With config file (CLI overrides):
  python -m agent_cap.runner.unified_runner \\
    --config configs/experiment.yaml \\
    --base-url http://localhost:30000

  # API model via OpenRouter:
  python -m agent_cap.runner.unified_runner \\
    --model-name deepseek-chat \\
    --dataset mcp-atlas \\
    --base-url https://openrouter.ai/api \\
    --api-key sk-xxx \\
    --no-local
""",
    )
    parser.add_argument(
        "--config", type=str, help="YAML config file path. CLI args override."
    )

    parser.add_argument("--model-name", type=str, help="Model name/ID")
    parser.add_argument(
        "--dataset", type=str, help="Dataset name (e.g. mcp-atlas, swe-bench-pro)"
    )
    parser.add_argument(
        "--base-url", type=str, help="LLM server URL (e.g. http://localhost:30000)"
    )
    parser.add_argument(
        "--mcp-server-url",
        type=str,
        help="MCP tool server URL (e.g. http://localhost:1984)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mcp",
        help="Tool backend: 'mcp' or 'swebench-docker' or 'swebench-modal'",
    )
    parser.add_argument(
        "--swebench-runtime",
        type=str,
        default="docker",
        help="SWE-bench runtime: 'docker' or 'modal'",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for LLM server (default: dummy)",
    )
    parser.add_argument(
        "--serving-engine",
        type=str,
        default=None,
        help="Serving engine name (default: sglang)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Model precision (default: bfloat16)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Max tool-call turns per example (default: 15)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens per LLM call (default: 8192)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Limit number of tasks (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: results)",
    )
    parser.add_argument(
        "--enabled-tools",
        nargs="*",
        default=None,
        help="Optional allowlist of tool names",
    )
    parser.add_argument(
        "--no-local", action="store_true", help="Mark model as API (not self-hosted)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming (use non-stream API)",
    )
    parser.add_argument(
        "--openrouter-provider",
        type=str,
        default=None,
        help="Pin to specific OpenRouter provider (e.g. 'Moonshot AI', 'Z.AI', 'Minimax')",
    )

    args = parser.parse_args()

    file_cfg: Dict[str, Any] = {}
    if args.config:
        try:
            import yaml
        except Exception as exc:
            parser.error(f"failed to import yaml for --config: {exc}")
        with open(args.config, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}

    def resolve(cli_val: Any, yaml_key: str, default: Any = None) -> Any:
        if cli_val is not None:
            return cli_val
        if yaml_key in file_cfg:
            return file_cfg[yaml_key]
        return default

    resolved_openrouter_provider = resolve(
        args.openrouter_provider, "openrouter_provider", ""
    )
    if not resolved_openrouter_provider:
        resolved_openrouter_provider = resolve(None, "openrouter_provider_pin", "")

    config = UnifiedConfig(
        model_name=resolve(args.model_name, "model_name", ""),
        serving_engine=resolve(args.serving_engine, "serving_engine", "sglang"),
        base_url=resolve(args.base_url, "base_url", "http://localhost:30000"),
        dataset=resolve(args.dataset, "dataset", ""),
        mcp_server_url=resolve(
            args.mcp_server_url, "mcp_server_url", "http://localhost:1984"
        ),
        backend=resolve(args.backend, "backend", "mcp"),
        swebench_runtime=resolve(args.swebench_runtime, "swebench_runtime", "docker"),
        api_key=resolve(args.api_key, "api_key", "dummy"),
        api_provider=resolve(None, "api_provider", ""),
        openrouter_provider_pin=resolve(
            None, "openrouter_provider_pin", resolved_openrouter_provider
        ),
        openrouter_provider=resolved_openrouter_provider,
        is_local=False if args.no_local else resolve(None, "is_local", True),
        precision=resolve(args.precision, "precision", "bfloat16"),
        max_turns=resolve(args.max_turns, "max_turns", 15),
        max_tokens=resolve(args.max_tokens, "max_tokens", 8192),
        temperature=resolve(args.temperature, "temperature", 0.0),
        enabled_tools=resolve(args.enabled_tools, "enabled_tools", []),
        output_root=Path(
            resolve(
                args.output_dir, "output_dir", file_cfg.get("output_root", "results")
            )
        ),
        use_streaming=(
            False if args.no_streaming else resolve(None, "use_streaming", True)
        ),
    )

    if not config.model_name:
        parser.error("--model-name is required (or set model_name in config file)")
    if not config.dataset:
        parser.error("--dataset is required (or set dataset in config file)")

    tasks = _load_dataset_tasks(config.dataset, resolve(args.num_tasks, "num_tasks", 0))

    import asyncio

    result = asyncio.run(run_experiment(config, tasks))
    print(f"\nDone. Output directory: {result.output_dir}")
    print(f"  metadata:         metadata_{result.suffix}.json")
    print(f"  metrics:          metrics_{result.suffix}.json")
    print(f"  detailed_results: detailed_results_{result.suffix}.jsonl")
    print(f"  output_data:      output_data_{result.suffix}.jsonl")


__all__ = [
    "UnifiedTask",
    "UnifiedConfig",
    "ChatCompletionTimedResult",
    "ExampleResult",
    "UnifiedRunResult",
    "collect_hardware_info",
    "flatten_tool_payload",
    "count_tool_calls",
    "extract_final_assistant_text",
    "chat_completion",
    "chat_completion_streaming",
    "run_single_example",
    "compute_aggregated_metrics",
    "run_experiment",
    "_load_dataset_tasks",
    "main",
]


if __name__ == "__main__":
    main()
