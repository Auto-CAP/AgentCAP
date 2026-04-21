from __future__ import annotations

import asyncio
import json
import os
import subprocess
import importlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Sequence, TextIO, Union

import aiohttp
from agent_cap.runner.llm_client import (
    ChatCompletionTimedResult,
    chat_completion,
    chat_completion_streaming,
    _chat_with_fallback,
    _to_int,
    _extract_cached_tokens,
    _extract_thinking_content,
    _fix_tool_schema,
    _clean_tool_args,
    _SCHEMA_PATCHES,
)
from agent_cap.runner.tool_backends import (
    MathPythonToolBackend,
    MedAgentBenchToolBackend,
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
    enabled_tools: Optional[List[str]] = None

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
    parallel_tool_calls: bool = True,
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
                parallel_tool_calls=parallel_tool_calls,
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
        # print(f'[TOOL CALLS (unified_runner.py line 294)]: {tool_calls}')

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
            if not name:
                continue
            raw_args = function.get("arguments", "{}")
            try:
                parsed_args = (
                    json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                )
            except json.JSONDecodeError:
                if isinstance(raw_args, str) and raw_args.strip():
                    parsed_args = {"raw": raw_args}
                else:
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

            if name == "python" and isinstance(raw_args, str) and not raw_args.strip():
                errors.append("python: skipped empty tool call")
                continue

            tool_start = time.perf_counter()
            is_error = False
            error_msg = None
            tool_result: Any = [{"type": "text", "text": "ERROR: unknown tool failure"}]
            for _retry in range(3):
                try:
                    tool_result = await backend.call_tool(name, cleaned_args)
                    break
                except Exception as exc:
                    if _retry < 2 and (
                        "Cannot connect" in str(exc) or "500" in str(exc)
                    ):
                        await asyncio.sleep(2**_retry)
                        continue
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
            "acc": round(
                sum(float(r.eval_score) for r in results if r.eval_score is not None)
                / max(sum(1 for r in results if r.eval_score is not None), 1),
                3,
            )
            if any(r.eval_score is not None for r in results)
            else None,
            "task_coverage": round(
                sum(1 for r in results if r.eval_passed)
                / max(sum(1 for r in results if r.eval_passed is not None), 1),
                3,
            )
            if any(r.eval_passed is not None for r in results)
            else None,
            "evaluator": next(
                (
                    r.eval_details.get("evaluator")
                    for r in results
                    if isinstance(r.eval_details, dict)
                    and r.eval_details.get("evaluator")
                ),
                None,
            ),
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
    detailed_results_path = out_dir / f"detailed-results_{suffix}.jsonl"
    output_data_path = out_dir / f"output-data_{suffix}.jsonl"

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
            elif backend_name in ("math-python", "math_python"):
                backend = MathPythonToolBackend()
            elif backend_name in ("swebench-k8s", "swe-bench-k8s"):
                backend = SWEBenchToolBackend(runtime="k8s")
            elif backend_name in ("medagentbench", "med-agent-bench"):
                fhir_url = (
                    config.mcp_server_url
                    if config.mcp_server_url and "fhir" in config.mcp_server_url
                    else None
                )
                backend = MedAgentBenchToolBackend(
                    session=session,
                    fhir_base_url=fhir_url,
                )
            else:
                raise ValueError(
                    "Unknown backend: "
                    f"{config.backend}. Supported: mcp, swebench-docker, swebench-modal, swebench-k8s, medagentbench"
                )

            if backend_name == "mcp":
                setup_ok = await backend.setup({})
                if not setup_ok:
                    raise RuntimeError("MCP backend setup failed")
                all_tools = await backend.list_tools()
            else:
                all_tools = []

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

                    # Per-task tool filtering (MCP-ATLAS exposes 10-25 tools per task)
                    if task.enabled_tools and all_tools:
                        allowed = set(
                            t if isinstance(t, str) else t.get("name", "")
                            for t in task.enabled_tools
                        )
                        tools = [
                            t
                            for t in all_tools
                            if t.get("function", {}).get("name", "") in allowed
                        ]
                    else:
                        tools = all_tools

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
                            parallel_tool_calls=True,
                        )
                    finally:
                        if backend_name != "mcp":
                            try:
                                print(f"[task {i}] getting patch...", flush=True)
                                patch = await backend.get_patch()
                                task_dir = traj_dir / f"task_{i:03d}"
                                task_dir.mkdir(parents=True, exist_ok=True)
                                if patch:
                                    (task_dir / "patch.diff").write_text(
                                        patch, encoding="utf-8"
                                    )
                                    print(
                                        f"[task {i}] patch saved ({len(patch)} chars)",
                                        flush=True,
                                    )
                                else:
                                    print(f"[task {i}] no patch generated", flush=True)
                                # Run tests and save results
                                print(f"[task {i}] running tests...", flush=True)
                                test_result = backend.run_tests(timeout=300)
                                print(
                                    f"[task {i}] test_result: {test_result}", flush=True
                                )
                                (task_dir / "test_result.json").write_text(
                                    json.dumps(
                                        test_result, ensure_ascii=False, indent=2
                                    ),
                                    encoding="utf-8",
                                )
                                print(
                                    f"[task {i}] TEST: "
                                    f"{'PASS' if test_result.get('passed') else 'FAIL'}",
                                    flush=True,
                                )
                            except Exception as exc:
                                print(
                                    f"[task {i}] post-loop error: {type(exc).__name__}: {exc}",
                                    flush=True,
                                )
                                import traceback

                                traceback.print_exc()
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

    if name in ("imo_answerbench", "imoanswerbench"):
        from datasets import load_dataset as hf_load

        ds = hf_load("Hwilner/imo-answerbench", split="train")
        tasks: List[UnifiedTask] = []
        for i, ex in enumerate(ds):
            if not isinstance(ex, dict):
                continue

            problem_id = ex.get("Problem ID", f"imo-{i}")
            problem = str(ex.get("Problem", "")).strip()
            if not problem:
                continue

            short_answer = str(ex.get("Short Answer", "")).strip()
            category = str(ex.get("Category", "math"))

            prompt = (
                f"{problem}\n\n"
                r"Solve this step by step. Place your final numerical answer inside \boxed{}"
            )

            tasks.append(
                UnifiedTask(
                    task_id=str(problem_id),
                    task_name=problem[:80],
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert mathematical problem solver with "
                                "access to a Python execution tool. Use it for calculations "
                                "when useful. Put your final answer in \\boxed{}."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    eval_config={
                        "type": "math_verify",
                        "expected": short_answer,
                        "category": category,
                    },
                )
            )

        if limit > 0:
            tasks = tasks[:limit]
        return tasks

    if name in ("tau2_banking", "tau2_bench_banking", "tau2"):
        tau2_root = Path(__file__).parent.parent.parent / "third_party" / "tau2-bench"
        tau2_src = tau2_root / "src"
        if not tau2_src.exists():
            raise FileNotFoundError(
                f"tau2-bench source not found at {tau2_src}. "
                "Clone https://github.com/sierra-research/tau2-bench into third_party/tau2-bench"
            )

        import sys

        tau2_src_str = str(tau2_src.resolve())
        if tau2_src_str not in sys.path:
            sys.path.insert(0, tau2_src_str)

        tasks_dir = (
            tau2_root / "data" / "tau2" / "domains" / "banking_knowledge" / "tasks"
        )
        if not tasks_dir.exists():
            raise FileNotFoundError(
                f"tau2 banking tasks directory not found: {tasks_dir}"
            )

        tasks: List[UnifiedTask] = []
        for task_file in sorted(tasks_dir.glob("task_*.json")):
            raw = json.loads(task_file.read_text(encoding="utf-8"))
            task_id = str(raw.get("id", task_file.stem))

            description = raw.get("description") or {}
            user_scenario = raw.get("user_scenario") or {}
            instructions = str(user_scenario.get("instructions", ""))
            persona = str(user_scenario.get("persona") or "")
            prompt = (
                "You are speaking to a banking customer. Continue the conversation and "
                "resolve their request using tools.\n\n"
                f"Customer persona:\n{persona}\n\n"
                f"Customer scenario instructions:\n{instructions}"
            )

            task_name = str(description.get("purpose") or task_id)
            evaluation_criteria = raw.get("evaluation_criteria") or {}
            reward_basis = evaluation_criteria.get("reward_basis") or []
            eval_cfg = {
                "type": "tau2",
                "domain": "banking_knowledge",
                "retrieval_variant": "bm25",
                "tau2_task": raw,
                "required_documents": raw.get("required_documents") or [],
                "reward_basis": list(reward_basis),
            }

            tasks.append(
                UnifiedTask(
                    task_id=task_id,
                    task_name=str(task_name),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a banking customer support agent. Use tools to "
                                "assist the customer safely and accurately."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    eval_config=eval_cfg,
                )
            )

        if limit > 0:
            tasks = tasks[:limit]
        return tasks

    if name in (
        "mcp_atlas",
        "mcpatlas",
        "mcp_atlas_finance",
        "mcp_atlas_general",
    ):
        from datasets import load_dataset as hf_load

        ds = hf_load("ScaleAI/mcp-atlas", split="train")

        # Servers available without paid API keys
        _FREE_SERVERS = {
            "arxiv",
            "brave-search",
            "calculator",
            "cli-mcp-server",
            "clinicaltrialsgov-mcp-server",
            "context7",
            "ddg-search",
            "desktop-commander",
            "fetch",
            "filesystem",
            "git",
            "github",
            "mcp-code-executor",
            "mcp-server-code-runner",
            "memory",
            "met-museum",
            "open-library",
            "osm-mcp-server",
            "pubmed",
            "weather",
            "whois",
            "wikipedia",
        }

        def _get_server(tool_name: str) -> str:
            for s in sorted(_FREE_SERVERS, key=len, reverse=True):
                if tool_name.startswith(s):
                    return s
            return tool_name.split("_")[0]

        subset_filter: Optional[str] = None
        if "finance" in name:
            subset_filter = "finance"
        elif "general" in name:
            subset_filter = "general"

        domain_task_ids: Optional[set[str]] = None
        if subset_filter:
            classifications_path = (
                Path(__file__).parent.parent.parent
                / "results"
                / "mcpatlas_domain_classifications.jsonl"
            )
            if not classifications_path.exists():
                raise FileNotFoundError(
                    f"Domain classifications not found at {classifications_path}. "
                    "Run scripts/classify_mcpatlas_domains.py first."
                )

            finance_ids: set[str] = set()
            all_ids: set[str] = set()
            with classifications_path.open("r", encoding="utf-8") as cf:
                for line in cf:
                    text = line.strip()
                    if not text:
                        continue
                    entry = json.loads(text)
                    task_id = str(entry.get("task_id", ""))
                    if not task_id:
                        continue
                    all_ids.add(task_id)
                    if "finance" in (entry.get("domains") or []):
                        finance_ids.add(task_id)

            if subset_filter == "finance":
                domain_task_ids = finance_ids
            else:
                # general = NOT in finance set
                domain_task_ids = all_ids - finance_ids

        tasks: List[UnifiedTask] = []
        for ex in ds:
            if not isinstance(ex, dict):
                continue

            # Filter: only keep tasks whose tools are all free
            raw_tools = ex.get("ENABLED_TOOLS", "[]")
            enabled = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
            tool_names = [
                t if isinstance(t, str) else t.get("name", "") for t in enabled
            ]
            servers = {_get_server(t) for t in tool_names if t}
            if not servers.issubset(_FREE_SERVERS):
                continue

            task_id = str(ex.get("TASK", ""))
            if domain_task_ids is not None and task_id not in domain_task_ids:
                continue
            claims = ex.get("GTFA_CLAIMS", [])
            if isinstance(claims, str):
                try:
                    claims = json.loads(claims)
                except (json.JSONDecodeError, ValueError):
                    try:
                        import ast

                        claims = ast.literal_eval(claims)
                    except (ValueError, SyntaxError):
                        claims = [claims] if claims.strip() else []
            if not isinstance(claims, list):
                claims = [str(claims)] if str(claims).strip() else []
            claims = [str(c).strip() for c in claims if str(c).strip()]
            tasks.append(
                UnifiedTask(
                    task_id=task_id,
                    task_name=ex.get("PROMPT", "")[:80],
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a factual, tool-aware assistant connected to a variety of tools. Use the available tools to answer the user query. Do not ask the user for clarification; fully complete the task using the information provided.",
                        },
                        {"role": "user", "content": ex.get("PROMPT", "")},
                    ],
                    enabled_tools=tool_names,
                    eval_config={"type": "gtfa", "gtfa_claims": claims},
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
                        "type": "swebench",
                        "instance_id": instance_id,
                        "repo": repo,
                        "base_commit": ex.get("base_commit", ""),
                        "dockerhub_tag": ex.get("dockerhub_tag", ""),
                        "test_patch": ex.get("test_patch", ""),
                        "fail_to_pass": ex.get("fail_to_pass", ""),
                        "FAIL_TO_PASS": ex.get("fail_to_pass", ""),
                        "patch": ex.get("patch", ""),
                    },
                )
            )
        if limit > 0:
            tasks = tasks[:limit]
        return tasks

    if name in ("medagentbench", "med_agent_bench"):
        data_path = (
            Path(__file__).parent.parent.parent
            / "third_party"
            / "MedAgentBench"
            / "data"
            / "medagentbench"
            / "test_data_v1.json"
        )
        funcs_path = (
            Path(__file__).parent.parent.parent
            / "third_party"
            / "MedAgentBench"
            / "data"
            / "medagentbench"
            / "funcs_v1.json"
        )
        if not data_path.exists() or not funcs_path.exists():
            raise FileNotFoundError(
                "MedAgentBench files not found at "
                f"{data_path} and/or {funcs_path}. "
                "Clone https://github.com/stanfordmlgroup/MedAgentBench into third_party/"
            )

        MedAgentBench_prompt = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke. Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Question: {question}"""

        with data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        with funcs_path.open("r", encoding="utf-8") as f:
            funcs = json.load(f)

        functions_json = json.dumps(funcs)
        tasks = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("id", ""))
            instruction = str(item.get("instruction", ""))
            context = str(item.get("context", ""))
            prompt_with_placeholder = MedAgentBench_prompt.format(
                api_base="{api_base}",
                functions=functions_json,
                context=context,
                question=instruction,
            )
            tasks.append(
                UnifiedTask(
                    task_id=task_id,
                    task_name=f"medagentbench_{task_id}",
                    messages=[
                        {"role": "user", "content": prompt_with_placeholder},
                    ],
                    eval_config={
                        "type": "medagentbench",
                        "task_data": item,
                        "task_index": idx,
                        "prompt_with_placeholder": prompt_with_placeholder,
                    },
                )
            )
        if limit > 0:
            tasks = tasks[:limit]
        return tasks

    if name in ("financebench", "finance-bench", "finance_bench"):
        data_path = (
            Path(__file__).parent.parent.parent
            / "third_party"
            / "financebench"
            / "data"
            / "financebench_open_source.jsonl"
        )
        if not data_path.exists():
            raise FileNotFoundError(
                f"FinanceBench data not found at {data_path}. "
                "Download from https://github.com/patronus-ai/financebench"
            )

        FINANCEBENCH_EXECUTOR_PROMPT = """You are a financial analyst with access to SEC filing documents.

You have the following tools:
- search_document(doc_name, query): Search pre-extracted text from an SEC filing for relevant information.
- calculate(expression): Evaluate a mathematical expression (e.g. "(1577 / 32136) * 100").

Use these tools to find the information needed to answer the question. When you have the answer, output it clearly prefixed with "ANSWER:".

Document: {doc_name}
Company: {company}

Question: {question}"""

        tasks = []
        with data_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                fb_id = item.get("financebench_id", f"fb_{idx:04d}")
                company = item.get("company", "")
                doc_name = item.get("doc_name", "")
                question = item.get("question", "")
                answer = item.get("answer", "")
                question_type = item.get("question_type", "")

                executor_prompt = FINANCEBENCH_EXECUTOR_PROMPT.format(
                    doc_name=doc_name,
                    company=company,
                    question=question,
                )
                tasks.append(
                    UnifiedTask(
                        task_id=fb_id,
                        task_name=f"financebench_{fb_id}",
                        messages=[{"role": "user", "content": executor_prompt}],
                        eval_config={
                            "type": "financebench",
                            "financebench_id": fb_id,
                            "company": company,
                            "doc_name": doc_name,
                            "question": question,
                            "answer": answer,
                            "question_type": question_type,
                            "justification": item.get("justification", ""),
                            "evidence": item.get("evidence", []),
                        },
                    )
                )
        if limit > 0:
            tasks = tasks[:limit]
        return tasks

    raise ValueError(
        "Unknown dataset: "
        f"{dataset_name}. Supported: imo-answerbench, mcp-atlas, swe-bench-pro, medagentbench, tau2-banking, financebench"
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
        help="Tool backend: 'mcp', 'math-python', 'swebench-docker', or 'swebench-modal'",
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
    print(f"  detailed_results: detailed-results_{result.suffix}.jsonl")
    print(f"  output_data:      output-data_{result.suffix}.jsonl")


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
