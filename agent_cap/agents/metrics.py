"""Aggregate metrics for the generic multi-agent CLI.

The legacy dataset runners write a ``metrics_<dataset>_<timestamp>.json`` file
next to their JSONL outputs. The generic ``agent_cap.agents`` CLI uses a
slightly different per-task row schema, so this module computes the same broad
sections (performance, agentic, quality, hardware) from those rows.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Mapping, Sequence


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _percentile(values: Sequence[float], p: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    frac = k - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _run_command(args: Sequence[str], timeout_s: float = 3.0) -> str:
    try:
        proc = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def collect_hardware_info() -> Dict[str, Any]:
    """Best-effort static hardware info, intentionally dependency-light."""

    gpu_type = "unknown"
    num_gpus = 0
    cpu_type = "unknown"
    num_cpus = int(os.cpu_count() or 0)

    gpu_name_raw = _run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    )
    if gpu_name_raw:
        lines = [line.strip() for line in gpu_name_raw.splitlines() if line.strip()]
        if lines:
            gpu_type = lines[0]
            num_gpus = len(lines)

    if num_gpus == 0:
        gpu_list_raw = _run_command(["nvidia-smi", "--list-gpus"])
        if gpu_list_raw:
            num_gpus = len([line for line in gpu_list_raw.splitlines() if line.strip()])

    lscpu_raw = _run_command(["lscpu"])
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


def _usage(row: Mapping[str, Any]) -> Mapping[str, Any]:
    usage = row.get("total_usage")
    return usage if isinstance(usage, Mapping) else {}


def _per_role_usage(row: Mapping[str, Any]) -> Mapping[str, Any]:
    usage = row.get("per_role_usage")
    return usage if isinstance(usage, Mapping) else {}


def aggregate_agent_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    wall_time_s: float,
    evaluator_name: str | None = None,
    hardware_info: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Aggregate ``RunResult.to_dict()``-style rows into metrics JSON.

    The output mirrors the legacy runner's high-level sections while adding a
    ``per_role`` block for multi-agent role accounting.
    """

    n = len(rows)
    e2e_latencies = [_safe_float(r.get("e2e_latency_s")) for r in rows]
    ttft_ms = [_safe_float(r.get("ttft_ms")) for r in rows if _safe_float(r.get("ttft_ms")) > 0]
    tpot_ms = [_safe_float(r.get("tpot_ms")) for r in rows if _safe_float(r.get("tpot_ms")) > 0]
    latency_ms = [_safe_float(r.get("latency_ms")) for r in rows if _safe_float(r.get("latency_ms")) > 0]

    input_tokens_by_task = [_safe_int(_usage(r).get("input_tokens")) for r in rows]
    output_tokens_by_task = [_safe_int(_usage(r).get("output_tokens")) for r in rows]
    cached_tokens_by_task = [_safe_int(_usage(r).get("cached_tokens")) for r in rows]
    request_counts = [_safe_int(_usage(r).get("requests"), _safe_int(r.get("num_turns"))) for r in rows]
    tool_counts = [_safe_int(r.get("tool_calls")) for r in rows]
    max_input_tokens_per_task = [
        max(
            [_safe_int(usage.get("input_tokens")) for usage in _per_role_usage(r).values() if isinstance(usage, Mapping)]
            or [_safe_int(_usage(r).get("input_tokens"))]
        )
        for r in rows
    ]

    total_input_tokens = sum(input_tokens_by_task)
    total_output_tokens = sum(output_tokens_by_task)
    total_cached_tokens = sum(cached_tokens_by_task)
    total_requests = sum(request_counts)
    total_tool_calls = sum(tool_counts)
    error_examples = sum(1 for r in rows if r.get("errors"))

    scores = [
        _safe_float(r.get("eval_score"))
        for r in rows
        if r.get("eval_score") is not None
    ]
    passes = [
        bool(r.get("eval_passed"))
        for r in rows
        if r.get("eval_passed") is not None
    ]

    inferred_evaluator = evaluator_name
    if inferred_evaluator is None:
        for row in rows:
            details = row.get("eval_details")
            if isinstance(details, Mapping) and details.get("evaluator"):
                inferred_evaluator = str(details["evaluator"])
                break

    per_role: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for role, usage in _per_role_usage(row).items():
            if not isinstance(usage, Mapping):
                continue
            slot = per_role.setdefault(
                str(role),
                {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_reasoning_tokens": 0,
                    "total_cached_tokens": 0,
                    "total_requests": 0,
                },
            )
            slot["total_input_tokens"] += _safe_int(usage.get("input_tokens"))
            slot["total_output_tokens"] += _safe_int(usage.get("output_tokens"))
            slot["total_completion_tokens"] += _safe_int(usage.get("completion_tokens"))
            slot["total_reasoning_tokens"] += _safe_int(usage.get("reasoning_tokens"))
            slot["total_cached_tokens"] += _safe_int(usage.get("cached_tokens"))
            slot["total_requests"] += _safe_int(usage.get("requests"))

    for slot in per_role.values():
        slot["avg_input_tokens_per_task"] = slot["total_input_tokens"] / n if n else 0.0
        slot["avg_output_tokens_per_task"] = slot["total_output_tokens"] / n if n else 0.0
        slot["avg_requests_per_task"] = slot["total_requests"] / n if n else 0.0

    hw = dict(hardware_info or collect_hardware_info())

    return {
        "status": "completed",
        "performance": {
            "e2e_s": float(wall_time_s),
            "avg_e2e_latency_s": _safe_mean(e2e_latencies),
            "p50_e2e_latency_s": _percentile(e2e_latencies, 50),
            "p99_e2e_latency_s": _percentile(e2e_latencies, 99),
            "examples_per_second": (n / wall_time_s) if wall_time_s > 0 and n else 0.0,
            "ttft": _safe_mean([v / 1000.0 for v in ttft_ms]),
            "p99_ttft": _percentile([v / 1000.0 for v in ttft_ms], 99),
            "ttft_ms": _safe_mean(ttft_ms),
            "p99_ttft_ms": _percentile(ttft_ms, 99),
            "tpot": _safe_mean([v / 1000.0 for v in tpot_ms]),
            "p99_tpot": _percentile([v / 1000.0 for v in tpot_ms], 99),
            "tpot_ms": _safe_mean(tpot_ms),
            "p99_tpot_ms": _percentile(tpot_ms, 99),
            "avg_latency_ms": _safe_mean(latency_ms),
            "p99_latency_ms": _percentile(latency_ms, 99),
        },
        "agentic": {
            "avg_total_input_tokens": _safe_mean([float(v) for v in input_tokens_by_task]),
            "avg_total_output_tokens": _safe_mean([float(v) for v in output_tokens_by_task]),
            "avg_tool_call_count": _safe_mean([float(v) for v in tool_counts]),
            "avg_num_requests": _safe_mean([float(v) for v in request_counts]),
            "avg_input_tokens_per_request": (
                total_input_tokens / total_requests if total_requests else 0.0
            ),
            "avg_output_tokens_per_request": (
                total_output_tokens / total_requests if total_requests else 0.0
            ),
            "avg_max_input_tokens_per_request": _safe_mean(
                [float(v) for v in max_input_tokens_per_task]
            ),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_tokens": total_cached_tokens,
            "avg_cache_hit_rate": _safe_mean([
                cached / inp for cached, inp in zip(cached_tokens_by_task, input_tokens_by_task)
                if inp > 0
            ]),
            "total_requests": total_requests,
            "total_tool_calls": total_tool_calls,
            "error_examples": error_examples,
            "completed_examples": n - error_examples,
            "per_role": per_role,
        },
        "quality": {
            "acc": round(sum(scores) / max(len(scores), 1), 3) if scores else None,
            "task_coverage": round(sum(1 for p in passes if p) / max(len(passes), 1), 3)
            if passes else None,
            "evaluator": inferred_evaluator,
        },
        "hardware": {
            "gpu_type": hw.get("gpu_type", "unknown"),
            "num_gpus": int(hw.get("num_gpus", 0) or 0),
            "cpu_type": hw.get("cpu_type", "unknown"),
            "num_cpus": int(hw.get("num_cpus", 0) or 0),
            # The generic CLI does not yet run live GPU/CPU samplers; keep the
            # legacy keys present but neutral so downstream readers do not fail.
            "avg_gpu_utilization_pct": float(hw.get("avg_gpu_utilization_pct", 0.0) or 0.0),
            "peak_gpu_memory_used_mb": float(hw.get("peak_gpu_memory_used_mb", 0.0) or 0.0),
            "avg_cpu_utilization_pct": float(hw.get("avg_cpu_utilization_pct", 0.0) or 0.0),
        },
    }
