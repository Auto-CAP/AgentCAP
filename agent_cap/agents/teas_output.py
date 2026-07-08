"""TEAS-format output writer — runs automatically at the end of a sweagent run.

Emits into the run's output dir (suffix = <dataset>_<YYYYMMDD_HHMMSS>):
    metadata_<suffix>.json          hardware / model_config / system_environment
    metrics_<suffix>.json           performance / agentic / quality / hardware
    detailed-results_<suffix>.jsonl one row per LLM request (from stream_stats)
    output-data_<suffix>.jsonl      one row per task

Schemas match the agentic/vastai reference runs in
TEAS_Development_Results_Private (e.g. .../swe-bench-lite/b300x4/...) exactly.

Hardware/scenario facts the CLI cannot know come from env vars (set by the
launch script): TEAS_GPU_TYPE, TEAS_NUM_GPUS, TEAS_TP, TEAS_ENGINE,
TEAS_ENGINE_VERSION, TEAS_MODEL_NAME, TEAS_PRECISION, TEAS_CPU_TYPE,
TEAS_NUM_CPUS, TEAS_BACKEND. Missing values degrade to "unknown"/0 rather
than failing the run.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _pct(vals: List[float], p: float) -> float:
    vals = sorted(vals)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] * (1 - (k - lo)) + vals[hi] * (k - lo)


def per_request_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for i, task_dir in enumerate(sorted(run_dir.glob("task_*"))):
        ss = task_dir / "stream_stats.jsonl"
        if not ss.exists():
            continue
        for j, line in enumerate(ss.read_text().splitlines()):
            if not line.strip():
                continue
            s = json.loads(line)
            out_tok = int(s.get("total_output_tokens") or 0)
            tpot_s = float(s.get("tpot_ms") or 0.0) / 1000.0
            decode_s = tpot_s * out_tok
            rows.append({
                "example_index": i,
                "request_index": j,
                "input_tokens": int(s.get("prompt_tokens") or 0),
                "output_tokens": out_tok,
                "completion_tokens": int(s.get("completion_tokens") or 0),
                "reasoning_tokens": int(s.get("reasoning_tokens") or 0),
                "cached_tokens": int(s.get("cached_tokens") or 0),
                "prefill_time_s": float(s.get("ttft_ms") or 0.0) / 1000.0,
                "decode_time_s": decode_s,
                "tpot_s": tpot_s,
                "output_throughput_tok_s": (out_tok / decode_s) if decode_s > 0 else 0.0,
                "has_tool_calls": False,   # sweagent actions are text commands
                "num_tool_calls": 0,
            })
    return rows


def per_task_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        u = r.get("total_usage") or {}
        out.append({
            "index": i,
            "task_id": r.get("task_id"),
            "input_tokens": u.get("input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
            "reasoning_tokens": u.get("reasoning_tokens", 0),
            "cached_tokens": u.get("cached_tokens", 0),
            "tool_call_count": r.get("tool_calls", 0),
            "num_requests": u.get("requests", 0),
            "e2e_latency_s": r.get("e2e_latency_s", 0.0),
            "ttft_ms": r.get("ttft_ms", 0.0),
            "tpot_ms": r.get("tpot_ms", 0.0),
            "eval_passed": r.get("eval_passed"),
            "eval_score": r.get("eval_score"),
            "output_text": r.get("output_text", ""),
            "errors": r.get("errors", []),
        })
    return out


def write_teas_outputs(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    dataset: str,
    wall_time_s: float,
    model_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Optional[Path]:
    """Write the four TEAS files. Returns the metrics path (or None if no rows)."""
    if not rows:
        return None
    env = os.environ.get
    engine = env("TEAS_ENGINE", "unknown")
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{dataset}_{ts}"

    req_rows = per_request_rows(out_dir)
    e2e = [r.get("e2e_latency_s", 0.0) for r in rows]
    ttfts = [r["prefill_time_s"] for r in req_rows if r["prefill_time_s"] > 0]
    tpots = [r["tpot_s"] for r in req_rows if r["tpot_s"] > 0]
    n = len(rows)
    passed = sum(1 for r in rows if r.get("eval_passed"))
    usage = lambda k: sum((r.get("total_usage") or {}).get(k, 0) for r in rows)  # noqa: E731
    tot_in, tot_out = usage("input_tokens"), usage("output_tokens")
    tot_cached, tot_reqs = usage("cached_tokens"), usage("requests")
    tot_tools = sum(r.get("tool_calls", 0) for r in rows)

    metrics = {
        "performance": {
            "total_wall_time_min": round(wall_time_s / 60.0, 2),
            "avg_e2e_latency_s": round(sum(e2e) / max(n, 1), 2),
            "p50_e2e_latency_s": round(_pct(e2e, 0.5), 2),
            "p99_e2e_latency_s": round(_pct(e2e, 0.99), 2),
            "ttft": round(sum(ttfts) / max(len(ttfts), 1), 6),
            "p99_ttft": round(_pct(ttfts, 0.99), 6),
            "tpot": round(sum(tpots) / max(len(tpots), 1), 6),
            "p99_tpot": round(_pct(tpots, 0.99), 6),
        },
        "agentic": {
            "avg_total_input_tokens": round(tot_in / max(n, 1), 2),
            "avg_total_output_tokens": round(tot_out / max(n, 1), 2),
            "avg_tool_call_count": round(tot_tools / max(n, 1), 2),
            "avg_num_requests": round(tot_reqs / max(n, 1), 2),
            "avg_input_tokens_per_request": round(tot_in / max(tot_reqs, 1), 2),
            "avg_output_tokens_per_request": round(tot_out / max(tot_reqs, 1), 2),
            "total_input_tokens": tot_in,
            "total_output_tokens": tot_out,
            "total_cached_tokens": tot_cached,
            "total_requests": tot_reqs,
            "total_tool_calls": tot_tools,
        },
        "quality": {
            "acc": round(passed / max(n, 1), 4),
            "total_examples": n,
            "passed": passed,
        },
        "hardware": {
            "gpu_type": env("TEAS_GPU_TYPE", "unknown"),
            "num_gpus": int(env("TEAS_NUM_GPUS", "0")),
            f"{engine}_version": env("TEAS_ENGINE_VERSION", "unknown"),
            "streaming": True,
        },
    }

    metadata = {
        "hardware": {
            "gpu_type": env("TEAS_GPU_TYPE", "unknown"),
            "num_gpus": int(env("TEAS_NUM_GPUS", "0")),
            "cpu_type": env("TEAS_CPU_TYPE", "unknown"),
            "num_cpus": int(env("TEAS_NUM_CPUS", "0")),
        },
        "model_config": {
            "model_name": env("TEAS_MODEL_NAME", model_name or "unknown"),
            "precision": env("TEAS_PRECISION", "unknown"),
        },
        "system_environment": {
            "inference_engine": engine,
            "base_url": "http://localhost:8000/v1",
            "is_local": True,
            "backend": env("TEAS_BACKEND", "swebench-k8s"),
            "dataset": dataset,
            "num_examples": n,
            "tensor_parallel_size": int(env("TEAS_TP", env("TEAS_NUM_GPUS", "0"))),
            "streaming": True,
            "timestamp": ts,
            "agentcap_strategy": "sweagent",
            "sweagent_streaming_patch": "AGENTCAP_STREAMING_PATCH_APPLIED",
            "reasoning_parser": "gpt-oss" if engine == "sglang" else "openai_gptoss",
            "tool_call_parser": "gpt-oss" if engine == "sglang" else "openai",
        },
    }

    (out_dir / f"metrics_{suffix}.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / f"metadata_{suffix}.json").write_text(json.dumps(metadata, indent=2))
    with (out_dir / f"detailed-results_{suffix}.jsonl").open("w") as f:
        for row in req_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / f"output-data_{suffix}.jsonl").open("w") as f:
        for row in per_task_rows(rows):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_dir / f"metrics_{suffix}.json"
