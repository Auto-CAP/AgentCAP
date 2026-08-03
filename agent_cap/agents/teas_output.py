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
TEAS_NUM_CPUS, TEAS_BACKEND, TEAS_BASE_URL, TEAS_MCP_SERVER_URL,
TEAS_CONCURRENCY, TEAS_OBSERVED_MAX_CONCURRENCY,
TEAS_REASONING_PARSER, TEAS_TOOL_CALL_PARSER, TEAS_MCP_TOOL_COUNT,
TEAS_MCP_ENABLED_SERVERS, TEAS_MCP_ATLAS_COMMIT, TEAS_MCP_DATA_DIR,
and TEAS_TASK_INDICES_SHA256.
Missing values degrade to stable defaults rather than failing the run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agent_cap.agents.metrics import resolve_rows_evaluator
from agent_cap.utils.precision import resolve_precision, normalize_model_id


class TeasOutputError(ValueError):
    """Raised when source artifacts cannot support a trustworthy TEAS export."""


_SECRET_PATTERN = re.compile(
    rb"(?:ghp_[A-Za-z0-9]{20,}|sk-or-v1-[A-Za-z0-9_-]{20,}|"
    rb"BSA[A-Za-z0-9_-]{20,})"
)


def _pct(vals: List[float], p: float) -> float:
    vals = sorted(vals)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] * (1 - (k - lo)) + vals[hi] * (k - lo)


def _swebench_reports(
    run_dir: Path,
    run_ids: Optional[Sequence[str]] = None,
) -> Dict[str, bool]:
    """Load and deduplicate official SWE-bench ``report.json`` results."""

    reports_root = run_dir / "logs" / "run_evaluation"
    if not reports_root.exists():
        return {}
    selected = set(run_ids or ())
    reports: Dict[str, bool] = {}
    for report_path in reports_root.glob("*/*/*/report.json"):
        relative = report_path.relative_to(reports_root)
        report_run_id = relative.parts[0]
        if selected and report_run_id not in selected:
            continue
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TeasOutputError(f"invalid SWE-bench report: {report_path}") from exc
        if not isinstance(payload, dict) or len(payload) != 1:
            raise TeasOutputError(f"unexpected SWE-bench report shape: {report_path}")
        instance_id, body = next(iter(payload.items()))
        if not isinstance(instance_id, str) or not isinstance(body, dict):
            raise TeasOutputError(f"unexpected SWE-bench report shape: {report_path}")
        resolved = body.get("resolved")
        if not isinstance(resolved, bool):
            raise TeasOutputError(f"missing boolean resolved field: {report_path}")
        previous = reports.get(instance_id)
        if previous is not None and previous != resolved:
            raise TeasOutputError(
                f"conflicting SWE-bench reports for {instance_id}: "
                f"{previous} versus {resolved}"
            )
        reports[instance_id] = resolved
    if selected:
        missing_run_ids = selected - {
            path.name for path in reports_root.iterdir() if path.is_dir()
        }
        if missing_run_ids:
            raise TeasOutputError(
                f"SWE-bench run_id directories not found: {sorted(missing_run_ids)}"
            )
    return reports


def _swebench_harness_failures(run_dir: Path) -> set[str]:
    """Return task IDs that the official harness explicitly failed to evaluate."""

    failures: set[str] = set()
    pattern = re.compile(r"EvaluationError:\s+([^:\s]+):")
    for log_path in run_dir.glob("swebench_eval*.log"):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise TeasOutputError(f"cannot read SWE-bench evaluation log: {log_path}") from exc
        failures.update(pattern.findall(text))
    return failures


def _has_explicit_swebench_failure(
    row: Mapping[str, Any],
    harness_failures: set[str],
) -> bool:
    """Distinguish a measured failure from a result that was never evaluated."""

    task_id = row.get("task_id")
    details = row.get("eval_details")
    if not isinstance(task_id, str) or not isinstance(details, Mapping):
        return False
    empty_patch = (
        details.get("evaluator") == "swebench"
        and details.get("instance_id") == task_id
        and details.get("error") == "empty patch"
        and not row.get("output_text")
    )
    generation_error = (
        details.get("evaluator") == "skipped"
        and details.get("reason") == "task errored"
        and bool(row.get("errors"))
    )
    return empty_patch or generation_error or task_id in harness_failures


def resolve_quality(
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
    *,
    swebench_run_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Resolve exactly the three quality fields accepted by TEAS.

    SWE-bench generation rows commonly contain placeholder false/zero values
    before the official harness runs. Those values are never trusted. Official
    report files take precedence; fully provenance-tagged inline evaluator
    results are the only fallback.
    """

    total = len(rows)
    if total == 0:
        raise TeasOutputError("cannot compute quality for an empty run")

    if dataset.startswith("swe-bench"):
        reports = _swebench_reports(run_dir, swebench_run_ids)
        task_ids = {
            row.get("task_id")
            for row in rows
            if isinstance(row.get("task_id"), str)
        }
        unknown_reports = set(reports) - task_ids
        if unknown_reports:
            raise TeasOutputError(
                "SWE-bench reports do not belong to this run: "
                f"{sorted(unknown_reports)[:5]}"
            )
        if reports:
            harness_failures = _swebench_harness_failures(run_dir)
            missing_outcomes = [
                row.get("task_id")
                for row in rows
                if row.get("task_id") not in reports
                and not _has_explicit_swebench_failure(row, harness_failures)
            ]
            if missing_outcomes:
                raise TeasOutputError(
                    "SWE-bench evaluation is incomplete: tasks have neither an "
                    "official report nor an explicit empty-patch, generation, or "
                    f"harness failure: {missing_outcomes[:5]}"
                )
            passed = sum(reports.values())
        else:
            verified_rows = []
            for row in rows:
                details = row.get("eval_details")
                verified = (
                    isinstance(row.get("eval_passed"), bool)
                    and isinstance(details, Mapping)
                    and details.get("evaluator") == "swebench"
                    and details.get("instance_id") == row.get("task_id")
                )
                if verified:
                    verified_rows.append(row)
            if len(verified_rows) != total:
                raise TeasOutputError(
                    "SWE-bench quality is not available: no official reports "
                    "and not every result row has verified evaluator provenance. "
                    "Refusing to turn placeholder zeros into TEAS metrics."
                )
            passed = sum(bool(row["eval_passed"]) for row in verified_rows)
    else:
        invalid = [
            row.get("task_id")
            for row in rows
            if not isinstance(row.get("eval_passed"), bool)
        ]
        if invalid:
            raise TeasOutputError(
                f"missing boolean eval_passed values for {len(invalid)} tasks"
            )
        passed = sum(bool(row["eval_passed"]) for row in rows)

    return {
        "acc": round(passed / total, 4),
        "total_examples": total,
        "passed": passed,
    }


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


def turn_stat_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-request rows from compact turn_stats (non-sweagent strategies)."""
    out = []
    for i, r in enumerate(rows):
        for j, s in enumerate(r.get("turn_stats") or []):
            out_tok = int(s.get("output_tokens") or 0)
            decode_s = float(s.get("decode_s") or 0.0)
            tpot_s = (decode_s / out_tok) if out_tok > 0 else 0.0
            out.append({
                "example_index": i,
                "request_index": j,
                "input_tokens": int(s.get("input_tokens") or 0),
                "output_tokens": out_tok,
                "completion_tokens": int(s.get("completion_tokens") or 0),
                "reasoning_tokens": int(s.get("reasoning_tokens") or 0),
                "cached_tokens": int(s.get("cached_tokens") or 0),
                "prefill_time_s": float(s.get("ttft_s") or 0.0),
                "decode_time_s": decode_s,
                "tpot_s": tpot_s,
                "output_throughput_tok_s": (out_tok / decode_s) if decode_s > 0 else 0.0,
                "has_tool_calls": int(s.get("num_tool_calls") or 0) > 0,
                "num_tool_calls": int(s.get("num_tool_calls") or 0),
            })
    return out


def write_teas_outputs(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    dataset: str,
    wall_time_s: float,
    model_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    *,
    destination_dir: Optional[Path] = None,
    include_output_data: bool = True,
    swebench_run_ids: Optional[Sequence[str]] = None,
    metadata_template: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    """Write the four TEAS files. Returns the metrics path (or None if no rows)."""
    if not rows:
        return None
    out_dir = Path(out_dir)
    destination_dir = Path(destination_dir or out_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.get
    engine = env("TEAS_ENGINE", "unknown")
    engine_version = env("TEAS_ENGINE_VERSION", "unknown")
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{dataset}_{ts}"
    is_mcp = dataset.startswith("mcp-atlas")

    req_rows = per_request_rows(out_dir)
    if not req_rows:
        req_rows = turn_stat_rows(rows)
    e2e = [r.get("e2e_latency_s", 0.0) for r in rows]
    ttfts = [r["prefill_time_s"] for r in req_rows if r["prefill_time_s"] > 0]
    tpots = [r["tpot_s"] for r in req_rows if r["tpot_s"] > 0]
    n = len(rows)
    quality = resolve_quality(
        out_dir,
        rows,
        dataset,
        swebench_run_ids=swebench_run_ids,
    )
    usage = lambda k: sum((r.get("total_usage") or {}).get(k, 0) for r in rows)  # noqa: E731
    tot_in, tot_out = usage("input_tokens"), usage("output_tokens")
    tot_cached, tot_reqs = usage("cached_tokens"), usage("requests")
    tot_tools = sum(r.get("tool_calls", 0) for r in rows)

    perf = {
        "total_wall_time_min": round(wall_time_s / 60.0, 2),
        "avg_e2e_latency_s": round(sum(e2e) / max(n, 1), 2),
        "p50_e2e_latency_s": round(_pct(e2e, 0.5), 2),
        "p99_e2e_latency_s": round(_pct(e2e, 0.99), 2),
        "ttft": round(sum(ttfts) / max(len(ttfts), 1), 6),
        "p99_ttft": round(_pct(ttfts, 0.99), 6),
        "tpot": round(sum(tpots) / max(len(tpots), 1), 6),
        "p99_tpot": round(_pct(tpots, 0.99), 6),
    }
    metrics = {
        "performance": perf,
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
        "quality": quality,
        "hardware": {
            "gpu_type": env("TEAS_GPU_TYPE", "unknown"),
            "num_gpus": int(env("TEAS_NUM_GPUS", "0")),
            f"{engine}_version": engine_version,
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
            "model_name": normalize_model_id(env("TEAS_MODEL_NAME", model_name) or "unknown"),
            "precision": (env("TEAS_PRECISION")
                          or resolve_precision(env("TEAS_MODEL_NAME", model_name))
                          or "unknown"),
        },
        "system_environment": ({
            "inference_engine": engine,
            "inference_engine_version": env("TEAS_ENGINE_VERSION", "unknown"),
            "base_url": env("TEAS_BASE_URL", "http://localhost:8000/v1"),
            "is_local": True,
            "backend": env("TEAS_BACKEND", "mcp-atlas-localmcp"),
            "dataset": dataset,
            "num_examples": n,
            "tensor_parallel_size": int(env("TEAS_TP", env("TEAS_NUM_GPUS", "0"))),
            "max_model_len": int(env("TEAS_MAX_MODEL_LEN", "131072")),
            "streaming": True,
            "timestamp": ts,
            "agentcap_strategy": "single",
            "concurrency": int(env("TEAS_CONCURRENCY", "1")),
            "observed_max_concurrency": int(
                env("TEAS_OBSERVED_MAX_CONCURRENCY", "0")
            ),
            "mcp_server_url": env("TEAS_MCP_SERVER_URL", "http://localhost:1984"),
            "mcp_tool_count": int(env("TEAS_MCP_TOOL_COUNT", "0")),
            "mcp_enabled_servers": [
                item
                for item in env("TEAS_MCP_ENABLED_SERVERS", "").split(",")
                if item
            ],
            "mcp_atlas_commit": env("TEAS_MCP_ATLAS_COMMIT", "unknown"),
            "mcp_data_dir": env("TEAS_MCP_DATA_DIR", "unknown"),
            "task_indices_sha256": env("TEAS_TASK_INDICES_SHA256", "unknown"),
            "reasoning_parser": env(
                "TEAS_REASONING_PARSER",
                "gpt-oss" if engine == "sglang" else "openai_gptoss",
            ),
            "tool_call_parser": env(
                "TEAS_TOOL_CALL_PARSER",
                "gpt-oss" if engine == "sglang" else "openai",
            ),
            "agentcap_commit": env("TEAS_AGENTCAP_COMMIT", "unknown"),
            "sweagent_streaming_patch": None,
        } if is_mcp else {
            "inference_engine": engine,
<<<<<<< HEAD
            "inference_engine_version": env("TEAS_ENGINE_VERSION", "unknown"),
            "base_url": env("TEAS_BASE_URL", "http://localhost:8000/v1"),
=======
            f"{engine}_version": engine_version,
            "base_url": "http://localhost:8000/v1",
>>>>>>> e04aeae (modified code to record engine version)
            "is_local": True,
            "backend": env("TEAS_BACKEND", "swebench-k8s"),
            "dataset": dataset,
            "num_examples": n,
            "tensor_parallel_size": int(env("TEAS_TP", env("TEAS_NUM_GPUS", "0"))),
            "max_model_len": int(env("TEAS_MAX_MODEL_LEN", "131072")),
            "streaming": True,
            "timestamp": ts,
            "agentcap_strategy": "sweagent",
            "concurrency": int(env("TEAS_CONCURRENCY", "1")),
            "observed_max_concurrency": int(
                env("TEAS_OBSERVED_MAX_CONCURRENCY", "0")
            ),
            "sweagent_streaming_patch": "AGENTCAP_STREAMING_PATCH_APPLIED",
            "reasoning_parser": "gpt-oss" if engine == "sglang" else "openai_gptoss",
            "tool_call_parser": "gpt-oss" if engine == "sglang" else "openai",
        }),
    }
    observed_llm_concurrency = env("TEAS_OBSERVED_MAX_LLM_CONCURRENCY")
    if observed_llm_concurrency is not None:
        metadata["system_environment"][
            "observed_max_simultaneous_llm_requests"
        ] = int(observed_llm_concurrency)
    if is_mcp:
        evaluator_scheme, eval_judge = resolve_rows_evaluator(
            [r.get("eval_details") for r in rows]
        )
        # None values are omitted rather than written, so a metadata_template
        # can still attest the judge for runs whose rows predate the field.
        if evaluator_scheme is not None:
            metadata["system_environment"]["evaluator"] = evaluator_scheme
        if eval_judge is not None:
            metadata["system_environment"]["eval_judge"] = eval_judge
    if metadata_template:
        generated_metadata = metadata
        metadata = dict(metadata_template)
        for section in ("hardware", "model_config", "system_environment"):
            metadata[section] = {
                **dict(metadata_template.get(section) or {}),
                **generated_metadata[section],
            }

    metrics_path = destination_dir / f"metrics_{suffix}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (destination_dir / f"metadata_{suffix}.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    with (destination_dir / f"detailed-results_{suffix}.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in req_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if include_output_data:
        with (destination_dir / f"output-data_{suffix}.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for row in per_task_rows(rows):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metrics_path


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file() or path.stat().st_size == 0:
        raise TeasOutputError(f"missing or empty JSONL file: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise TeasOutputError(f"blank JSONL line {line_number}: {path}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TeasOutputError(
                    f"invalid JSONL line {line_number}: {path}"
                ) from exc
            if not isinstance(row, dict):
                raise TeasOutputError(f"non-object JSONL line {line_number}: {path}")
            rows.append(row)
    return rows


def validate_teas_leaf(
    leaf_dir: Path,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    wall_time_s: float,
) -> None:
    """Validate a repository-ready four-file agentic TEAS leaf."""

    leaf_dir = Path(leaf_dir)
    files = [path for path in leaf_dir.iterdir() if path.is_file()]
    if len(files) != 4:
        raise TeasOutputError(
            f"TEAS leaf must contain exactly four files, found {len(files)}"
        )
    metadata_files = list(leaf_dir.glob(f"metadata_{dataset}_*.json"))
    metrics_files = list(leaf_dir.glob(f"metrics_{dataset}_*.json"))
    detailed_files = list(leaf_dir.glob(f"detailed-results_{dataset}_*.jsonl"))
    run_script = leaf_dir / "run.sh"
    if not (
        len(metadata_files) == len(metrics_files) == len(detailed_files) == 1
        and run_script.is_file()
        and run_script.stat().st_size > 0
    ):
        raise TeasOutputError(
            "TEAS leaf must contain metadata, metrics, detailed-results, and run.sh"
        )
    if any(_SECRET_PATTERN.search(path.read_bytes()) for path in files):
        raise TeasOutputError("credential-shaped value found in TEAS leaf")

    try:
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        metrics = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeasOutputError("invalid metadata or metrics JSON") from exc
    detailed = _load_jsonl(detailed_files[0])
    if not detailed:
        raise TeasOutputError("detailed-results must contain at least one request")

    quality = metrics.get("quality")
    if not isinstance(quality, dict) or list(quality) != [
        "acc",
        "total_examples",
        "passed",
    ]:
        raise TeasOutputError(
            "metrics.quality must contain exactly acc, total_examples, passed"
        )
    if quality["total_examples"] != len(source_rows):
        raise TeasOutputError("quality denominator does not match result rows")
    if (
        not isinstance(quality["passed"], int)
        or isinstance(quality["passed"], bool)
        or not 0 <= quality["passed"] <= quality["total_examples"]
    ):
        raise TeasOutputError("quality passed count is outside the valid range")
    expected_accuracy = round(quality["passed"] / len(source_rows), 4)
    if quality["acc"] != expected_accuracy:
        raise TeasOutputError("quality accuracy does not match passed/total")

    environment = metadata.get("system_environment") or {}
    hardware = metadata.get("hardware") or {}
    model = metadata.get("model_config") or {}
    required_strings = {
        "inference engine": environment.get("inference_engine"),
        "engine version": environment.get("inference_engine_version"),
        "GPU type": hardware.get("gpu_type"),
        "CPU type": hardware.get("cpu_type"),
        "model": model.get("model_name"),
        "precision": model.get("precision"),
    }
    unknown = [
        name
        for name, value in required_strings.items()
        if not isinstance(value, str) or not value or value == "unknown"
    ]
    if unknown:
        raise TeasOutputError(f"missing required metadata fields: {unknown}")
    for name, value in (
        ("GPU count", hardware.get("num_gpus")),
        ("CPU count", hardware.get("num_cpus")),
        ("TP size", environment.get("tensor_parallel_size")),
        ("max model length", environment.get("max_model_len")),
        ("concurrency", environment.get("concurrency")),
        ("observed concurrency", environment.get("observed_max_concurrency")),
        (
            "observed LLM concurrency",
            environment.get("observed_max_simultaneous_llm_requests"),
        ),
    ):
        if not isinstance(value, int) or value <= 0:
            raise TeasOutputError(f"{name} must be a positive integer")
    if environment.get("dataset") != dataset:
        raise TeasOutputError("metadata dataset does not match export dataset")
    if environment.get("num_examples") != len(source_rows):
        raise TeasOutputError("metadata example count does not match result rows")
    if dataset.startswith("mcp-atlas"):
        for name in ("evaluator", "eval_judge"):
            value = environment.get(name)
            if not isinstance(value, str) or not value or value == "unknown":
                raise TeasOutputError(
                    f"mcp-atlas leaves must attest {name} in system_environment"
                )

    by_example: Dict[int, List[int]] = {}
    detailed_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
    }
    for row in detailed:
        example_index = row.get("example_index")
        request_index = row.get("request_index")
        if not isinstance(example_index, int) or not isinstance(request_index, int):
            raise TeasOutputError("detailed indices must be integers")
        if not 0 <= example_index < len(source_rows) or request_index < 0:
            raise TeasOutputError("detailed indices are outside the valid range")
        by_example.setdefault(example_index, []).append(request_index)
        for key in detailed_totals:
            value = row.get(key)
            if not isinstance(value, int) or value < 0:
                raise TeasOutputError(f"invalid detailed {key}")
            detailed_totals[key] += value
    for example_index, request_indices in by_example.items():
        if sorted(request_indices) != list(range(len(request_indices))):
            raise TeasOutputError(
                f"non-contiguous request indices for example {example_index}"
            )
    usage_totals = {
        key: sum(int((row.get("total_usage") or {}).get(key, 0)) for row in source_rows)
        for key in detailed_totals
    }
    if detailed_totals != usage_totals:
        raise TeasOutputError(
            f"detailed token totals do not match results: "
            f"{detailed_totals} versus {usage_totals}"
        )
    total_requests = sum(
        int((row.get("total_usage") or {}).get("requests", 0))
        for row in source_rows
    )
    if len(detailed) != total_requests:
        raise TeasOutputError(
            f"detailed request count {len(detailed)} != results {total_requests}"
        )

    agentic = metrics.get("agentic") or {}
    expected_agentic = {
        "total_input_tokens": usage_totals["input_tokens"],
        "total_output_tokens": usage_totals["output_tokens"],
        "total_cached_tokens": usage_totals["cached_tokens"],
        "total_requests": total_requests,
        "total_tool_calls": sum(
            int(row.get("tool_calls", 0)) for row in source_rows
        ),
    }
    for key, expected in expected_agentic.items():
        if agentic.get(key) != expected:
            raise TeasOutputError(
                f"metrics.agentic.{key}={agentic.get(key)!r}, expected {expected!r}"
            )

    performance = metrics.get("performance") or {}
    ttfts = [
        float(row["prefill_time_s"])
        for row in detailed
        if float(row["prefill_time_s"]) > 0
    ]
    tpots = [
        float(row["tpot_s"])
        for row in detailed
        if float(row["tpot_s"]) > 0
    ]
    if not ttfts or not tpots:
        raise TeasOutputError("positive TTFT and TPOT records are required")
    expected_performance = {
        "total_wall_time_min": round(wall_time_s / 60.0, 2),
        "ttft": round(sum(ttfts) / len(ttfts), 6),
        "p99_ttft": round(_pct(ttfts, 0.99), 6),
        "tpot": round(sum(tpots) / len(tpots), 6),
        "p99_tpot": round(_pct(tpots, 0.99), 6),
    }
    for key, expected in expected_performance.items():
        if performance.get(key) != expected:
            raise TeasOutputError(
                f"metrics.performance.{key}={performance.get(key)!r}, "
                f"expected {expected!r}"
            )

    script_text = run_script.read_text(encoding="utf-8")
    concurrency = environment["concurrency"]
    identity_fragments = (
        environment["inference_engine"],
        environment["dataset"],
        model["model_name"],
    )
    missing_fragments = [
        fragment for fragment in identity_fragments if fragment not in script_text
    ]
    if missing_fragments:
        raise TeasOutputError(
            f"run.sh is missing identity fragments: {missing_fragments}"
        )
    if not re.search(rf"--concurrency(?:=|\s+){concurrency}\b", script_text):
        raise TeasOutputError(
            f"run.sh does not record --concurrency {concurrency}"
        )


def export_teas_leaf(
    source_dir: Path,
    destination_dir: Path,
    rows: List[Dict[str, Any]],
    dataset: str,
    wall_time_s: float,
    *,
    run_script: Path,
    model_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    swebench_run_ids: Optional[Sequence[str]] = None,
    metadata_template: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically create a strict four-file TEAS repository leaf."""

    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    run_script = Path(run_script)
    if destination_dir.exists():
        raise TeasOutputError(f"destination already exists: {destination_dir}")
    if not run_script.is_file() or run_script.stat().st_size == 0:
        raise TeasOutputError(f"missing or empty run script: {run_script}")
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_dir.name}.tmp-",
            dir=destination_dir.parent,
        )
    )
    try:
        metrics_path = write_teas_outputs(
            source_dir,
            rows,
            dataset,
            wall_time_s,
            model_name=model_name,
            timestamp=timestamp,
            destination_dir=temporary_dir,
            include_output_data=False,
            swebench_run_ids=swebench_run_ids,
            metadata_template=metadata_template,
        )
        if metrics_path is None:
            raise TeasOutputError("TEAS writer produced no metrics")
        shutil.copy2(run_script, temporary_dir / "run.sh")
        validate_teas_leaf(
            temporary_dir,
            rows,
            dataset=dataset,
            wall_time_s=wall_time_s,
        )
        temporary_dir.rename(destination_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return destination_dir
