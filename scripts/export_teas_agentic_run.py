#!/usr/bin/env python3
"""Convert one completed AgentCAP run into a strict TEAS repository leaf.

The destination contains exactly four files:

* ``metadata_<dataset>_<timestamp>.json``
* ``metrics_<dataset>_<timestamp>.json``
* ``detailed-results_<dataset>_<timestamp>.jsonl``
* ``run.sh``

Existing AgentCAP outputs stay untouched. SWE-bench quality is resolved from
official harness reports, not generation-time zero placeholders.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from agent_cap.agents.teas_output import TeasOutputError, export_teas_leaf


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeasOutputError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise TeasOutputError(f"expected a JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    if not path.is_file() or path.stat().st_size == 0:
        raise TeasOutputError(f"missing or empty results file: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise TeasOutputError(f"blank results line {line_number}: {path}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TeasOutputError(f"invalid results line {line_number}: {path}") from exc
            if not isinstance(row, dict):
                raise TeasOutputError(f"non-object results line {line_number}: {path}")
            rows.append(row)
    return rows


def _canonical_rows(
    run_dir: Path,
    rows: list[Dict[str, Any]],
    dataset: str,
    timestamp: Optional[str],
) -> list[Dict[str, Any]]:
    """Restore dataset order when concurrent completion reordered results.jsonl."""

    expected = run_dir / f"output-data_{dataset}_{timestamp}.jsonl" if timestamp else None
    if expected is not None and expected.is_file():
        order_file = expected
    else:
        candidates = sorted(run_dir.glob(f"output-data_{dataset}_*.jsonl"))
        if not candidates:
            return rows
        if len(candidates) > 1:
            raise TeasOutputError(
                "multiple canonical output-data files found; select matching "
                "metadata/timestamp explicitly: "
                f"{[path.name for path in candidates]}"
            )
        order_file = candidates[0]

    ordered_task_ids = [row.get("task_id") for row in _load_jsonl(order_file)]
    if any(not isinstance(task_id, str) for task_id in ordered_task_ids) or len(
        ordered_task_ids
    ) != len(set(ordered_task_ids)):
        raise TeasOutputError(f"invalid task order file: {order_file}")
    by_task = {row.get("task_id"): row for row in rows}
    if len(by_task) != len(rows) or set(by_task) != set(ordered_task_ids):
        raise TeasOutputError(f"task IDs in {order_file.name} do not match results.jsonl")
    return [by_task[task_id] for task_id in ordered_task_ids]


def _existing_metadata(
    run_dir: Path,
    explicit_path: Optional[Path],
) -> Dict[str, Any]:
    if explicit_path is not None:
        return _load_json(explicit_path)
    candidates = sorted(run_dir.glob("metadata_*.json"))
    if not candidates:
        return {}
    if len(candidates) > 1:
        raise TeasOutputError(
            "multiple metadata files found; select one with --metadata: "
            f"{[path.name for path in candidates]}"
        )
    return _load_json(candidates[0])


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _required(value: Any, name: str) -> Any:
    if value is None or value == "" or value == "unknown":
        raise TeasOutputError(
            f"{name} is required; pass the corresponding CLI option or provide "
            "a complete --metadata file"
        )
    return value


def _set_env(name: str, value: Any) -> None:
    if value is not None:
        os.environ[name] = str(value)


def _comma_join(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return ",".join(str(item) for item in value)
    raise TeasOutputError(f"cannot serialize metadata list: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--run-script", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--timestamp", help="YYYYMMDD_HHMMSS")
    parser.add_argument("--engine", choices=("vllm", "sglang"))
    parser.add_argument("--engine-version")
    parser.add_argument("--gpu-type")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--cpu-type")
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--model-name")
    parser.add_argument("--precision")
    parser.add_argument("--base-url")
    parser.add_argument("--backend")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--observed-concurrency", type=int)
    parser.add_argument("--observed-llm-concurrency", type=int)
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--tool-call-parser")
    parser.add_argument(
        "--swebench-run-id",
        action="append",
        default=None,
        help="official evaluation run_id to include; repeat for retries",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise TeasOutputError(f"run directory does not exist: {run_dir}")

    existing = _existing_metadata(
        run_dir,
        args.metadata.expanduser().resolve() if args.metadata else None,
    )
    hardware = existing.get("hardware") or {}
    model = existing.get("model_config") or {}
    environment = existing.get("system_environment") or {}

    dataset = _required(_first(args.dataset, environment.get("dataset")), "dataset")
    engine = _required(
        _first(args.engine, environment.get("inference_engine")),
        "engine",
    )
    engine_version = _required(
        _first(args.engine_version, environment.get("inference_engine_version")),
        "engine version",
    )
    gpu_type = _required(_first(args.gpu_type, hardware.get("gpu_type")), "GPU type")
    num_gpus = _required(
        _first(args.num_gpus, hardware.get("num_gpus")),
        "GPU count",
    )
    cpu_type = _required(_first(args.cpu_type, hardware.get("cpu_type")), "CPU type")
    num_cpus = _required(
        _first(args.num_cpus, hardware.get("num_cpus")),
        "CPU count",
    )
    tp_size = _required(
        _first(args.tp, environment.get("tensor_parallel_size")),
        "TP size",
    )
    max_model_len = _required(
        _first(args.max_model_len, environment.get("max_model_len")),
        "max model length",
    )
    model_name = _required(
        _first(args.model_name, model.get("model_name")),
        "model name",
    )
    precision = _required(
        _first(args.precision, model.get("precision")),
        "precision",
    )
    base_url = _required(
        _first(args.base_url, environment.get("base_url")),
        "base URL",
    )
    backend = _required(
        _first(args.backend, environment.get("backend")),
        "backend",
    )
    concurrency = _required(
        _first(args.concurrency, environment.get("concurrency")),
        "concurrency",
    )
    observed_concurrency = _required(
        _first(
            args.observed_concurrency,
            environment.get("observed_max_concurrency"),
        ),
        "observed concurrency",
    )
    observed_llm_concurrency = _required(
        _first(
            args.observed_llm_concurrency,
            environment.get("observed_max_simultaneous_llm_requests"),
        ),
        "observed LLM concurrency",
    )
    reasoning_parser = _first(
        args.reasoning_parser,
        environment.get("reasoning_parser"),
        "gpt-oss" if engine == "sglang" else "openai_gptoss",
    )
    tool_call_parser = _first(
        args.tool_call_parser,
        environment.get("tool_call_parser"),
        "gpt-oss" if engine == "sglang" else "openai",
    )
    timestamp = _first(args.timestamp, environment.get("timestamp"))
    run_script = args.run_script.expanduser().resolve() if args.run_script else run_dir / "run.sh"

    rows = _canonical_rows(
        run_dir,
        _load_jsonl(run_dir / "results.jsonl"),
        dataset,
        timestamp,
    )
    raw_metrics = _load_json(run_dir / "metrics.json")
    wall_time_s = (raw_metrics.get("performance") or {}).get("e2e_s")
    if not isinstance(wall_time_s, (int, float)) or wall_time_s <= 0:
        raise TeasOutputError("metrics.json performance.e2e_s must be positive")

    values = {
        "TEAS_ENGINE": engine,
        "TEAS_ENGINE_VERSION": engine_version,
        "TEAS_GPU_TYPE": gpu_type,
        "TEAS_NUM_GPUS": num_gpus,
        "TEAS_CPU_TYPE": cpu_type,
        "TEAS_NUM_CPUS": num_cpus,
        "TEAS_TP": tp_size,
        "TEAS_MAX_MODEL_LEN": max_model_len,
        "TEAS_MODEL_NAME": model_name,
        "TEAS_PRECISION": precision,
        "TEAS_BASE_URL": base_url,
        "TEAS_BACKEND": backend,
        "TEAS_CONCURRENCY": concurrency,
        "TEAS_OBSERVED_MAX_CONCURRENCY": observed_concurrency,
        "TEAS_OBSERVED_MAX_LLM_CONCURRENCY": observed_llm_concurrency,
        "TEAS_REASONING_PARSER": reasoning_parser,
        "TEAS_TOOL_CALL_PARSER": tool_call_parser,
        "TEAS_MCP_SERVER_URL": environment.get("mcp_server_url"),
        "TEAS_MCP_TOOL_COUNT": environment.get("mcp_tool_count"),
        "TEAS_MCP_ENABLED_SERVERS": _comma_join(environment.get("mcp_enabled_servers")),
        "TEAS_MCP_ATLAS_COMMIT": environment.get("mcp_atlas_commit"),
        "TEAS_MCP_DATA_DIR": environment.get("mcp_data_dir"),
        "TEAS_TASK_INDICES_SHA256": environment.get("task_indices_sha256"),
        "TEAS_AGENTCAP_COMMIT": environment.get("agentcap_commit"),
    }
    for name, value in values.items():
        _set_env(name, value)

    leaf = export_teas_leaf(
        run_dir,
        output_dir,
        rows,
        dataset,
        float(wall_time_s),
        run_script=run_script,
        model_name=model_name,
        timestamp=timestamp,
        swebench_run_ids=args.swebench_run_id,
        metadata_template=existing,
    )
    metrics_path = next(leaf.glob(f"metrics_{dataset}_*.json"))
    quality = _load_json(metrics_path)["quality"]
    print(f"wrote strict TEAS leaf: {leaf}")
    print(f"quality: {quality['passed']}/{quality['total_examples']} " f"(acc={quality['acc']})")
    print("files:")
    for path in sorted(leaf.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TeasOutputError as exc:
        raise SystemExit(f"error: {exc}") from exc
