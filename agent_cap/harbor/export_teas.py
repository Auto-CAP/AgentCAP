"""Turn a Harbor job directory into TEAS files and AgentCAP result rows.

Each Harbor trial directory holds ``result.json`` (task name, verifier rewards,
timing, exceptions) and, for CAPAgent trials, ``agent/capagent_result.json``
(the AgentCAP per-task row with per-request ``turn_stats``). This module joins
the two and hands the rows to ``agent_cap.agents.teas_output.write_teas_outputs``,
so the four TEAS files have exactly the fields TEASBench's cost scripts read.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from agent_cap.harbor import RESULT_FILENAME


def iter_trial_dirs(job_dir: Path) -> List[Path]:
    """Trial directories of a Harbor job (those with a result.json), sorted by name."""
    job_dir = Path(job_dir)
    return sorted(p.parent for p in job_dir.glob("*/result.json") if p.parent != job_dir)


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _task_name(result: Mapping[str, Any], trial_dir: Path) -> str:
    for key in ("task_name", "task_id"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    name = trial_dir.name
    return name.rsplit("__", 1)[0] if "__" in name else name


def _rewards(result: Mapping[str, Any]) -> Dict[str, float]:
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    return dict(rewards) if isinstance(rewards, Mapping) else {}


def _empty_row(task_id: str, error: str) -> Dict[str, Any]:
    zero = {
        "input_tokens": 0,
        "output_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "requests": 0,
    }
    return {
        "task_id": task_id,
        "strategy": "unknown",
        "output_text": "",
        "e2e_latency_s": 0.0,
        "per_role_usage": {},
        "total_usage": dict(zero),
        "errors": [error],
        "num_turns": 0,
        "tool_calls": 0,
        "turn_stats": [],
    }


def trial_row(trial_dir: Path, *, pass_threshold: float = 1.0) -> Dict[str, Any]:
    """One AgentCAP result row for a Harbor trial, with eval fields from the verifier."""
    trial_dir = Path(trial_dir)
    result = _load_json(trial_dir / "result.json") or {}
    harbor_task = _task_name(result, trial_dir)
    # Harbor names tasks "<benchmark>/<id>"; AgentCAP/TEAS rows use the bare id.
    task_id = harbor_task.split("/", 1)[-1]
    row = _load_json(trial_dir / "agent" / RESULT_FILENAME)
    if not isinstance(row, dict):
        row = _empty_row(task_id, f"missing agent/{RESULT_FILENAME} (agent did not report)")
    row["task_id"] = task_id

    rewards = _rewards(result)
    grade = _load_json(trial_dir / "verifier" / "grade.json")
    exception = result.get("exception_info")
    details: Dict[str, Any] = {"evaluator": "harbor-verifier", "rewards": rewards}
    if isinstance(grade, Mapping):
        details["grade"] = dict(grade)
    if exception:
        details["exception"] = exception

    if "reward" in rewards:
        score = float(rewards["reward"])
        if "pass_all" in rewards:
            passed = bool(rewards["pass_all"])
        else:
            passed = score >= pass_threshold
        row["eval_passed"] = passed
        row["eval_score"] = score
    else:
        row["eval_passed"] = False
        row["eval_score"] = 0.0
        details["reason"] = "no verifier reward"
    row["eval_details"] = details
    row["harbor_task"] = harbor_task
    row["harbor_trial"] = trial_dir.name
    return row


def job_rows(job_dir: Path, *, pass_threshold: float = 1.0) -> List[Dict[str, Any]]:
    return [trial_row(t, pass_threshold=pass_threshold) for t in iter_trial_dirs(job_dir)]


def job_wall_time_s(job_dir: Path) -> float:
    """Wall-clock span of the job: job result.json if present, else trial min/max."""
    job_dir = Path(job_dir)
    job_result = _load_json(job_dir / "result.json") or {}
    start = _parse_time(job_result.get("started_at"))
    end = _parse_time(job_result.get("finished_at"))
    if start and end:
        return max(0.0, (end - start).total_seconds())
    starts, ends = [], []
    for trial in iter_trial_dirs(job_dir):
        res = _load_json(trial / "result.json") or {}
        s, e = _parse_time(res.get("started_at")), _parse_time(res.get("finished_at"))
        if s:
            starts.append(s)
        if e:
            ends.append(e)
    if starts and ends:
        return max(0.0, (max(ends) - min(starts)).total_seconds())
    return 0.0


def export_job(
    job_dir: Path,
    out_dir: Path,
    *,
    dataset: str,
    model_name: Optional[str] = None,
    pass_threshold: float = 1.0,
    timestamp: Optional[str] = None,
) -> Tuple[Optional[Path], Path]:
    """Write TEAS files plus ``results.jsonl`` for a Harbor job.

    Returns ``(metrics_path, results_path)``; ``metrics_path`` is None when the
    job has no trials.
    """
    from agent_cap.agents.teas_output import write_teas_outputs

    job_dir, out_dir = Path(job_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = job_rows(job_dir, pass_threshold=pass_threshold)
    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not rows:
        return None, results_path
    model = model_name or _first_model(rows)
    metrics_path = write_teas_outputs(
        out_dir,
        rows,
        dataset=dataset,
        wall_time_s=job_wall_time_s(job_dir),
        model_name=model,
        timestamp=timestamp,
    )
    if metrics_path is not None:
        _stamp_harbor_provenance(metrics_path, job_dir, rows)
    return metrics_path, results_path


def _first_model(rows: List[Mapping[str, Any]]) -> Optional[str]:
    for row in rows:
        roles = (row.get("agentcard") or {}).get("roles") or {}
        for model in roles.values():
            if model:
                return str(model)
    return None


def _stamp_harbor_provenance(metrics_path: Path, job_dir: Path, rows: List[Mapping[str, Any]]) -> None:
    """Add a ``harbor`` section to metadata_*.json; existing TEAS fields are untouched."""
    meta_path = metrics_path.with_name(metrics_path.name.replace("metrics_", "metadata_", 1))
    metadata = _load_json(meta_path)
    if not isinstance(metadata, dict):
        return
    try:
        from importlib.metadata import version

        harbor_version = version("harbor")
    except Exception:
        harbor_version = "unknown"
    job_config = _load_json(Path(job_dir) / "config.json") or {}
    first = (rows[0].get("agentcard") if rows else None) or {}
    metadata["harbor"] = {
        "harbor_version": harbor_version,
        "job_dir": str(Path(job_dir).resolve()),
        "job_name": job_config.get("job_name") or Path(job_dir).name,
        "environment": (job_config.get("environment") or {}).get("type"),
        "n_trials": len(rows),
        "agent": first.get("agent"),
        "strategy": first.get("strategy"),
        "tool_backend": first.get("tool_backend"),
        "roles": first.get("roles"),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


__all__ = ["export_job", "job_rows", "trial_row", "iter_trial_dirs", "job_wall_time_s"]
