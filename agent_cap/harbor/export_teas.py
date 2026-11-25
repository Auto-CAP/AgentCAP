"""Turn a Harbor job directory into TEAS files and AgentCAP result rows, for any harness.

Each Harbor trial directory holds ``result.json`` (task name, verifier rewards,
agent token totals, timing, exceptions). The agent's usage row comes from the
best source the trial has, recorded in ``agentcard.metering``:

* ``agentcap-host`` / ``dsh-session``: ``agent/capagent_result.json`` written
  by CAPAgent or DSHAgent (per role, per request);
* ``meter-proxy``: ``agent/meter_result.json`` from ``agent_cap.harbor.metered``
  (per request, measured on the host);
* ``harbor-context``: otherwise, Harbor's own totals plus per-step tokens from
  the agent's ATIF ``trajectory.json`` (no TTFT or decode time).

The rows are joined with the verifier rewards and handed to
``agent_cap.agents.teas_output.write_teas_outputs``, so the four TEAS files have
exactly the fields TEASBench's cost scripts read.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from agent_cap.harbor import METER_FILENAME, RESULT_FILENAME


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


def _agent_identity(result: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    info = result.get("agent_info") or {}
    agent_cfg = ((result.get("config") or {}).get("agent")) or {}
    model = agent_cfg.get("model_name") or ((info.get("model_info") or {}).get("name") if isinstance(info.get("model_info"), Mapping) else None)
    return info.get("name"), model


def _atif_steps(trial_dir: Path) -> List[Dict[str, Any]]:
    """Agent steps with token metrics from agent/trajectory.json, subagents included."""
    traj = _load_json(trial_dir / "agent" / "trajectory.json")
    steps: List[Dict[str, Any]] = []

    def walk(t: Mapping[str, Any], role: str) -> None:
        for step in t.get("steps") or []:
            metrics = step.get("metrics") or {}
            if step.get("source") != "agent" or not metrics:
                continue
            steps.append({
                "role": role,
                "model": step.get("model_name"),
                "timestamp": step.get("timestamp"),
                "input_tokens": int(metrics.get("prompt_tokens") or 0),
                "output_tokens": int(metrics.get("completion_tokens") or 0),
                "completion_tokens": int(metrics.get("completion_tokens") or 0),
                "reasoning_tokens": int(((metrics.get("extra") or {}).get("reasoning_tokens")) or 0),
                "cached_tokens": int(metrics.get("cached_tokens") or 0),
                "num_tool_calls": len(step.get("tool_calls") or []),
            })
        for i, sub in enumerate(t.get("subagent_trajectories") or []):
            name = ((sub.get("agent") or {}).get("name")) or f"subagent-{i}"
            walk(sub, f"subagent:{name}")

    if isinstance(traj, Mapping):
        walk(traj, "agent")
    return steps


def harbor_context_row(trial_dir: Path, result: Mapping[str, Any], task_id: str, note: Optional[str]) -> Dict[str, Any]:
    """Row from Harbor's agent context and ATIF trajectory (any Harbor agent)."""
    agent, model = _agent_identity(result)
    ctx = result.get("agent_result") or {}
    steps = _atif_steps(trial_dir)
    row = _empty_row(task_id, note or "")
    row["errors"] = [note] if note else []
    total = {
        "input_tokens": int(ctx.get("n_input_tokens") or 0),
        "output_tokens": int(ctx.get("n_output_tokens") or 0),
        "completion_tokens": int(ctx.get("n_output_tokens") or 0),
        "reasoning_tokens": sum(s["reasoning_tokens"] for s in steps),
        "cached_tokens": int(ctx.get("n_cache_tokens") or 0),
        "requests": len(steps),
    }
    per_role: Dict[str, Dict[str, int]] = {}
    for st in steps:
        u = per_role.setdefault(st["role"], {k: 0 for k in total})
        for k in ("input_tokens", "output_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens"):
            u[k] += st[k]
        u["requests"] += 1
    if not per_role:
        per_role = {"agent": dict(total)}
    start, end = _parse_time((result.get("agent_execution") or {}).get("started_at")), _parse_time(
        (result.get("agent_execution") or {}).get("finished_at"))
    row.update({
        "strategy": agent or "unknown",
        "e2e_latency_s": round((end - start).total_seconds(), 4) if start and end else 0.0,
        "per_role_usage": per_role,
        "total_usage": total,
        "num_turns": len(steps),
        "tool_calls": sum(s["num_tool_calls"] for s in steps),
        "turn_stats": [{**s, "latency_s": 0.0, "ttft_s": 0.0, "decode_s": 0.0} for s in steps],
        "agentcard": {"agent": agent, "strategy": agent, "model": model, "metering": "harbor-context",
                      "roles": {"agent": model}, "cost_usd": ctx.get("cost_usd")},
    })
    return row


def _agent_row(trial_dir: Path, result: Mapping[str, Any], task_id: str) -> Dict[str, Any]:
    own = _load_json(trial_dir / "agent" / RESULT_FILENAME)
    meter = _load_json(trial_dir / "agent" / METER_FILENAME)
    metered = isinstance(meter, dict) and meter.get("metered")
    if metered and not (meter.get("total_usage") or {}).get("requests") and not isinstance(own, dict):
        # Proxy in place but no request it could parse (e.g. another protocol): Harbor's totals.
        meter = {"metered": False, "reason": "the proxy recognised no model requests"}
        metered = False
    if isinstance(own, dict):
        card = own.setdefault("agentcard", {})
        card.setdefault("metering", "dsh-session" if card.get("agent") == "deepseek-harness" else "agentcap-host")
        if metered:  # DSH run behind the proxy too: keep both views
            card["meter"] = {"requests": meter.get("total_usage", {}).get("requests"),
                             "turn_stats": meter.get("turn_stats")}
        return own
    if metered:
        row = {k: v for k, v in meter.items() if k not in ("metered", "requests_raw")}
        agent, model = _agent_identity(result)
        card = row.setdefault("agentcard", {})
        card.setdefault("agent", agent)
        card.setdefault("model", model)
        ctx = result.get("agent_result") or {}
        card["harbor_context"] = {k: ctx.get(k) for k in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd")}
        return row
    if result.get("agent_result") or (trial_dir / "agent" / "trajectory.json").exists():
        note = meter.get("reason") if isinstance(meter, dict) else None
        return harbor_context_row(trial_dir, result, task_id, f"unmetered: {note}" if note else None)
    return _empty_row(task_id, f"missing agent/{RESULT_FILENAME} (agent did not report)")


def trial_row(trial_dir: Path, *, pass_threshold: float = 1.0) -> Dict[str, Any]:
    """One AgentCAP result row for a Harbor trial, with eval fields from the verifier."""
    trial_dir = Path(trial_dir)
    result = _load_json(trial_dir / "result.json") or {}
    harbor_task = _task_name(result, trial_dir)
    # Harbor names tasks "<benchmark>/<id>"; AgentCAP/TEAS rows use the bare id.
    task_id = harbor_task.split("/", 1)[-1]
    row = _agent_row(trial_dir, result, task_id)
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
        "agent_import_path": ((job_config.get("agents") or [{}])[0] or {}).get("import_path"),
        "model": first.get("model") or _first_model(rows),
        "metering": sorted({str((r.get("agentcard") or {}).get("metering")) for r in rows}),
        "strategy": first.get("strategy"),
        "tool_backend": first.get("tool_backend"),
        "roles": first.get("roles"),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


__all__ = ["export_job", "job_rows", "trial_row", "harbor_context_row", "iter_trial_dirs", "job_wall_time_s"]
