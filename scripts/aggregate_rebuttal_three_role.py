#!/usr/bin/env python3
"""Compare predeclared paired FinanceBench two-role and PVE rebuttal arms."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.aggregate_rebuttal_prompt_ablation import paired_bootstrap_delta


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exactly_one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def exact_mcnemar_p(fixed: int, broken: int) -> float:
    n = fixed + broken
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(fixed, broken) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def arm_from_baseline(root: Path, key: str) -> dict[str, Any]:
    spec = load_json(root / "ablation_spec.json")
    out_dir = exactly_one(root, "team_plan_execute")
    rows = load_jsonl(exactly_one(out_dir, "output-data_*.jsonl"))
    return {"key": key, "spec": spec, "rows": rows, "output_dir": out_dir}


def arm_from_pve(root: Path, key: str) -> dict[str, Any]:
    spec = load_json(root / "three_role_spec.json")
    summary = load_json(root / "run_summary.json")
    out_dir = Path(summary["output_dir"])
    rows = load_jsonl(exactly_one(out_dir, "output-data_*.jsonl"))
    return {
        "key": key,
        "spec": spec,
        "summary": summary,
        "rows": rows,
        "output_dir": out_dir,
    }


def validate_arm(arm: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    rows = arm["rows"]
    by_id = {str(row["task_id"]): row for row in rows}
    if len(rows) != len(by_id):
        raise SystemExit(f"duplicate task IDs in {arm['key']}")
    if set(by_id) != set(task_ids):
        raise SystemExit(
            f"coverage mismatch in {arm['key']}: {len(by_id)}/{len(task_ids)}"
        )
    ordered = [by_id[task_id] for task_id in task_ids]
    scores = [1 if row.get("eval_passed") is True else 0 for row in ordered]
    unscored = sum(row.get("eval_passed") is None for row in ordered)
    errors = sum(bool(row.get("errors")) for row in ordered)
    if unscored or errors:
        raise SystemExit(
            f"invalid arm {arm['key']}: unscored={unscored}, error_tasks={errors}"
        )
    return {
        "scores": scores,
        "passed": sum(scores),
        "errors": errors,
        "requests": sum(int(row.get("num_requests") or 0) for row in ordered),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in ordered),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in ordered),
        "by_id": by_id,
    }


def verifier_process(arm: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    trajectories = exactly_one(arm["output_dir"], "trajectories_*")
    statuses: Counter[str] = Counter()
    task_revision_rounds: dict[str, int] = {}
    invalid = 0
    for index, task_id in enumerate(task_ids):
        task_dir = trajectories / f"task_{index:03d}"
        verdicts = sorted(task_dir.glob("verifier_verdict_round_*.json"))
        task_revisions = 0
        for verdict_path in verdicts:
            verdict = load_json(verdict_path)
            if not isinstance(verdict, dict):
                invalid += 1
                continue
            status = str(verdict.get("status", "INVALID"))
            statuses[status] += 1
            if status == "NEEDS_REVISION":
                task_revisions += 1
        task_revision_rounds[task_id] = task_revisions
    return {
        "verifier_calls": sum(statuses.values()),
        "verdicts": dict(statuses),
        "tasks_revised": sum(value > 0 for value in task_revision_rounds.values()),
        "total_revision_rounds": sum(task_revision_rounds.values()),
        "max_revision_rounds": max(task_revision_rounds.values(), default=0),
        "invalid_verdict_files": invalid,
        "task_revision_rounds": task_revision_rounds,
    }


def compare(
    baseline: dict[str, Any],
    pve: dict[str, Any],
    task_ids: list[str],
    seed: int,
) -> dict[str, Any]:
    b = validate_arm(baseline, task_ids)
    p = validate_arm(pve, task_ids)
    fixed_ids = [
        task_id
        for task_id, old, new in zip(task_ids, b["scores"], p["scores"])
        if old == 0 and new == 1
    ]
    broken_ids = [
        task_id
        for task_id, old, new in zip(task_ids, b["scores"], p["scores"])
        if old == 1 and new == 0
    ]
    unchanged_pass = sum(old == new == 1 for old, new in zip(b["scores"], p["scores"]))
    unchanged_fail = sum(old == new == 0 for old, new in zip(b["scores"], p["scores"]))
    lo, hi = paired_bootstrap_delta(p["scores"], b["scores"], seed=seed)
    n = len(task_ids)
    return {
        "baseline_arm": baseline["key"],
        "pve_arm": pve["key"],
        "baseline_passed": b["passed"],
        "pve_passed": p["passed"],
        "n": n,
        "absolute_delta_percent": 100.0 * (p["passed"] - b["passed"]) / n,
        "paired_bootstrap_95_percent": [lo, hi],
        "fixed": len(fixed_ids),
        "broken": len(broken_ids),
        "unchanged_pass": unchanged_pass,
        "unchanged_fail": unchanged_fail,
        "exact_mcnemar_p": exact_mcnemar_p(len(fixed_ids), len(broken_ids)),
        "fixed_task_ids": fixed_ids,
        "broken_task_ids": broken_ids,
        "baseline_usage": {
            key: b[key]
            for key in ["requests", "input_tokens", "output_tokens", "errors"]
        },
        "pve_usage": {
            key: p[key]
            for key in ["requests", "input_tokens", "output_tokens", "errors"]
        },
        "pve_process": verifier_process(pve, task_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-mini-gpt54", type=Path, required=True)
    parser.add_argument("--baseline-gpt54-mini", type=Path, required=True)
    parser.add_argument("--pve-mini-gpt54-gpt54", type=Path, required=True)
    parser.add_argument("--pve-gpt54-mini-mini", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    baseline_a = arm_from_baseline(args.baseline_mini_gpt54, "mini planner, GPT-5.4 executor")
    baseline_b = arm_from_baseline(args.baseline_gpt54_mini, "GPT-5.4 planner, mini executor")
    pve_c = arm_from_pve(args.pve_mini_gpt54_gpt54, "mini planner, GPT-5.4 verifier, GPT-5.4 executor")
    pve_d = arm_from_pve(args.pve_gpt54_mini_mini, "GPT-5.4 planner, mini verifier, mini executor")

    specs = [baseline_a["spec"], baseline_b["spec"], pve_c["spec"], pve_d["spec"]]
    hashes = {spec["task_manifest_sha256"] for spec in specs}
    task_id_sets = {tuple(spec["task_ids"]) for spec in specs}
    if len(hashes) != 1 or len(task_id_sets) != 1:
        raise SystemExit("task manifests differ")
    task_ids = list(next(iter(task_id_sets)))
    if len(task_ids) != 30:
        raise SystemExit(f"expected n=30, got {len(task_ids)}")
    plan_hashes = {spec["planner_prompt_sha256"] for spec in specs}
    exec_hashes = {spec["executor_prompt_sha256"] for spec in specs}
    if len(plan_hashes) != 1 or len(exec_hashes) != 1:
        raise SystemExit("planner/executor prompt hashes differ")

    comparisons = [
        compare(baseline_a, pve_c, task_ids, 20260724),
        compare(baseline_b, pve_d, task_ids, 20260725),
    ]
    out = {
        "dataset": "financebench",
        "n": len(task_ids),
        "task_manifest_sha256": next(iter(hashes)),
        "planner_prompt_sha256": next(iter(plan_hashes)),
        "executor_prompt_sha256": next(iter(exec_hashes)),
        "verifier_prompt_sha256": pve_c["spec"]["verifier_prompt_sha256"],
        "failures_scored_as_zero": True,
        "comparisons": comparisons,
    }
    if pve_c["spec"]["verifier_prompt_sha256"] != pve_d["spec"]["verifier_prompt_sha256"]:
        raise SystemExit("verifier prompt hashes differ")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# FinanceBench three-role response-period experiment",
        "",
        f"N={len(task_ids)} matched tasks; manifest `{out['task_manifest_sha256']}`; failures are scored as zero.",
        "",
        "| Two-role baseline | Added verifier | Baseline | Three-role | Delta | Fixed / broken | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        added = "GPT-5.4" if "GPT-5.4 verifier" in row["pve_arm"] else "GPT-5.4-mini"
        lines.append(
            f"| {row['baseline_arm']} | {added} | {row['baseline_passed']},{row['n']} | "
            f"{row['pve_passed']},{row['n']} | {row['absolute_delta_percent']:+.1f}% | "
            f"{row['fixed']} / {row['broken']} | {row['exact_mcnemar_p']:.4f} |"
        )
    lines.extend(["", "Verifier process:", ""])
    for row in comparisons:
        process = row["pve_process"]
        lines.append(
            f"- {row['pve_arm']}: {process['verifier_calls']} verifier calls; "
            f"{process['tasks_revised']} tasks revised; {process['total_revision_rounds']} total revision rounds; "
            f"max {process['max_revision_rounds']} rounds/task."
        )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
