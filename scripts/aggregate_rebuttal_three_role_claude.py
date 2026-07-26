#!/usr/bin/env python3
"""Aggregate preregistered Claude/GPT FinanceBench PE-vs-PVE pairs.

Reads only completed raw TeamRunner artifacts. It preserves the primary evaluator,
requires exact matched coverage, and computes public API-equivalent cost from
provider-reported usage rather than proxy billing.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scripts.aggregate_rebuttal_three_role import (
    arm_from_pve,
    compare,
    exactly_one,
    load_json,
)

RATE_CARD_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-5.4": {"uncached_input": 2.50, "cached_input": 0.25, "output": 15.00},
    "claude-opus-4-6": {"uncached_input": 5.00, "cached_input": 0.50, "output": 25.00},
}
RATE_CARD_SOURCES = {
    "gpt-5.4": "https://developers.openai.com/api/docs/models/gpt-5.4",
    "claude-opus-4-6": "https://www.anthropic.com/news/claude-opus-4-7",
    "claude_cache_read": "https://docs.anthropic.com/en/docs/about-claude/pricing",
}


def resource_summary(arm: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    trajectories = exactly_one(arm["output_dir"], "trajectories_*")
    roles: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    for index, task_id in enumerate(task_ids):
        summary = load_json(trajectories / f"task_{index:03d}" / "summary.json")
        if str(summary.get("task_id")) != task_id:
            raise SystemExit(
                f"trajectory ID mismatch in {arm['key']} at {index}: "
                f"{summary.get('task_id')} != {task_id}"
            )
        latencies.append(float(summary.get("e2e_latency_s") or 0.0))
        for role, values in (summary.get("roles") or {}).items():
            model = str(values.get("model") or "")
            if model not in RATE_CARD_USD_PER_MTOK:
                raise SystemExit(f"no frozen public rate card for {model}")
            bucket = roles.setdefault(
                role,
                {
                    "model": model,
                    "requests": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                },
            )
            if bucket["model"] != model:
                raise SystemExit(f"role {role} changes model in {arm['key']}")
            bucket["requests"] += int(values.get("num_requests") or 0)
            bucket["input_tokens"] += int(values.get("input_tokens") or 0)
            bucket["cached_input_tokens"] += int(values.get("cached_tokens") or 0)
            bucket["output_tokens"] += int(values.get("output_tokens") or 0)

    total_cost = 0.0
    for values in roles.values():
        cached = values["cached_input_tokens"]
        uncached = values["input_tokens"] - cached
        if uncached < 0:
            raise SystemExit(f"cached input exceeds input for {arm['key']}")
        values["uncached_input_tokens"] = uncached
        values["cache_write_tokens"] = None
        rates = RATE_CARD_USD_PER_MTOK[values["model"]]
        cost = (
            uncached * rates["uncached_input"]
            + cached * rates["cached_input"]
            + values["output_tokens"] * rates["output"]
        ) / 1_000_000
        values["public_api_equivalent_cost_usd"] = cost
        total_cost += cost

    metrics = load_json(exactly_one(arm["output_dir"], "metrics_*.json"))
    agentic = metrics.get("agentic") or {}
    role_requests = sum(v["requests"] for v in roles.values())
    role_input = sum(v["input_tokens"] for v in roles.values())
    role_output = sum(v["output_tokens"] for v in roles.values())
    role_cached = sum(v["cached_input_tokens"] for v in roles.values())
    expected = {
        "requests": int(agentic.get("total_requests") or 0),
        "input_tokens": int(agentic.get("total_input_tokens") or 0),
        "output_tokens": int(agentic.get("total_output_tokens") or 0),
        "cached_input_tokens": int(agentic.get("total_cached_tokens") or 0),
    }
    observed = {
        "requests": role_requests,
        "input_tokens": role_input,
        "output_tokens": role_output,
        "cached_input_tokens": role_cached,
    }
    if observed != expected:
        raise SystemExit(f"resource cross-check failed in {arm['key']}: {observed} != {expected}")

    return {
        "roles": roles,
        "totals": {
            **observed,
            "uncached_input_tokens": role_input - role_cached,
            "cache_write_tokens": None,
            "public_api_equivalent_cost_usd": total_cost,
        },
        "latency": {
            "sum_task_latency_s": sum(latencies),
            "mean_task_latency_s": statistics.mean(latencies),
            "median_task_latency_s": statistics.median(latencies),
            "run_wall_s": float((metrics.get("performance") or {}).get("e2e_s") or 0.0),
        },
    }


def assert_pair_invariant(baseline: dict[str, Any], pve: dict[str, Any]) -> None:
    bspec, pspec = baseline["spec"], pve["spec"]
    for key in [
        "task_manifest_sha256",
        "task_ids",
        "planner_prompt_sha256",
        "executor_prompt_sha256",
        "base_url",
        "temperature",
        "max_turns",
    ]:
        if bspec.get(key) != pspec.get(key):
            raise SystemExit(f"matched-pair invariant differs for {key}")
    for role in ["planner", "executor"]:
        if bspec["roles"].get(role) != pspec["roles"].get(role):
            raise SystemExit(f"matched-pair {role} model differs")
        if bspec["role_max_tokens"].get(role) != pspec["role_max_tokens"].get(role):
            raise SystemExit(f"matched-pair {role} max_tokens differs")
    if set(bspec["roles"]) != {"planner", "executor"}:
        raise SystemExit("baseline is not exactly two-role")
    if set(pspec["roles"]) != {"planner", "verifier", "executor"}:
        raise SystemExit("treatment is not exactly three-role")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-gpt54-claude46", type=Path, required=True)
    parser.add_argument("--pve-gpt54-claude46-claude46", type=Path, required=True)
    parser.add_argument("--baseline-claude46-gpt54", type=Path, required=True)
    parser.add_argument("--pve-claude46-claude46-gpt54", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    arms = {
        "a0": arm_from_pve(args.baseline_gpt54_claude46, "GPT-5.4 planner, Claude Opus 4.6 executor"),
        "a1": arm_from_pve(args.pve_gpt54_claude46_claude46, "GPT-5.4 planner, Claude Opus 4.6 verifier, Claude Opus 4.6 executor"),
        "b0": arm_from_pve(args.baseline_claude46_gpt54, "Claude Opus 4.6 planner, GPT-5.4 executor"),
        "b1": arm_from_pve(args.pve_claude46_claude46_gpt54, "Claude Opus 4.6 planner, Claude Opus 4.6 verifier, GPT-5.4 executor"),
    }
    assert_pair_invariant(arms["a0"], arms["a1"])
    assert_pair_invariant(arms["b0"], arms["b1"])

    specs = [arm["spec"] for arm in arms.values()]
    manifests = {spec["task_manifest_sha256"] for spec in specs}
    task_sets = {tuple(spec["task_ids"]) for spec in specs}
    verifier_hashes = {
        spec["verifier_prompt_sha256"] for spec in specs if spec.get("verifier_prompt_sha256")
    }
    if len(manifests) != 1 or len(task_sets) != 1:
        raise SystemExit("task manifests differ across four arms")
    if len(verifier_hashes) != 1:
        raise SystemExit("verifier prompts differ across treatment arms")
    task_ids = list(next(iter(task_sets)))
    if len(task_ids) != 30:
        raise SystemExit(f"expected 30 tasks, found {len(task_ids)}")

    comparisons = [
        compare(arms["a0"], arms["a1"], task_ids, 20260726),
        compare(arms["b0"], arms["b1"], task_ids, 20260727),
    ]
    for row in comparisons:
        revisions = row["pve_process"]["task_revision_rounds"]
        revised_ids = {task_id for task_id, count in revisions.items() if count > 0}
        fixed_ids = set(row["fixed_task_ids"])
        broken_ids = set(row["broken_task_ids"])
        row["revision_attribution_diagnostic"] = {
            "classification": "posthoc deterministic trajectory audit",
            "revised_task_count": len(revised_ids),
            "fixed_with_revision": len(fixed_ids & revised_ids),
            "broken_with_revision": len(broken_ids & revised_ids),
            "interpretation": (
                "Accuracy changes on tasks without a revision cannot be attributed to "
                "verifier-requested plan correction; temperature-zero API runs may still "
                "differ across independently executed arms."
            ),
        }
    resources = {key: resource_summary(arm, task_ids) for key, arm in arms.items()}
    out = {
        "dataset": "financebench",
        "n": len(task_ids),
        "task_manifest_sha256": next(iter(manifests)),
        "verifier_prompt_sha256": next(iter(verifier_hashes)),
        "failures_scored_as_zero": True,
        "primary_evaluator": "AgentCARD FinanceBench deterministic evaluator",
        "rate_card_usd_per_mtok": RATE_CARD_USD_PER_MTOK,
        "rate_card_sources": RATE_CARD_SOURCES,
        "cost_semantics": "public API-equivalent; not CLIPROXY subscription billing",
        "cache_write_tokens": None,
        "comparisons": comparisons,
        "resources": resources,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# FinanceBench Claude three-role matched experiment",
        "",
        f"N={len(task_ids)}; manifest `{out['task_manifest_sha256']}`; execution failures score zero.",
        "",
        "| Baseline | Added role | Baseline | Three-role | Delta | Fixed / broken | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['baseline_arm']} | Claude Opus 4.6 verifier | "
            f"{row['baseline_passed']},{row['n']} | {row['pve_passed']},{row['n']} | "
            f"{row['absolute_delta_percent']:+.1f}% | {row['fixed']} / {row['broken']} | "
            f"{row['exact_mcnemar_p']:.4f} |"
        )
    lines.extend(["", "## Revision-attribution diagnostic", ""])
    for row in comparisons:
        diagnostic = row["revision_attribution_diagnostic"]
        lines.append(
            f"- {row['pve_arm']}: {diagnostic['fixed_with_revision']}/{row['fixed']} fixed "
            f"tasks and {diagnostic['broken_with_revision']}/{row['broken']} broken tasks "
            "received a verifier-requested revision. This posthoc deterministic audit does "
            "not change the primary score."
        )
    lines.extend(["", "## Resources", "", "| Arm | Requests | Input | Cached input | Output | Public API-equivalent cost | Mean latency/task |", "|---|---:|---:|---:|---:|---:|---:|"])
    for key in ["a0", "a1", "b0", "b1"]:
        total = resources[key]["totals"]
        latency = resources[key]["latency"]
        lines.append(
            f"| {arms[key]['key']} | {total['requests']} | {total['input_tokens']} | "
            f"{total['cached_input_tokens']} | {total['output_tokens']} | "
            f"${total['public_api_equivalent_cost_usd']:.3f} | "
            f"{latency['mean_task_latency_s']:.1f}s |"
        )
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
