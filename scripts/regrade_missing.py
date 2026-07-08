#!/usr/bin/env python3
"""Grade tasks that a --resume run skipped (their patches never entered the
batch evaluator buffer). Merges verdicts into results.jsonl, eval_k8s_results.json
and metrics.json in place.

Usage: python scripts/regrade_missing.py --run-dir DIR
"""
import argparse
import json
from pathlib import Path

from agent_cap.agents.evaluators import get_evaluator
from agent_cap.agents.metrics import aggregate_agent_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()

    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()
            if l.strip()]
    graded_path = run_dir / "eval_k8s_results.json"
    graded = json.loads(graded_path.read_text()) if graded_path.exists() else {}

    missing = [r for r in rows
               if (r.get("output_text") or "").strip()
               and not r.get("errors")
               and r["task_id"] not in graded]
    print(f"{len(rows)} rows, {len(graded)} already graded, {len(missing)} to grade")
    if missing:
        ev = get_evaluator("swebench-k8s")
        for r in missing:
            ev.evaluate({"eval_config": {"instance_id": r["task_id"]}},
                        r["output_text"])
        sub = run_dir / "eval_k8s_regrade"
        sub.mkdir(exist_ok=True)
        new_results = ev.finalize(sub)
        graded.update(new_results)
        graded_path.write_text(json.dumps(graded, indent=2))

    for r in rows:
        info = graded.get(r["task_id"])
        if info is not None:
            r["eval_passed"] = bool(info.get("resolved"))
            r["eval_score"] = 1.0 if info.get("resolved") else 0.0
            r["eval_details"] = {"evaluator": "swebench-k8s",
                                 "instance_id": r["task_id"],
                                 **(info.get("details") or {})}
    with (run_dir / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    old_metrics = json.loads((run_dir / "metrics.json").read_text())
    wall = old_metrics.get("performance", {}).get("e2e_s", 0.0)
    metrics = aggregate_agent_metrics(rows, wall_time_s=wall,
                                      evaluator_name="swebench-k8s")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=4))
    q = metrics["quality"]
    print(f"final: acc={q['acc']} passed={sum(1 for r in rows if r.get('eval_passed'))}"
          f"/{len(rows)}")


if __name__ == "__main__":
    main()
