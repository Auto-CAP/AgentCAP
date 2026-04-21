import argparse
import ast
import json
import sqlite3

from datasets import load_dataset

from agent_cap.evaluators.gtfa_eval import GTFAEvaluator


def load_claims():
    ds = load_dataset("ScaleAI/mcp-atlas", split="train")
    claims_by_task = {}
    for ex in ds:
        task_id = str(ex.get("TASK", ""))
        raw = ex.get("GTFA_CLAIMS", [])
        if isinstance(raw, str):
            try:
                claims = json.loads(raw)
            except Exception:
                try:
                    claims = ast.literal_eval(raw)
                except Exception:
                    claims = [raw] if raw.strip() else []
        elif isinstance(raw, list):
            claims = raw
        else:
            claims = []
        claims_by_task[task_id] = [str(c).strip() for c in claims if str(c).strip()]
    return claims_by_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    claims_by_task = load_claims()
    print(f"Loaded claims for {len(claims_by_task)} tasks")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    query = "SELECT id, experiment_name, task_id, exec_response FROM hybrid_runs"
    params = ()
    if args.experiment:
        query += " WHERE experiment_name = ?"
        params = (args.experiment,)
    rows = conn.execute(query, params).fetchall()
    print(f"Found {len(rows)} runs")

    evaluator = GTFAEvaluator()

    by_exp = {}
    for row in rows:
        exp = row["experiment_name"]
        task_id = row["task_id"]
        output = row["exec_response"] or ""
        claims = claims_by_task.get(task_id, [])
        if not claims or not output:
            continue
        eval_result = evaluator.evaluate(
            {"gtfa_claims": claims, "response": output}, backend=None
        )
        by_exp.setdefault(exp, []).append((eval_result.passed, eval_result.score))
        conn.execute(
            "UPDATE hybrid_runs SET task_success = ?, quality_score = ? WHERE id = ?",
            (int(bool(eval_result.passed)), float(eval_result.score), row["id"]),
        )
    conn.commit()

    print()
    for exp, res in sorted(by_exp.items()):
        n = len(res)
        passed = sum(1 for p, _ in res if p)
        avg_score = sum(s for _, s in res) / n
        print(f"{exp}: acc={avg_score:.3f} task_coverage={passed/n:.3f} ({passed}/{n})")


if __name__ == "__main__":
    main()
