import json, ast, sys
from pathlib import Path
from agent_cap.evaluators.gtfa_eval import GTFAEvaluator
from datasets import load_dataset

ds = load_dataset("ScaleAI/mcp-atlas", split="train")
task_claims = {}
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
    claims = [str(c).strip() for c in claims if str(c).strip()]
    task_claims[task_id] = claims

print(f"Loaded claims for {len(task_claims)} tasks")

evaluator = GTFAEvaluator()
base_dir = Path("/data/sicheng/agent-team-data")
runs = sorted(
    list(base_dir.glob("*_mcpatlas/team_plan_execute/output-data_*.jsonl"))
    + list(base_dir.glob("*_mcpatlas/team_plan_execute/output_data_*.jsonl"))
)

print(f"Found {len(runs)} runs to re-evaluate\n")

for run_path in runs:
    run_name = run_path.parts[-3]
    print(f"=== {run_name} ===")

    results = []
    with open(run_path) as f:
        for line in f:
            results.append(json.loads(line))

    updated = []
    passed_count = 0
    scores = []

    for i, r in enumerate(results):
        task_id = r["task_id"]
        output = r.get("output_text", "")
        claims = task_claims.get(task_id, [])

        if not claims or not output:
            r["eval_passed"] = None
            r["eval_score"] = None
            r["eval_details"] = None
            updated.append(r)
            continue

        eval_result = evaluator.evaluate(
            {"gtfa_claims": claims, "response": output},
            backend=None,
        )

        r["eval_passed"] = eval_result.passed
        r["eval_score"] = eval_result.score
        r["eval_details"] = eval_result.details
        scores.append(eval_result.score)
        if eval_result.passed:
            passed_count += 1
        updated.append(r)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(results)} done")

    with open(run_path, "w") as f:
        for r in updated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if scores:
        acc = sum(scores) / len(scores)
        task_coverage = passed_count / len(scores)
        print(
            f"  acc={acc:.3f} task_coverage={task_coverage:.3f} ({passed_count}/{len(scores)})"
        )

        metrics_files = list(run_path.parent.glob("metrics_*.json"))
        for mf in metrics_files:
            with open(mf) as f:
                metrics = json.load(f)
            metrics["eval"] = {
                "acc": round(acc, 3),
                "task_coverage": round(task_coverage, 3),
                "total_evaluated": len(scores),
                "evaluator": "google/gemini-3.1-flash-lite-preview",
            }
            with open(mf, "w") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
    else:
        print("  no evaluable tasks")
    print()

print("DONE")
