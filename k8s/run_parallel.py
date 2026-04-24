#!/usr/bin/env python3
"""Parallel SWE-bench runner - spawns N concurrent unified_runner processes."""
import argparse, subprocess, sys, os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path

lock = Lock()
def log(msg):
    with lock:
        print(msg, flush=True)

def run_one(task_idx, total, base_url, output_dir, model_name="openai/gpt-oss-120b",
            serving_engine="vllm", api_key="", dataset="swe-bench-pro", no_streaming=False):
    cmd = [
        sys.executable, "-m", "agent_cap.runner.unified_runner",
        "--model-name", model_name,
        "--dataset", dataset,
        "--backend", "swebench-k8s",
        "--serving-engine", serving_engine,
        "--base-url", base_url,
        "--max-turns", "50",
        "--num-tasks", "1",
        "--task-offset", str(task_idx),
        "--output-dir", str(Path(output_dir) / f"task_{task_idx:03d}"),
    ]
    if no_streaming:
        cmd.append("--no-streaming")
    env = os.environ.copy()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                       cwd=str(Path(__file__).resolve().parent.parent), env=env)
    # Extract result from stdout
    has_patch = "patch saved" in r.stdout
    test_pass = "TEST: PASS" in r.stdout
    log(f"[{task_idx}] done patch={'Y' if has_patch else 'N'} test={'PASS' if test_pass else 'FAIL'}")
    return {"idx": task_idx, "has_patch": has_patch, "test_pass": test_pass, "rc": r.returncode}

def get_task_indices(language=None, num_tasks=100, task_offset=0, dataset="swe-bench-pro"):
    """Get task indices, optionally filtered by language."""
    if language:
        from datasets import load_dataset
        ds_name = "princeton-nlp/SWE-bench_Lite" if "lite" in dataset else "ScaleAI/SWE-bench_Pro"
        ds = load_dataset(ds_name, split="test")
        indices = [i for i, ex in enumerate(ds) if ex.get("repo_language", "").lower() == language.lower()]
        log(f"Found {len(indices)} {language} tasks out of {len(ds)} total")
        return indices[task_offset:task_offset + num_tasks]
    else:
        return list(range(task_offset, task_offset + num_tasks))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-tasks", type=int, default=100)
    p.add_argument("--task-offset", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--base-url", default="http://localhost:30002/v1")
    p.add_argument("--output-dir", default="results/swebench_h200_final")
    p.add_argument("--language", type=str, default=None,
                   help="Filter tasks by language (e.g. python, go, js, ts)")
    p.add_argument("--model-name", default="openai/gpt-oss-120b")
    p.add_argument("--serving-engine", default="vllm")
    p.add_argument("--api-key", default="")
    p.add_argument("--dataset", default="swe-bench-pro",
                   help="Dataset: swe-bench-pro or swe-bench-lite")
    p.add_argument("--task-indices", type=str, default=None,
                   help="JSON file with 'new_indices' list, or comma-separated indices")
    p.add_argument("--no-streaming", action="store_true",
                   help="Disable streaming (needed for SGLang tool calls)")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.task_indices:
        if args.task_indices.endswith(".json"):
            with open(args.task_indices) as f:
                data = json.load(f)
            tasks = data.get("new_indices", data.get("indices", []))
        else:
            tasks = [int(x) for x in args.task_indices.split(",")]
        log(f"Using custom task indices: {len(tasks)} tasks")
    else:
        tasks = get_task_indices(args.language, args.num_tasks, args.task_offset, args.dataset)
    log(f"Running {len(tasks)} tasks with concurrency {args.concurrency}")
    results = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_one, i, len(tasks), args.base_url, args.output_dir,
                              args.model_name, args.serving_engine, args.api_key, args.dataset,
                              args.no_streaming): i for i in tasks}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            patches = sum(1 for x in results if x["has_patch"])
            passes = sum(1 for x in results if x["test_pass"])
            log(f"[progress] {len(results)}/{len(tasks)} done, {patches} patches, {passes} passed")

    patches = sum(1 for x in results if x["has_patch"])
    passes = sum(1 for x in results if x["test_pass"])
    log(f"\nDone! {passes}/{len(tasks)} passed = {passes/max(len(tasks),1)*100:.1f}% acc, {patches} patches")

    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump({"passed": passes, "total": len(tasks), "patches": patches,
                   "acc": passes / max(len(tasks), 1),
                   "language": args.language or "all",
                   "results": sorted(results, key=lambda x: x["idx"])}, f, indent=2)

    # --- Merge per-task results into 4 consolidated files ---
    merge_results(args.output_dir, len(tasks), passes, patches, args)

def merge_results(output_dir, total, passes, patches, args):
    """Merge per-task result files into 4 consolidated files."""
    import statistics
    output_dir = Path(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = getattr(args, 'dataset', 'swe-bench-pro')
    sfx = f"{dataset_name}_{timestamp}"

    all_detailed = []
    all_output = []
    all_metrics = []

    task_dirs = sorted(output_dir.glob("task_*/openai/gpt-oss-120b/"))
    if not task_dirs:
        # try other model paths
        task_dirs = sorted(output_dir.glob("task_*/*/*/"))
    if not task_dirs:
        log("WARNING: no per-task result dirs found, skipping merge")
        return

    for i, td in enumerate(task_dirs):
        dr_files = list(td.glob("detailed-results_*.jsonl"))
        od_files = list(td.glob("output-data_*.jsonl"))
        m_files = list(td.glob("metrics_*.json"))
        if not (dr_files and od_files and m_files):
            continue
        with open(dr_files[0]) as f:
            for line in f:
                row = json.loads(line)
                row["example_index"] = i
                all_detailed.append(row)
        with open(od_files[0]) as f:
            for line in f:
                row = json.loads(line)
                row["index"] = i
                all_output.append(row)
        all_metrics.append(json.load(open(m_files[0])))

    if not all_metrics:
        log("WARNING: no metrics found, skipping merge")
        return

    n = len(all_metrics)
    e2e_lats = [m["performance"]["avg_e2e_latency_s"] for m in all_metrics if m["performance"]["avg_e2e_latency_s"] > 0]
    ttfts = [m["performance"]["ttft"] for m in all_metrics if m["performance"]["ttft"] > 0]
    tpots = [m["performance"]["tpot"] for m in all_metrics if m["performance"]["tpot"] > 0]
    decode_ts = [m["performance"]["decode_time_s"] for m in all_metrics if m["performance"]["decode_time_s"] > 0]
    throughputs = [m["performance"]["output_throughput_tok_s"] for m in all_metrics if m["performance"]["output_throughput_tok_s"] > 0]

    total_input = sum(m["agentic"]["total_input_tokens"] for m in all_metrics)
    total_output_tok = sum(m["agentic"]["total_output_tokens"] for m in all_metrics)
    total_requests = sum(m["agentic"]["total_requests"] for m in all_metrics)
    total_tool_calls = sum(m["agentic"]["total_tool_calls"] for m in all_metrics)
    total_cached = sum(m["agentic"]["total_cached_tokens"] for m in all_metrics)

    def pct(vals, p):
        s = sorted(vals)
        k = int(len(s) * p / 100)
        return s[min(k, len(s)-1)]

    conc = max(args.concurrency, 1)
    merged_metrics = {
        "performance": {
            "total_wall_time_min": round(sum(e2e_lats) / conc / 60) if e2e_lats else 0,
            "avg_e2e_latency_s": round(statistics.mean(e2e_lats), 2) if e2e_lats else 0,
            "p50_e2e_latency_s": round(pct(e2e_lats, 50), 2) if e2e_lats else 0,
            "p99_e2e_latency_s": round(pct(e2e_lats, 99), 2) if e2e_lats else 0,
            "ttft": round(statistics.mean(ttfts), 6) if ttfts else 0,
            "p99_ttft": round(pct(ttfts, 99), 6) if ttfts else 0,
            "tpot": round(statistics.mean(tpots), 6) if tpots else 0,
            "p99_tpot": round(pct(tpots, 99), 6) if tpots else 0,
            "decode_time_s": round(statistics.mean(decode_ts), 2) if decode_ts else 0,
            "output_throughput_tok_s": round(statistics.mean(throughputs), 2) if throughputs else 0,
        },
        "agentic": {
            "avg_total_input_tokens": round(total_input / n, 2),
            "avg_total_output_tokens": round(total_output_tok / n, 2),
            "avg_tool_call_count": round(total_tool_calls / n, 2),
            "avg_num_requests": round(total_requests / n, 2),
            "avg_input_tokens_per_request": round(total_input / total_requests, 2) if total_requests else 0,
            "avg_output_tokens_per_request": round(total_output_tok / total_requests, 2) if total_requests else 0,
            "avg_max_input_tokens_per_request": round(statistics.mean(
                [m["agentic"]["avg_max_input_tokens_per_request"] for m in all_metrics]), 2),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output_tok,
            "total_cached_tokens": total_cached,
            "total_requests": total_requests,
            "total_tool_calls": total_tool_calls,
        },
        "quality": {
            "acc": passes / max(total, 1),
            "total_examples": total,
        },
        "hardware": {
            "gpu_type": all_metrics[0].get("hardware", {}).get("gpu_type", "unknown"),
            "num_gpus": all_metrics[0].get("hardware", {}).get("num_gpus", 0),
            "vllm_version": "0.19.0",
            "concurrency": args.concurrency,
        },
    }

    metadata = {
        "model_name": args.model_name,
        "dataset": dataset_name,
        "backend": "swebench-k8s",
        "serving_engine": args.serving_engine,
        "max_turns": 50,
        "max_tokens": 8192,
        "temperature": 0.0,
        "num_tasks": total,
        "timestamp": timestamp,
    }

    with open(output_dir / f"metadata_{sfx}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / f"metrics_{sfx}.json", "w") as f:
        json.dump(merged_metrics, f, indent=2)
    with open(output_dir / f"detailed-results_{sfx}.jsonl", "w") as f:
        for row in all_detailed:
            f.write(json.dumps(row) + "\n")
    with open(output_dir / f"output-data_{sfx}.jsonl", "w") as f:
        for row in all_output:
            f.write(json.dumps(row) + "\n")

    log(f"\nMerged results: {len(all_detailed)} detailed rows, {len(all_output)} output rows")
    log(f"Files written to {output_dir}/")
    for name in ["metadata", "metrics", "detailed-results", "output-data"]:
        log(f"  {name}_{sfx}.*")

if __name__ == "__main__":
    main()
