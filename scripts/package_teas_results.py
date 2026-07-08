#!/usr/bin/env python3
"""Package an `agent_cap.agents` sweagent run dir into the TEAS results layout.

Produces (suffix = swe-bench-lite_<YYYYMMDD_HHMMSS>):
    metadata_<suffix>.json          hardware / model_config / system_environment
    metrics_<suffix>.json           performance / agentic / quality / hardware
    detailed-results_<suffix>.jsonl one row per LLM request (from stream_stats)
    output-data_<suffix>.jsonl      one row per task (from results.jsonl)
    run.sh                          how the run was launched (copied verbatim)

Usage:
    python scripts/package_teas_results.py \
        --run-dir ~/agentcap-results/sglang_h200x1_... \
        --engine sglang --engine-version 0.5.9 \
        --gpu-type "NVIDIA H200" --num-gpus 1 --tp 1 \
        [--dest ~/TEAS_Development_Results_Private/agentic_results/eidf]
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def per_request_rows(run_dir: Path):
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


def per_task_rows(run_dir: Path):
    rows = []
    results = run_dir / "results.jsonl"
    for i, line in enumerate(results.read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        u = r.get("total_usage") or {}
        rows.append({
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
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine", required=True, choices=["sglang", "vllm"])
    ap.add_argument("--engine-version", required=True)
    ap.add_argument("--gpu-type", required=True, help='e.g. "NVIDIA H200"')
    ap.add_argument("--num-gpus", type=int, required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--model-name", default="unsloth/gpt-oss-120b")
    ap.add_argument("--dataset", default="swe-bench-lite")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--cpu-type", default=None)
    ap.add_argument("--num-cpus", type=int, default=None)
    ap.add_argument("--dest", default=None,
                    help="agentic_results/eidf root; default: <run-dir>/teas")
    ap.add_argument("--timestamp", default=None, help="YYYYMMDD_HHMMSS; default now")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.dataset}_{ts}"
    gpu_short = args.gpu_type.split()[-1].lower()          # h200 / h100 / a100
    combo = f"{gpu_short}x{args.num_gpus}"
    ts_dir = f"{ts[2:8]}-{ts[9:13]}"                        # 260707-1234

    if args.dest:
        out = (Path(args.dest).expanduser() / args.engine / "gpt-oss-120b"
               / args.dataset / combo / "batch-size-default" / ts_dir)
    else:
        out = run_dir / "teas" / combo / ts_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- metrics: exact TEAS reference schema (see agentic/vastai/... b300x4) ----
    def pct(vals, p):
        vals = sorted(vals)
        if not vals:
            return 0.0
        k = (len(vals) - 1) * p
        lo = int(k)
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] * (1 - (k - lo)) + vals[hi] * (k - lo)

    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()
            if l.strip()]
    req_rows = per_request_rows(run_dir)
    cli_metrics = json.loads((run_dir / "metrics.json").read_text())
    wall_s = cli_metrics.get("performance", {}).get("e2e_s", 0.0)

    e2e = [r.get("e2e_latency_s", 0.0) for r in rows]
    ttfts = [r["prefill_time_s"] for r in req_rows if r["prefill_time_s"] > 0]
    tpots = [r["tpot_s"] for r in req_rows if r["tpot_s"] > 0]
    n = len(rows)
    passed = sum(1 for r in rows if r.get("eval_passed"))
    tot_in = sum(r["total_usage"]["input_tokens"] for r in rows)
    tot_out = sum(r["total_usage"]["output_tokens"] for r in rows)
    tot_cached = sum(r["total_usage"]["cached_tokens"] for r in rows)
    tot_reqs = sum(r["total_usage"]["requests"] for r in rows)
    tot_tools = sum(r.get("tool_calls", 0) for r in rows)

    metrics = {
        "performance": {
            "total_wall_time_min": round(wall_s / 60.0, 2),
            "avg_e2e_latency_s": round(sum(e2e) / max(n, 1), 2),
            "p50_e2e_latency_s": round(pct(e2e, 0.5), 2),
            "p99_e2e_latency_s": round(pct(e2e, 0.99), 2),
            "ttft": round(sum(ttfts) / max(len(ttfts), 1), 6),
            "p99_ttft": round(pct(ttfts, 0.99), 6),
            "tpot": round(sum(tpots) / max(len(tpots), 1), 6),
            "p99_tpot": round(pct(tpots, 0.99), 6),
        },
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
        "quality": {
            "acc": round(passed / max(n, 1), 4),
            "total_examples": n,
            "passed": passed,
        },
        "hardware": {
            "gpu_type": args.gpu_type,
            "num_gpus": args.num_gpus,
            f"{args.engine}_version": args.engine_version,
            "streaming": True,
        },
    }
    (out / f"metrics_{suffix}.json").write_text(json.dumps(metrics, indent=2))

    # ---- metadata: exact TEAS reference schema ----
    cpu_type, num_cpus = args.cpu_type, args.num_cpus
    if not cpu_type or not num_cpus:
        # best effort: read from the serving pod's node if it is still up
        import subprocess
        gpu_key = combo.split("x")[0]
        pod = subprocess.run(
            ["kubectl", "get", "pods", "-l", f"app={args.engine}-gptoss-{gpu_key}",
             "-o", "jsonpath={.items[0].metadata.name}"],
            capture_output=True, text=True).stdout.strip()
        if pod:
            r = subprocess.run(
                ["kubectl", "exec", pod, "--", "sh", "-c",
                 "nproc; lscpu | grep 'Model name' | head -1 | cut -d: -f2"],
                capture_output=True, text=True)
            lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
            if len(lines) >= 1 and not num_cpus:
                try:
                    num_cpus = int(lines[0])
                except ValueError:
                    pass
            if len(lines) >= 2 and not cpu_type:
                cpu_type = lines[1]
    metadata = {
        "hardware": {
            "gpu_type": args.gpu_type,
            "num_gpus": args.num_gpus,
            "cpu_type": cpu_type or "unknown",
            "num_cpus": num_cpus or 0,
        },
        "model_config": {"model_name": args.model_name, "precision": "mxfp4"},
        "system_environment": {
            "inference_engine": args.engine,
            "base_url": "http://localhost:8000/v1",
            "is_local": True,
            "backend": "swebench-k8s",
            "dataset": args.dataset,
            "num_examples": n,
            "tensor_parallel_size": args.tp,
            "streaming": True,
            "timestamp": ts,
            "agentcap_strategy": "sweagent",
            "sweagent_streaming_patch": "AGENTCAP_STREAMING_PATCH_APPLIED",
            "reasoning_parser": "gpt-oss" if args.engine == "sglang" else "openai_gptoss",
            "tool_call_parser": "gpt-oss" if args.engine == "sglang" else "openai",
        },
    }
    (out / f"metadata_{suffix}.json").write_text(json.dumps(metadata, indent=2))

    # ---- per-request / per-task jsonl ----
    with (out / f"detailed-results_{suffix}.jsonl").open("w") as f:
        for row in per_request_rows(run_dir):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / f"output-data_{suffix}.jsonl").open("w") as f:
        for row in per_task_rows(run_dir):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- run.sh provenance ----
    run_sh = run_dir / "run.sh"
    if run_sh.exists():
        shutil.copy(run_sh, out / "run.sh")

    print(f"packaged -> {out}")
    q = metrics.get("quality", {})
    print(f"acc={q.get('acc')} examples={n}")


if __name__ == "__main__":
    main()
