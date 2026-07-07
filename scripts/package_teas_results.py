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

    # ---- metrics: reuse CLI aggregate, override hardware, add wall time ----
    metrics_src = run_dir / "metrics.json"
    metrics = json.loads(metrics_src.read_text())
    perf = metrics.get("performance", {})
    perf["total_wall_time_min"] = round(perf.get("e2e_s", 0.0) / 60.0, 2)
    metrics["hardware"] = {
        "gpu_type": args.gpu_type,
        "num_gpus": args.num_gpus,
        f"{args.engine}_version": args.engine_version,
        "concurrency": args.concurrency,
        "streaming": True,
    }
    (out / f"metrics_{suffix}.json").write_text(json.dumps(metrics, indent=2))

    # ---- metadata ----
    n_examples = sum(1 for line in (run_dir / "results.jsonl").read_text().splitlines()
                     if line.strip())
    metadata = {
        "hardware": {"gpu_type": args.gpu_type, "num_gpus": args.num_gpus},
        "model_config": {"model_name": args.model_name, "precision": "mxfp4"},
        "system_environment": {
            "inference_engine": args.engine,
            "engine_version": args.engine_version,
            "engine_image": ("lmsysorg/sglang:v" if args.engine == "sglang"
                             else "vllm/vllm-openai:v") + args.engine_version,
            "base_url": "http://localhost:8000/v1",
            "is_local": True,
            "backend": "swebench-k8s",
            "dataset": args.dataset,
            "num_examples": n_examples,
            "tensor_parallel_size": args.tp,
            "streaming": True,
            "timestamp": ts,
            "agentcap_strategy": "sweagent",
            "reasoning_parser": "gpt-oss" if args.engine == "sglang" else "openai_gptoss",
            "tool_call_parser": "gpt-oss" if args.engine == "sglang" else "openai",
            "notes": ("EIDF k8s; sandboxes + eval as k8s pods from official "
                      "swebench images (no dind); stock official engine image."),
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
    print(f"acc={q.get('acc')} examples={n_examples}")


if __name__ == "__main__":
    main()
