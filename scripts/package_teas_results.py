#!/usr/bin/env python3
"""Re-generate TEAS-format outputs for an EXISTING run dir.

New runs write these automatically at the end (agent_cap.agents.teas_output,
wired into the sweagent CLI path); this wrapper exists only to backfill runs
that finished before that, or to re-emit with corrected hardware facts.

Usage:
    python scripts/package_teas_results.py --run-dir DIR \
        --engine sglang --engine-version 0.5.9 \
        --gpu-type "NVIDIA H200" --num-gpus 1 --tp 1 \
        [--cpu-type STR --num-cpus N] [--model-name M] [--timestamp TS]
"""
import argparse
import json
import os
from pathlib import Path

from agent_cap.agents.teas_output import write_teas_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine", required=True, choices=["sglang", "vllm"])
    ap.add_argument("--engine-version", required=True)
    ap.add_argument("--gpu-type", required=True)
    ap.add_argument("--num-gpus", type=int, required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--model-name", default="unsloth/gpt-oss-120b")
    ap.add_argument("--precision", default="mxfp4")
    ap.add_argument("--dataset", default="swe-bench-lite")
    ap.add_argument("--cpu-type", default=None)
    ap.add_argument("--num-cpus", type=int, default=None)
    ap.add_argument("--timestamp", default=None, help="YYYYMMDD_HHMMSS; default now")
    args = ap.parse_args()

    os.environ.update({
        "TEAS_ENGINE": args.engine,
        "TEAS_ENGINE_VERSION": args.engine_version,
        "TEAS_GPU_TYPE": args.gpu_type,
        "TEAS_NUM_GPUS": str(args.num_gpus),
        "TEAS_TP": str(args.tp),
        "TEAS_MODEL_NAME": args.model_name,
        "TEAS_PRECISION": args.precision,
    })
    if args.cpu_type:
        os.environ["TEAS_CPU_TYPE"] = args.cpu_type
    if args.num_cpus:
        os.environ["TEAS_NUM_CPUS"] = str(args.num_cpus)

    run_dir = Path(args.run_dir).expanduser().resolve()
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()
            if l.strip()]
    wall_s = json.loads((run_dir / "metrics.json").read_text()) \
        .get("performance", {}).get("e2e_s", 0.0)
    p = write_teas_outputs(run_dir, rows, args.dataset, wall_s,
                           model_name=args.model_name, timestamp=args.timestamp)
    metrics = json.loads(p.read_text())
    print(f"wrote {p.parent}/{{metadata,metrics,detailed-results,output-data}}_*")
    print(f"acc={metrics['quality']['acc']} examples={metrics['quality']['total_examples']}")


if __name__ == "__main__":
    main()
