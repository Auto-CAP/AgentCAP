#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from agent_cap import Tracer, StepType  
from agent_cap.core.types import Trace  



def percentile(xs: List[float], p: float) -> float:

    if not xs:
        return 0.0

    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)

    if f == c:
        return xs_sorted[f]

    return xs_sorted[f] * (c - k) + xs_sorted[c] * (k - f)


def summarize_numbers(xs: List[float]) -> Dict[str, float]:

    if not xs:
        return {
            "n": 0.0,
            "mean": 0.0,
            "stdev": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "n": float(len(xs)),
        "mean": float(statistics.mean(xs)),
        "stdev": float(statistics.pstdev(xs) if len(xs) > 1 else 0.0),
        "p50": float(percentile(xs, 50)),
        "p95": float(percentile(xs, 95)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }



# Synthetic Workload 

@dataclass(frozen=True)
class SyntheticWorkloadConfig:
    """
    定义一个 synthetic agent run 的“执行模板参数”。

    每个阶段给一个 base_ms（基础耗时，毫秒），再用 jitter 生成微小抖动：
        duration = base_ms * Uniform(1-jitter, 1+jitter)

    可构造不同的baseline：
    - decode-heavy：把 decode_base_ms 调大
    - tool-heavy：把 tool_base_ms 调大 + 增加 tool_calls
    - retrieval-heavy：把 retrieval_base_ms 调大
    """

    # unit = ms
    planning_base_ms: float = 80.0
    retrieval_base_ms: float = 200.0
    embedding_base_ms: float = 70.0
    prefill_base_ms: float = 150.0
    decode_base_ms: float = 350.0
    tool_base_ms: float = 120.0
    code_base_ms: float = 160.0
    reasoning_base_ms: float = 180.0

    # tool call no.
    tool_calls: int = 1
    jitter: float = 0.10


def _sleep_ms(base_ms: float, jitter: float) -> float:

    j = max(0.0, jitter)
    factor = random.uniform(1.0 - j, 1.0 + j)
    duration_ms = max(0.0, base_ms * factor)
    time.sleep(duration_ms / 1000.0)
    return duration_ms


def run_one(config: SyntheticWorkloadConfig, seed: int) -> Trace:


    random.seed(seed)
    tracer = Tracer(name="synthetic-e2e-wallclock")
    tracer.start()

    with tracer.step("agent_run", StepType.OTHER) as s:

        s.metadata["profile"] = "synthetic_step_mix"
        s.metadata["jitter"] = config.jitter
        s.metadata["tool_calls"] = config.tool_calls

# 1) planning
        with tracer.step("planning", StepType.PLANNING) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.planning_base_ms, config.jitter)

# 2) retrieval
        with tracer.step("retrieval", StepType.RETRIEVAL) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.retrieval_base_ms, config.jitter)

# 3) embedding
        with tracer.step("embedding", StepType.EMBEDDING) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.embedding_base_ms, config.jitter)
            step.metadata["embedding_dim"] = 768

# 4) prefill
        with tracer.step("llm_prefill", StepType.PREFILL) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.prefill_base_ms, config.jitter)
            step.metadata["input_tokens"] = 1024 

# 5) decode
        with tracer.step("llm_decode", StepType.DECODE) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.decode_base_ms, config.jitter)
            step.metadata["output_tokens"] = 256  

# 6) tool calls
        for i in range(config.tool_calls):
            with tracer.step(f"tool_call_{i+1}", StepType.TOOL_CALLING) as step:
                step.metadata["slept_ms"] = _sleep_ms(config.tool_base_ms, config.jitter)
                step.metadata["tool_name"] = "mock_tool"

# 7) code simulation
        with tracer.step("code_execution", StepType.CODE_EXECUTION) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.code_base_ms, config.jitter)
            step.metadata["language"] = "python"

# 8) reasoning
        with tracer.step("reasoning", StepType.REASONING) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.reasoning_base_ms, config.jitter)

    trace = tracer.stop()
    return trace


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run synthetic E2E wall-clock baseline for Agent-CAP.")
    ap.add_argument("--runs", type=int, default=20, help="Number of runs used for formal statistics (recommended >= 20)")
    ap.add_argument("--warmup", type=int, default=2, help="Number of warm-up runs (not included in statistics)")
    ap.add_argument("--seed", type=int, default=12345, help="Base random seed (for reproducibility)")
    ap.add_argument("--out_dir", type=str, default="baselines/synthetic_e2e_runs", help="Directory for trace outputs")
    ap.add_argument(
        "--summary_path",
        type=str,
        default="baselines/synthetic_e2e_wallclock_summary.json",
        help="Output path for baseline summary JSON",
    )

    ap.add_argument("--jitter", type=float, default=0.10, help="Jitter ratio, e.g. 0.10 means ±10%")
    ap.add_argument("--tool_calls", type=int, default=1, help="Number of tool calls")

    ap.add_argument("--decode_ms", type=float, default=350.0, help="Base latency of the decode stage (ms)")
    ap.add_argument("--retrieval_ms", type=float, default=200.0, help="Base latency of the retrieval stage (ms)")
    ap.add_argument("--prefill_ms", type=float, default=150.0, help="Base latency of the prefill stage (ms)")


    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SyntheticWorkloadConfig(
        retrieval_base_ms=args.retrieval_ms,
        prefill_base_ms=args.prefill_ms,
        decode_base_ms=args.decode_ms,
        tool_calls=args.tool_calls,
        jitter=args.jitter,
    )


# 1) warmup
    for i in range(max(0, args.warmup)):
        _ = run_one(config, seed=args.seed + i)


# 2) formal runs
    totals_ms: List[float] = []
    trace_paths: List[str] = []

    for i in range(max(1, args.runs)):
        seed_i = args.seed + 1000 + i
        trace = run_one(config, seed=seed_i)

        totals_ms.append(trace.total_duration_ms)

        path = out_dir / f"trace_{i+1:03d}.json"
        path.write_text(trace.to_json(indent=2), encoding="utf-8")
        trace_paths.append(str(path))

 # 3) summary
    summary = {
        "definition": "E2E wall-clock latency per agent run = Trace.total_duration_ms",
        "workload": "synthetic agent execution template (step-mix)",
        "runs": args.runs,
        "warmup": args.warmup,
        "config": {
            "retrieval_base_ms": config.retrieval_base_ms,
            "prefill_base_ms": config.prefill_base_ms,
            "decode_base_ms": config.decode_base_ms,
            "tool_calls": config.tool_calls,
            "jitter": config.jitter,
        },
        "env": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "trace_dir": str(out_dir),
        "trace_files": trace_paths[:5] + (["..."] if len(trace_paths) > 5 else []),
        "total_duration_ms": summarize_numbers(totals_ms),
    }

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


    print(f"[OK] saved {len(trace_paths)} traces -> {out_dir}")
    print(f"[OK] saved summary -> {summary_path}")
    print(
        "[E2E] total_duration_ms  "
        f"p50={summary['total_duration_ms']['p50']:.2f}  "
        f"p95={summary['total_duration_ms']['p95']:.2f}  "
        f"mean={summary['total_duration_ms']['mean']:.2f}"
    )


if __name__ == "__main__":
    main()
