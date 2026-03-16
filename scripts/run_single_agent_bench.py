#!/usr/bin/env python3
"""Run a single-agent benchmark from a YAML config.

Usage::

    # With a pre-running vLLM server:
    python scripts/run_single_agent_bench.py configs/single_agent_gpt_oss_120b.yaml

    # Override output dir:
    python scripts/run_single_agent_bench.py configs/single_agent_gpt_oss_120b.yaml \\
        --output-dir results/my_run

    # Dry-run (print config, don't execute):
    python scripts/run_single_agent_bench.py configs/single_agent_gpt_oss_120b.yaml \\
        --dry-run
"""

import argparse
import json
import logging
import sys

from agent_cap.single_agent.config import SingleAgentBenchConfig
from agent_cap.single_agent.runner import SingleAgentRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-agent performance benchmark runner",
    )
    parser.add_argument(
        "config",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without running",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run first N tasks (0 = all, useful for quick testing)",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = SingleAgentBenchConfig.from_yaml(args.config)

    print("=" * 70)
    print("Single-Agent Benchmark")
    print("=" * 70)
    print(f"  Name:            {config.name}")
    print(f"  Model:           {config.model_id}")
    print(f"  Engine:          {config.serving_engine}")
    print(f"  Server:          {config.base_url}")
    print(f"  Dataset:         {config.dataset} (n={config.dataset_count})")
    print(f"  Batch sizes:     {config.batch_sizes}")
    print(f"  Tool calls:      {'yes' if config.enable_tool_calls else 'no'}")
    print(f"  Max tokens:      {config.max_tokens}")
    print(f"  Repetitions:     {config.repetitions}")
    print(f"  Output dir:      {args.output_dir or config.output_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n--- Dry run: full config ---")
        print(json.dumps(config.to_dict(), indent=2))
        return

    runner = SingleAgentRunner(config)
    results, predictions = runner.run(limit=args.limit)

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(
        f"{'Batch':>5}  {'Mode':<12}  {'E2E_avg':>10}  {'RPS':>7}  "
        f"{'TTFT_avg':>10}  {'TTFT_p99':>10}  "
        f"{'TPOT_avg':>10}  {'TPOT_p99':>10}  "
        f"{'In_tok':>8}  {'Out_tok':>8}  "
        f"{'Tools':>5}  {'GPU%':>5}  {'CPU%':>5}  {'Errs':>4}"
    )
    print("-" * 130)
    for m in results:
        print(
            f"{m.batch_size:>5}  {m.tool_mode:<12}  "
            f"{m.e2e_latency_avg_ms:>9.1f}ms  {m.requests_per_second:>7.2f}  "
            f"{m.ttft_avg_ms:>9.1f}ms  {m.ttft_p99_ms:>9.1f}ms  "
            f"{m.tpot_avg_ms:>9.1f}ms  {m.tpot_p99_ms:>9.1f}ms  "
            f"{m.total_input_tokens:>8}  {m.total_output_tokens:>8}  "
            f"{m.total_tool_calls:>5}  "
            f"{m.avg_gpu_util_pct:>5.1f}  {m.avg_cpu_util_pct:>5.1f}  "
            f"{m.error_count:>4}"
        )

    out_dir = runner.save_results(results, predictions, args.output_dir)
    print(f"\nResults saved to: {out_dir}")
    if predictions:
        print(f"SWE-bench predictions: {out_dir}/predictions.jsonl")
        print(
            "\n  To evaluate with SWE-bench harness:\n"
            "    python -m swebench.harness.run_evaluation \\\n"
            f"      --predictions_path {out_dir}/predictions.jsonl \\\n"
            "      --run_id my_run"
        )


if __name__ == "__main__":
    main()
