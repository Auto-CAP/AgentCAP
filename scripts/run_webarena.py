#!/usr/bin/env python3
"""Run WebArena benchmark.

Usage:
    # Start WebArena Docker services first (one-time)
    python -c "from agent_cap.webarena.env_setup import start_services; start_services('YOUR_HOST')"

    # Run 1 task
    python scripts/run_webarena.py --config-dir config_files --limit 1

    # Run specific task
    python scripts/run_webarena.py --config-dir config_files --task-ids 0
"""

import argparse
import json
import logging

from agent_cap.webarena.runner import WebArenaRunner
from agent_cap.webarena.task_loader import load_webarena_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="config_files")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="default")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-ids", type=str, default="")
    parser.add_argument("--output", default="results/webarena/results.jsonl")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    task_ids = None
    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",")]

    tasks = load_webarena_tasks(
        config_dir=args.config_dir,
        limit=args.limit,
        task_ids=task_ids,
    )
    print(f"Loaded {len(tasks)} tasks")

    runner = WebArenaRunner(
        base_url=args.base_url,
        model_id=args.model,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
    )

    results = runner.run_tasks(tasks)

    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"\nResults saved to {args.output}")
    completed = sum(1 for r in results if "error" not in r)
    print(f"Completed: {completed}/{len(results)}")


if __name__ == "__main__":
    main()
