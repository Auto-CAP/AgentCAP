import argparse
import json
import logging
import sys
from pathlib import Path


def cmd_run(args):
    from agent_cap.config import ExperimentConfig
    from agent_cap.runner import ExperimentExecutor, TaskDef
    from agent_cap.db import ResultStore

    config = ExperimentConfig.from_yaml(args.config)
    print(f"Experiment: {config.name}")
    print(f"Models: {len(config.models)}")
    print(f"Total configurations: {config.total_configs}")
    print(f"Repetitions: {config.repetitions}")

    if args.dry_run:
        print("\n--- Dry run: listing all configurations ---")
        for i, cfg in enumerate(config.iter_configs()):
            print(
                f"  [{i+1}] {cfg['model'].id} q={cfg['quantization']} "
                f"skills={cfg['skill_subset']} retries={cfg['num_retries']}"
            )
        return

    if not args.tasks:
        print("Error: --tasks file required (JSON list of tasks)")
        sys.exit(1)

    tasks_data = json.loads(Path(args.tasks).read_text())
    tasks = [
        TaskDef(
            id=t["id"],
            name=t["name"],
            messages=t["messages"],
            category=t.get("category", ""),
        )
        for t in tasks_data
    ]

    store = ResultStore(args.db)
    try:
        executor = ExperimentExecutor(config=config, tasks=tasks, store=store)
        executor.run()
    finally:
        store.close()


def cmd_metrics(args):
    from agent_cap.db import ResultStore

    store = ResultStore(args.db)
    runs = store.get_runs(experiment_name=args.experiment)
    store.close()

    print(f"Loaded {len(runs)} runs for experiment '{args.experiment}'")

    models = set(r.model_id for r in runs)
    quants = set(r.quantization for r in runs)
    print(f"Models: {sorted(models)}")
    print(f"Quantizations: {sorted(quants)}")

    if runs and runs[0].quality_score is not None:
        avg_q = sum(r.quality_score for r in runs if r.quality_score) / len(runs)
        avg_cost = sum(r.gpu_seconds for r in runs) / len(runs)
        print(f"Avg quality: {avg_q:.2f}, Avg GPU-seconds: {avg_cost:.1f}")


def cmd_pareto(args):
    from agent_cap.db import ResultStore
    from agent_cap.analysis import compute_pareto_frontier, ParetoPoint

    store = ResultStore(args.db)
    runs = store.get_runs(experiment_name=args.experiment)
    store.close()

    config_agg = {}
    for r in runs:
        key = f"{r.model_id}|{r.quantization}|{r.skill_subset}"
        if key not in config_agg:
            config_agg[key] = {"qualities": [], "gpu_seconds": [], "latencies": []}
        if r.quality_score is not None:
            config_agg[key]["qualities"].append(r.quality_score)
        config_agg[key]["gpu_seconds"].append(r.gpu_seconds)
        config_agg[key]["latencies"].append(r.latency_e2e_ms)

    points = []
    for key, data in config_agg.items():
        if not data["qualities"]:
            continue
        points.append(
            ParetoPoint(
                config_id=key,
                quality=sum(data["qualities"]) / len(data["qualities"]),
                gpu_seconds=sum(data["gpu_seconds"]) / len(data["gpu_seconds"]),
                latency_ms=sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0,
            )
        )

    frontier = compute_pareto_frontier(points)

    print(f"\nPareto Frontier ({len(frontier)}/{len(points)} configs):")
    print(f"{'Config':<50} {'Quality':>8} {'GPU-sec':>10}")
    print("-" * 70)
    for p in frontier:
        print(f"{p.config_id:<50} {p.quality:>8.2f} {p.gpu_seconds:>10.1f}")


def main():
    parser = argparse.ArgumentParser(
        prog="agent-cap",
        description="AgentCAP — Experiment sweep engine for multi-skill agent evaluation",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run an experiment from YAML config")
    p_run.add_argument("config", help="Path to experiment YAML config")
    p_run.add_argument("--tasks", help="Path to tasks JSON file")
    p_run.add_argument("--db", default="results/experiments.db", help="Results database path")
    p_run.add_argument("--dry-run", action="store_true", help="List configs without running")
    p_run.add_argument("-v", "--verbose", action="store_true")

    p_metrics = sub.add_parser("metrics", help="Compute metrics from results")
    p_metrics.add_argument("experiment", help="Experiment name")
    p_metrics.add_argument("--db", default="results/experiments.db")

    p_pareto = sub.add_parser("pareto", help="Compute Pareto frontier")
    p_pareto.add_argument("experiment", help="Experiment name")
    p_pareto.add_argument("--db", default="results/experiments.db")

    args = parser.parse_args()

    if args.command == "run":
        if hasattr(args, "verbose") and args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        cmd_run(args)
    elif args.command == "metrics":
        cmd_metrics(args)
    elif args.command == "pareto":
        cmd_pareto(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
