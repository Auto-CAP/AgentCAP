import argparse
import json
import logging
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


logger = logging.getLogger("agent_cap.cli")


@dataclass
class ComboModelConfig:
    id: str
    arch: str = "dense"
    params_b: float = 0
    active_b: float = 0
    tp: int = 1
    port: int = 30000
    cuda_visible_devices: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComboModelConfig":
        return cls(
            id=str(data["id"]),
            arch=str(data.get("arch", "dense")),
            params_b=float(data.get("params_b", 0)),
            active_b=float(data.get("active_b", 0)),
            tp=int(data.get("tp", 1)),
            port=int(data.get("port", 30000)),
            cuda_visible_devices=str(data.get("cuda_visible_devices", "")),
        )


@dataclass
class ComboConfig:
    name: str
    description: str = ""
    small_model: ComboModelConfig | None = None
    large_model: ComboModelConfig | None = None
    strategies: List[str] = field(default_factory=list)
    serving_engine: str = "sglang"
    max_tokens: int = 8192
    gpu_type: str = ""
    python_path: str = "python"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComboConfig":
        if "small_model" not in data or "large_model" not in data:
            raise ValueError(
                "Combo config requires both 'small_model' and 'large_model'"
            )
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            small_model=ComboModelConfig.from_dict(data["small_model"]),
            large_model=ComboModelConfig.from_dict(data["large_model"]),
            strategies=[str(x) for x in (data.get("strategies", []) or [])],
            serving_engine=str(data.get("serving_engine", "sglang")),
            max_tokens=int(data.get("max_tokens", 8192)),
            gpu_type=str(data.get("gpu_type", "")),
            python_path=str(data.get("python_path", "python")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ComboConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("Combo config YAML must contain a mapping at top level")
        return cls.from_dict(data)


def _parse_benchmark_spec(spec: str) -> Tuple[str, int]:
    parts = spec.split(":", 1)
    bench_name = parts[0]
    bench_count = int(parts[1]) if len(parts) > 1 else 50
    return bench_name, bench_count


def _short_model(model_id: str) -> str:
    return model_id.split("/")[-1] if model_id else "unknown"


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
                f"  [{i + 1}] {cfg['model'].id} q={cfg['quantization']} "
                f"skills={cfg['skill_subset']} retries={cfg['num_retries']}"
            )
        return

    if args.benchmark:
        from agent_cap.benchmarks import load_benchmark

        bench_name, bench_count = _parse_benchmark_spec(args.benchmark)
        tasks = load_benchmark(bench_name, bench_count)
        print(f"Loaded {len(tasks)} tasks from benchmark '{bench_name}'")
    elif args.tasks:
        tasks_data = json.loads(Path(args.tasks).read_text())
        tasks = [
            TaskDef(
                id=t["id"],
                name=t["name"],
                messages=t["messages"],
                category=t.get("category", ""),
                eval_config=t.get("eval"),
            )
            for t in tasks_data
        ]
    else:
        print("Error: --tasks or --benchmark required")
        sys.exit(1)

    store = ResultStore(args.db)
    try:
        executor = ExperimentExecutor(config=config, tasks=tasks, store=store)
        executor.run()
    finally:
        store.close()


def cmd_combo(args):
    from agent_cap.benchmarks import load_benchmark
    from agent_cap.combinations import (
        run_adaptive_cascade,
        run_best_of_n,
        run_cascade,
        run_generate_verify,
        run_multi_model_vote,
        run_self_critique,
        run_single_pass,
    )
    from agent_cap.db import ResultStore, RunResult
    from agent_cap.server import (
        ChatClient,
        GPUMonitor,
        ModelServerManager,
        ServerConfig,
    )

    config = ComboConfig.from_yaml(args.config)
    benchmark_name, benchmark_count = _parse_benchmark_spec(args.benchmark)
    tasks = load_benchmark(benchmark_name, benchmark_count)

    supported_strategies = {
        "single-pass-small",
        "single-pass-large",
        "best-of-n-small",
        "best-of-n-large",
        "cascade",
        "adaptive-cascade",
        "self-critique-small",
        "self-critique-large",
        "vote",
        "generate-verify",
    }
    invalid = [s for s in config.strategies if s not in supported_strategies]
    if invalid:
        raise ValueError(
            f"Unknown combo strategies: {invalid}. Supported: {sorted(supported_strategies)}"
        )
    if not config.strategies:
        raise ValueError("Combo config must include at least one strategy")

    print(f"Combo Experiment: {config.name}")
    print(f"Description: {config.description}")
    print(
        f"Models: small={config.small_model.id} (port={config.small_model.port}), "
        f"large={config.large_model.id} (port={config.large_model.port})"
    )
    print(f"Benchmark: {benchmark_name} ({len(tasks)} tasks)")
    print(f"Strategies ({len(config.strategies)}): {', '.join(config.strategies)}")

    if args.dry_run:
        print("\n--- Dry run: strategy/model execution plan ---")
        for strategy in config.strategies:
            if strategy == "single-pass-small":
                plan = f"{strategy}: {_short_model(config.small_model.id)}"
            elif strategy == "single-pass-large":
                plan = f"{strategy}: {_short_model(config.large_model.id)}"
            elif strategy == "best-of-n-small":
                plan = (
                    f"{strategy}: {_short_model(config.small_model.id)} (n=3, temp=0.7)"
                )
            elif strategy == "best-of-n-large":
                plan = (
                    f"{strategy}: {_short_model(config.large_model.id)} (n=3, temp=0.7)"
                )
            elif strategy == "self-critique-small":
                plan = f"{strategy}: {_short_model(config.small_model.id)} (generate/critique/revise)"
            elif strategy == "self-critique-large":
                plan = f"{strategy}: {_short_model(config.large_model.id)} (generate/critique/revise)"
            elif strategy == "vote":
                plan = (
                    f"{strategy}: vote({_short_model(config.small_model.id)}, "
                    f"{_short_model(config.large_model.id)})"
                )
            elif strategy == "cascade":
                plan = f"{strategy}: {_short_model(config.small_model.id)} -> {_short_model(config.large_model.id)}"
            elif strategy == "adaptive-cascade":
                plan = (
                    f"{strategy}: {_short_model(config.small_model.id)} self-assess -> "
                    f"{_short_model(config.large_model.id)} on low confidence"
                )
            else:
                plan = (
                    f"{strategy}: generate({_short_model(config.small_model.id)}) "
                    f"verify({_short_model(config.large_model.id)})"
                )
            print(f"  - {plan}")
        print(f"Total planned runs: {len(config.strategies) * len(tasks)}")
        return

    store = ResultStore(args.db)
    strategy_runs: Dict[str, List[RunResult]] = defaultdict(list)
    existing_runs = store.get_runs(experiment_name=config.name)
    existing_run_ids = {run.id for run in existing_runs}
    for run in existing_runs:
        if run.combination_strategy in supported_strategies:
            strategy_runs[run.combination_strategy].append(run)

    small_server_config = ServerConfig(
        engine=config.serving_engine,
        model_id=config.small_model.id,
        quantization="fp16",
        tp=config.small_model.tp,
        port=config.small_model.port,
        python_path=config.python_path,
        env_vars={"CUDA_VISIBLE_DEVICES": config.small_model.cuda_visible_devices}
        if config.small_model.cuda_visible_devices
        else {},
    )
    large_server_config = ServerConfig(
        engine=config.serving_engine,
        model_id=config.large_model.id,
        quantization="fp16",
        tp=config.large_model.tp,
        port=config.large_model.port,
        python_path=config.python_path,
        env_vars={"CUDA_VISIBLE_DEVICES": config.large_model.cuda_visible_devices}
        if config.large_model.cuda_visible_devices
        else {},
    )

    small_server = ModelServerManager(small_server_config)
    large_server = ModelServerManager(large_server_config)
    try:
        print("\nLaunching model servers...")
        small_server.launch()
        large_server.launch()

        if not small_server.wait_until_ready(timeout=600):
            raise RuntimeError("Small model server failed to start")
        if not large_server.wait_until_ready(timeout=600):
            raise RuntimeError("Large model server failed to start")

        small_client = ChatClient(
            base_url=f"http://localhost:{config.small_model.port}"
        )
        large_client = ChatClient(
            base_url=f"http://localhost:{config.large_model.port}"
        )

        for strategy in config.strategies:
            print(f"\n--- Strategy: {strategy} ---")
            for i, task in enumerate(tasks, start=1):
                run_key = "|".join([config.name, strategy, task.id])
                run_id = f"combo-{uuid.uuid5(uuid.NAMESPACE_URL, run_key).hex}"
                if run_id in existing_run_ids:
                    logger.debug("Skipping %s (already exists)", run_id)
                    continue

                monitor = GPUMonitor(interval=0.5)
                started_at = datetime.now().isoformat()
                monitor.start()

                try:
                    if strategy == "single-pass-small":
                        combo_result = run_single_pass(
                            task.messages,
                            small_client,
                            config.small_model.id,
                            task.eval_config,
                            config.max_tokens,
                        )
                    elif strategy == "single-pass-large":
                        combo_result = run_single_pass(
                            task.messages,
                            large_client,
                            config.large_model.id,
                            task.eval_config,
                            config.max_tokens,
                        )
                    elif strategy == "best-of-n-small":
                        combo_result = run_best_of_n(
                            task.messages,
                            small_client,
                            config.small_model.id,
                            task.eval_config,
                            n=3,
                            temperature=0.7,
                            max_tokens=config.max_tokens,
                        )
                    elif strategy == "best-of-n-large":
                        combo_result = run_best_of_n(
                            task.messages,
                            large_client,
                            config.large_model.id,
                            task.eval_config,
                            n=3,
                            temperature=0.7,
                            max_tokens=config.max_tokens,
                        )
                    elif strategy == "cascade":
                        combo_result = run_cascade(
                            task.messages,
                            small_client,
                            config.small_model.id,
                            large_client,
                            config.large_model.id,
                            task.eval_config,
                            config.max_tokens,
                        )
                    elif strategy == "adaptive-cascade":
                        combo_result = run_adaptive_cascade(
                            task.messages,
                            small_client,
                            config.small_model.id,
                            large_client,
                            config.large_model.id,
                            task.eval_config,
                            confidence_threshold=7,
                            max_tokens=config.max_tokens,
                        )
                    elif strategy == "self-critique-small":
                        combo_result = run_self_critique(
                            task.messages,
                            small_client,
                            config.small_model.id,
                            task.eval_config,
                            config.max_tokens,
                        )
                    elif strategy == "self-critique-large":
                        combo_result = run_self_critique(
                            task.messages,
                            large_client,
                            config.large_model.id,
                            task.eval_config,
                            config.max_tokens,
                        )
                    elif strategy == "vote":
                        combo_result = run_multi_model_vote(
                            task.messages,
                            [
                                (small_client, config.small_model.id),
                                (large_client, config.large_model.id),
                            ],
                            task.eval_config,
                            config.max_tokens,
                        )
                    elif strategy == "generate-verify":
                        combo_result = run_generate_verify(
                            task.messages,
                            small_client,
                            config.small_model.id,
                            large_client,
                            config.large_model.id,
                            task.eval_config,
                            config.max_tokens,
                        )
                    else:
                        raise ValueError(f"Unsupported strategy: {strategy}")
                except Exception:
                    monitor.stop()
                    logger.exception(
                        "Run failed: strategy=%s task=%s", strategy, task.id
                    )
                    continue

                gpu_stats = monitor.stop()
                completed_at = datetime.now().isoformat()

                if strategy in {
                    "single-pass-small",
                    "self-critique-small",
                    "best-of-n-small",
                }:
                    model_id = config.small_model.id
                    model_params_b = config.small_model.params_b
                    model_arch = config.small_model.arch
                    tensor_parallel = config.small_model.tp
                elif strategy in {
                    "single-pass-large",
                    "self-critique-large",
                    "best-of-n-large",
                }:
                    model_id = config.large_model.id
                    model_params_b = config.large_model.params_b
                    model_arch = config.large_model.arch
                    tensor_parallel = config.large_model.tp
                else:
                    model_id = f"{config.small_model.id}+{config.large_model.id}"
                    model_params_b = (
                        config.small_model.params_b + config.large_model.params_b
                    )
                    model_arch = f"{config.small_model.arch}+{config.large_model.arch}"
                    tensor_parallel = config.small_model.tp + config.large_model.tp

                result = RunResult(
                    id=run_id,
                    experiment_name=config.name,
                    model_id=model_id,
                    model_params_b=model_params_b,
                    model_arch=model_arch,
                    serving_engine=config.serving_engine,
                    quantization="fp16",
                    tensor_parallel=tensor_parallel,
                    gpu_type=config.gpu_type,
                    skill_subset="all",
                    num_retries=0,
                    temperature=0.0,
                    agent_mode="combo",
                    task_id=task.id,
                    task_name=task.name,
                    repetition=0,
                    task_success=combo_result.task_success,
                    quality_score=combo_result.quality_score,
                    input_tokens=combo_result.total_input_tokens,
                    output_tokens=combo_result.total_output_tokens,
                    gpu_seconds=gpu_stats.duration_s
                    * (gpu_stats.avg_gpu_util_pct / 100.0),
                    peak_vram_mb=gpu_stats.peak_memory_used_mb,
                    latency_e2e_ms=combo_result.total_latency_ms,
                    avg_gpu_util_pct=gpu_stats.avg_gpu_util_pct,
                    avg_power_w=gpu_stats.avg_power_w,
                    output_text=combo_result.final_output,
                    trajectory_log=json.dumps(
                        {
                            "strategy": combo_result.strategy,
                            "steps": [asdict(step) for step in combo_result.steps],
                            "eval_explanation": combo_result.eval_explanation,
                        },
                        ensure_ascii=False,
                    ),
                    combination_strategy=strategy,
                    combination_detail=json.dumps(
                        [asdict(step) for step in combo_result.steps],
                        ensure_ascii=False,
                    ),
                    tool_call_count=combo_result.tool_call_count,
                    started_at=started_at,
                    completed_at=completed_at,
                )
                store.save_run(result)
                existing_run_ids.add(run_id)
                strategy_runs[strategy].append(result)

                status = (
                    "PASS"
                    if bool(combo_result.task_success)
                    else "FAIL"
                    if combo_result.task_success is not None
                    else "N/A"
                )
                print(
                    f"[{i}/{len(tasks)}] {task.id} -> {status} "
                    f"({combo_result.total_input_tokens}+{combo_result.total_output_tokens} tok, "
                    f"{combo_result.total_latency_ms / 1000.0:.1f}s)"
                )
    finally:
        store.close()
        small_server.shutdown()
        large_server.shutdown()

    print("\n═══ COMBINATION STRATEGY COMPARISON ═══")
    print(
        f"{'Strategy':<20}  {'Tasks':>5}  {'Pass':>4}  {'Fail':>4}  {'Accuracy%':>9}  "
        f"{'Avg Latency(s)':>14}  {'Avg Tokens':>10}  {'Avg TCs':>8}  {'Total GPU-sec':>13}"
    )
    print("─" * 110)
    for strategy in config.strategies:
        runs = strategy_runs.get(strategy, [])
        tasks_n = len(runs)
        pass_n = sum(1 for r in runs if bool(r.task_success))
        fail_n = tasks_n - pass_n
        accuracy = (pass_n / tasks_n * 100.0) if tasks_n else 0.0
        avg_latency_s = (
            sum(float(r.latency_e2e_ms or 0.0) for r in runs) / tasks_n / 1000.0
            if tasks_n
            else 0.0
        )
        avg_tokens = (
            sum(int(r.output_tokens or 0) for r in runs) / tasks_n if tasks_n else 0.0
        )
        avg_tool_calls = (
            sum(int(r.tool_call_count or 0) for r in runs) / tasks_n if tasks_n else 0.0
        )
        total_gpu_sec = sum(float(r.gpu_seconds or 0.0) for r in runs)
        print(
            f"{strategy:<20}  {tasks_n:>5}  {pass_n:>4}  {fail_n:>4}  {accuracy:>8.1f}%  "
            f"{avg_latency_s:>13.1f}s  {avg_tokens:>10.0f}  {avg_tool_calls:>8.1f}  {total_gpu_sec:>13.1f}"
        )


def cmd_metrics(args):
    from agent_cap.db import ResultStore
    from agent_cap.metrics.compute import compute_mcv, compute_gar
    import itertools
    import statistics

    store = ResultStore(args.db)
    runs = store.get_runs(experiment_name=args.experiment)
    store.close()

    if not runs:
        print(f"No runs found for experiment '{args.experiment}'")
        return

    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    groups = defaultdict(list)
    for r in runs:
        key = (r.model_id, r.quantization, r.model_arch, r.model_params_b)
        groups[key].append(r)

    model_ids = sorted({r.model_id for r in runs})
    quantizations = sorted({r.quantization for r in runs})
    task_ids = {r.task_id for r in runs}

    print(f"═══ EXPERIMENT: {args.experiment} ═══")
    print(
        f"Models: {len(model_ids)}  |  Quantizations: {quantizations}  |  "
        f"Tasks: {len(task_ids)}  |  Total Runs: {len(runs)}"
    )

    rows = []
    for (model_id, quant, arch, params_b), group_runs in groups.items():
        latencies_ms = [float(r.latency_e2e_ms or 0.0) for r in group_runs]
        latencies_s = [ms / 1000.0 for ms in latencies_ms]
        sorted_lats = sorted(latencies_s)

        p50_latency_s = statistics.median(sorted_lats) if sorted_lats else 0.0
        p95_latency_s = (
            sorted_lats[min(int(len(sorted_lats) * 0.95), len(sorted_lats) - 1)]
            if sorted_lats
            else 0.0
        )
        avg_latency_s = _mean(latencies_s)
        total_wall_seconds = sum(latencies_s)

        in_tokens = [int(r.input_tokens or 0) for r in group_runs]
        out_tokens = [int(r.output_tokens or 0) for r in group_runs]
        total_out_tokens = sum(out_tokens)
        avg_in_tokens = _mean(in_tokens)
        avg_out_tokens = _mean(out_tokens)
        output_tok_per_s = (
            total_out_tokens / total_wall_seconds if total_wall_seconds > 0 else 0.0
        )

        gpu_seconds = [float(r.gpu_seconds or 0.0) for r in group_runs]
        avg_gpu_seconds = _mean(gpu_seconds)
        total_gpu_seconds = sum(gpu_seconds)

        peak_vram_gb = (
            max((float(r.peak_vram_mb or 0.0) for r in group_runs), default=0.0)
            / 1024.0
        )
        avg_gpu_util = _mean([float(r.avg_gpu_util_pct or 0.0) for r in group_runs])
        avg_power_w = _mean([float(r.avg_power_w or 0.0) for r in group_runs])

        energy_wh = sum(
            float(r.avg_power_w or 0.0)
            * (float(r.latency_e2e_ms or 0.0) / 1000.0)
            / 3600.0
            for r in group_runs
        )

        pass_count = sum(1 for r in group_runs if bool(r.task_success))
        total_runs = len(group_runs)
        accuracy_pct = (pass_count / total_runs * 100.0) if total_runs else 0.0

        quality_values = [
            float(r.quality_score) for r in group_runs if r.quality_score is not None
        ]
        avg_quality = _mean(quality_values)

        quality_per_gpu_sec = (
            avg_quality / avg_gpu_seconds if avg_gpu_seconds > 0 else 0.0
        )
        tokens_per_gpu_sec = (
            total_out_tokens / total_gpu_seconds if total_gpu_seconds > 0 else 0.0
        )
        quality_per_kwh = avg_quality / (energy_wh / 1000.0) if energy_wh > 0 else 0.0

        rows.append(
            {
                "model_id": model_id,
                "model_short": _short_model(model_id),
                "model_label": f"{_short_model(model_id)}@{quant}",
                "quant": quant,
                "arch": arch or "-",
                "params_b": float(params_b or 0.0),
                "pass_count": pass_count,
                "total_runs": total_runs,
                "accuracy_pct": accuracy_pct,
                "avg_quality": avg_quality,
                "avg_latency_s": avg_latency_s,
                "p50_latency_s": p50_latency_s,
                "p95_latency_s": p95_latency_s,
                "output_tok_per_s": output_tok_per_s,
                "avg_in_tokens": avg_in_tokens,
                "avg_out_tokens": avg_out_tokens,
                "avg_gpu_seconds": avg_gpu_seconds,
                "total_gpu_seconds": total_gpu_seconds,
                "peak_vram_gb": peak_vram_gb,
                "avg_gpu_util": avg_gpu_util,
                "avg_power_w": avg_power_w,
                "energy_wh": energy_wh,
                "quality_per_gpu_sec": quality_per_gpu_sec,
                "tokens_per_gpu_sec": tokens_per_gpu_sec,
                "quality_per_kwh": quality_per_kwh,
                "total_wall_seconds": total_wall_seconds,
                "gar": compute_gar(total_gpu_seconds, total_wall_seconds),
            }
        )

    rows.sort(key=lambda x: (x["params_b"], x["model_id"], x["quant"]))

    print("\n── Accuracy & Quality (per model) ──")
    print(
        f"{'Model':<26} {'Arch':<10} {'Params':>8} {'Quant':<10} "
        f"{'Accuracy (pass/total = %)':>28} {'Avg Score':>14}"
    )
    print("─" * 106)
    for row in rows:
        acc_text = (
            f"{row['pass_count']}/{row['total_runs']} = {row['accuracy_pct']:.1f}%"
        )
        print(
            f"{row['model_short']:<26.26} {row['arch']:<10.10} {row['params_b']:>8.1f} "
            f"{row['quant']:<10.10} {acc_text:>28} {row['avg_quality']:>9.2f}/5.0"
        )

    print("\n── Latency & Throughput (per model) ──")
    print(
        f"{'Model':<30} {'Avg Latency (s)':>15} {'P50 Lat (s)':>12} {'P95 Lat (s)':>12} "
        f"{'Output Tok/s':>13} {'Avg In Tokens':>14} {'Avg Out Tokens':>15}"
    )
    print("─" * 122)
    for row in rows:
        print(
            f"{row['model_label']:<30.30} {row['avg_latency_s']:>15.2f} {row['p50_latency_s']:>12.2f} "
            f"{row['p95_latency_s']:>12.2f} {row['output_tok_per_s']:>13.1f} "
            f"{row['avg_in_tokens']:>14.1f} {row['avg_out_tokens']:>15.1f}"
        )

    print("\n── Cost & Resources (per model) ──")
    print(
        f"{'Model':<30} {'Avg GPU-sec':>12} {'Total GPU-sec':>14} {'Peak VRAM (GB)':>15} "
        f"{'Avg GPU Util (%)':>16} {'Avg Power (W)':>14} {'Energy (Wh)':>12} {'Total Wall (s)':>14}"
    )
    print("─" * 138)
    for row in rows:
        print(
            f"{row['model_label']:<30.30} {row['avg_gpu_seconds']:>12.2f} {row['total_gpu_seconds']:>14.1f} "
            f"{row['peak_vram_gb']:>15.2f} {row['avg_gpu_util']:>16.1f} {row['avg_power_w']:>14.1f} "
            f"{row['energy_wh']:>12.2f} {row['total_wall_seconds']:>14.1f}"
        )

    print("\n── Efficiency Ratios (per model) ──")
    print(
        f"{'Model':<30} {'Quality/GPU-sec':>17} {'Tokens/GPU-sec':>16} {'Quality/kWh':>14}"
    )
    print("─" * 83)
    for row in rows:
        print(
            f"{row['model_label']:<30.30} {row['quality_per_gpu_sec']:>17.4f} "
            f"{row['tokens_per_gpu_sec']:>16.2f} {row['quality_per_kwh']:>14.2f}"
        )

    print("\n── Novel Metrics ──")

    print("GAR (total_gpu_seconds / total_wall_seconds):")
    print(f"{'Model':<30} {'GAR':>10}")
    print("─" * 42)
    for row in rows:
        print(f"{row['model_label']:<30.30} {row['gar']:>10.3f}")

    print("\nMCV between model pairs:")
    print(f"{'Pair':<64} {'MCV':>12}")
    print("─" * 78)
    for before, after in itertools.combinations(rows, 2):
        mcv = compute_mcv(
            quality_before=before["avg_quality"],
            quality_after=after["avg_quality"],
            gpu_seconds_before=before["avg_gpu_seconds"],
            gpu_seconds_after=after["avg_gpu_seconds"],
        )
        mcv_text = "inf" if mcv == float("inf") else f"{mcv:.4f}"
        pair = f"{before['model_label']} -> {after['model_label']}"
        print(f"{pair:<64.64} {mcv_text:>12}")

    print("\nSDR/EPR/EPD: N/A — requires multi-step tasks")


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
                latency_ms=sum(data["latencies"]) / len(data["latencies"])
                if data["latencies"]
                else 0,
            )
        )

    frontier = compute_pareto_frontier(points)

    print(f"\nPareto Frontier ({len(frontier)}/{len(points)} configs):")
    print(f"{'Config':<50} {'Quality':>8} {'GPU-sec':>10}")
    print("-" * 70)
    for p in frontier:
        print(f"{p.config_id:<50} {p.quality:>8.2f} {p.gpu_seconds:>10.1f}")


def cmd_reeval(args):
    from agent_cap.benchmarks import load_benchmark
    from agent_cap.db import ResultStore
    from agent_cap.evaluator import EvalConfig, evaluate

    bench_name, bench_count = _parse_benchmark_spec(args.benchmark)
    tasks = load_benchmark(bench_name, bench_count)
    task_eval_configs = {t.id: t.eval_config for t in tasks}

    store = ResultStore(args.db)
    runs = store.get_runs(experiment_name=args.experiment)

    if not runs:
        store.close()
        print(f"No runs found for experiment '{args.experiment}'")
        return

    old_pass = sum(1 for r in runs if bool(r.task_success))
    updated = 0
    new_pass = 0

    for run in runs:
        eval_config = task_eval_configs.get(run.task_id)
        if not eval_config or not run.output_text:
            continue

        cfg = EvalConfig.from_dict(eval_config)
        result = evaluate(run.output_text, cfg)
        updated += 1
        if result.task_success:
            new_pass += 1

        if args.verbose:
            old_status = "PASS" if run.task_success else "FAIL"
            new_status = "PASS" if result.task_success else "FAIL"
            if old_status != new_status:
                print(
                    f"  {run.task_id}: {old_status} -> {new_status} "
                    f"(score: {run.quality_score} -> {result.quality_score})"
                )

        if args.dry_run:
            continue

        store._conn.execute(
            "UPDATE runs SET task_success = ?, quality_score = ? WHERE id = ?",
            (result.task_success, result.quality_score, run.id),
        )

    if not args.dry_run:
        store._conn.commit()
    store.close()

    print(f"\nRe-evaluated {updated} runs for '{args.experiment}'")
    print(
        f"  Before: {old_pass}/{len(runs)} passed ({old_pass / len(runs) * 100:.1f}%)"
    )
    if updated:
        print(
            f"  After:  {new_pass}/{updated} passed ({new_pass / updated * 100:.1f}%)"
        )


def cmd_single_agent(args):
    """Run single-agent performance benchmark."""
    import json as _json

    from agent_cap.single_agent.config import SingleAgentBenchConfig
    from agent_cap.single_agent.runner import SingleAgentRunner

    config = SingleAgentBenchConfig.from_yaml(args.config)

    print("=" * 70)
    print("Single-Agent Benchmark")
    print("=" * 70)
    print(f"  Name:        {config.name}")
    print(f"  Model:       {config.model_id}")
    print(f"  Engine:      {config.serving_engine}")
    print(f"  Server:      {config.base_url}")
    print(f"  Dataset:     {config.dataset} (n={config.dataset_count})")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  Tool calls:  {'yes' if config.enable_tool_calls else 'no'}")
    print("=" * 70)

    if args.dry_run:
        print("\n--- Dry run: full config ---")
        print(_json.dumps(config.to_dict(), indent=2))
        return

    runner = SingleAgentRunner(config)
    limit = getattr(args, "limit", 0) or 0
    results, task_results = runner.run(limit=limit)

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    for m in results:
        print(
            f"  batch={m.batch_size:<3d}  mode={m.tool_mode:<12s}  "
            f"E2E_avg={m.e2e_latency_avg_ms:>8.1f}ms  "
            f"RPS={m.requests_per_second:>6.2f}  "
            f"TTFT_avg={m.ttft_avg_ms:>7.1f}ms  "
            f"TPOT_avg={m.tpot_avg_ms:>7.1f}ms  "
            f"GPU={m.avg_gpu_util_pct:>5.1f}%  "
            f"CPU={m.avg_cpu_util_pct:>5.1f}%"
        )

    out_dir = runner.save_results(results, task_results, args.output_dir)
    print(f"\nResults saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        prog="agent-cap",
        description="AgentCAP — Experiment sweep engine for multi-skill agent evaluation",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run an experiment from YAML config")
    p_run.add_argument("config", help="Path to experiment YAML config")
    p_run.add_argument("--tasks", help="Path to tasks JSON file")
    p_run.add_argument(
        "--db", default="results/experiments.db", help="Results database path"
    )
    p_run.add_argument(
        "--dry-run", action="store_true", help="List configs without running"
    )
    p_run.add_argument(
        "--benchmark",
        help="Load public benchmark: gsm8k:N or humaneval:N (N = num tasks, default 50)",
    )
    p_run.add_argument("-v", "--verbose", action="store_true")

    p_combo = sub.add_parser("combo", help="Run multi-agent combination strategies")
    p_combo.add_argument("config", help="Path to combo YAML config")
    p_combo.add_argument(
        "--benchmark", required=True, help="Benchmark: bigcodebench:50"
    )
    p_combo.add_argument(
        "--db", default="results/combo.db", help="Results database path"
    )
    p_combo.add_argument(
        "--dry-run", action="store_true", help="Show plan without running"
    )
    p_combo.add_argument("-v", "--verbose", action="store_true")

    p_metrics = sub.add_parser("metrics", help="Compute metrics from results")
    p_metrics.add_argument("experiment", help="Experiment name")
    p_metrics.add_argument("--db", default="results/experiments.db")

    p_pareto = sub.add_parser("pareto", help="Compute Pareto frontier")
    p_pareto.add_argument("experiment", help="Experiment name")
    p_pareto.add_argument("--db", default="results/experiments.db")

    p_reeval = sub.add_parser(
        "reeval", help="Re-evaluate stored results with current evaluator"
    )
    p_reeval.add_argument("experiment", help="Experiment name")
    p_reeval.add_argument(
        "--benchmark", required=True, help="Benchmark for eval configs: bigcodebench:50"
    )
    p_reeval.add_argument("--db", default="results/experiments.db")
    p_reeval.add_argument(
        "--dry-run", action="store_true", help="Recompute without writing to DB"
    )
    p_reeval.add_argument("-v", "--verbose", action="store_true")

    p_single = sub.add_parser(
        "single-agent",
        help="Run single-agent performance benchmark (batch-size sweep)",
    )
    p_single.add_argument("config", help="Path to single-agent YAML config")
    p_single.add_argument(
        "--output-dir", default=None, help="Override output directory"
    )
    p_single.add_argument(
        "--dry-run", action="store_true", help="Print config without running"
    )
    p_single.add_argument("-v", "--verbose", action="store_true")
    p_single.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run first N tasks (0 = all, useful for quick testing)",
    )

    args = parser.parse_args()

    if args.command == "run":
        if hasattr(args, "verbose") and args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        cmd_run(args)
    elif args.command == "combo":
        if hasattr(args, "verbose") and args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        cmd_combo(args)
    elif args.command == "metrics":
        cmd_metrics(args)
    elif args.command == "pareto":
        cmd_pareto(args)
    elif args.command == "reeval":
        if hasattr(args, "verbose") and args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        cmd_reeval(args)
    elif args.command == "single-agent":
        if hasattr(args, "verbose") and args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        cmd_single_agent(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
