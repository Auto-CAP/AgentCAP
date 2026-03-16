"""Single-agent benchmark runner.

Orchestrates a full benchmark sweep:

1. Load tasks from the configured dataset.
2. For each batch_size in the sweep:
   a. Run WITHOUT tool calls → collect metrics
   b. Run WITH tool calls (if enabled) → collect metrics
3. Aggregate and report all results.

Hardware monitors (GPU via nvidia-smi, CPU via /proc/stat) run in the
background during each batch.
"""

import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_cap.benchmarks import load_benchmark
from agent_cap.server.cpu_monitor import CPUMonitor
from agent_cap.server.gpu_monitor import GPUMonitor
from agent_cap.server.streaming_client import StreamingChatClient, StreamingChatResponse
from agent_cap.single_agent.config import SingleAgentBenchConfig
from agent_cap.single_agent.metrics import BenchmarkMetrics, aggregate_metrics
from agent_cap.single_agent.tool_simulator import (
    DEFAULT_TOOL_DEFINITIONS,
    simulate_tool_execution,
)

logger = logging.getLogger("agent_cap.single_agent")


class SingleAgentRunner:
    """Run a single-agent performance benchmark across batch sizes.

    Usage::

        config = SingleAgentBenchConfig.from_yaml("configs/single_agent.yaml")
        runner = SingleAgentRunner(config)
        results = runner.run()
        runner.save_results(results, "results/single_agent")
    """

    def __init__(self, config: SingleAgentBenchConfig) -> None:
        self.config = config
        self.client = StreamingChatClient(base_url=config.base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[BenchmarkMetrics]:
        """Execute the full benchmark sweep.

        Returns:
            List of BenchmarkMetrics (one per batch_size × tool_mode combo).
        """
        tasks = load_benchmark(self.config.dataset, self.config.dataset_count)
        logger.info("Loaded %d tasks from '%s'", len(tasks), self.config.dataset)

        all_messages = [t.messages for t in tasks]
        results: List[BenchmarkMetrics] = []

        tool_modes = ["no_tools"]
        if self.config.enable_tool_calls:
            tool_modes.append("with_tools")

        for batch_size in self.config.batch_sizes:
            for tool_mode in tool_modes:
                for rep in range(self.config.repetitions):
                    logger.info(
                        "batch_size=%d  tool_mode=%s  rep=%d/%d",
                        batch_size,
                        tool_mode,
                        rep + 1,
                        self.config.repetitions,
                    )
                    metrics = self._run_batch(all_messages, batch_size, tool_mode)
                    results.append(metrics)
                    self._print_summary(metrics)

        return results

    def save_results(
        self, results: List[BenchmarkMetrics], output_dir: Optional[str] = None
    ) -> Path:
        """Save benchmark results to CSV + JSON.

        Args:
            results: Benchmark metrics from ``run()``.
            output_dir: Override for config.output_dir.

        Returns:
            Path to the output directory.
        """
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON (full)
        json_path = out / "metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": self.config.to_dict(),
                    "results": [m.to_dict() for m in results],
                },
                f,
                indent=2,
            )
        logger.info("Wrote %s", json_path)

        # CSV (flat table)
        csv_path = out / "metrics.csv"
        if results:
            fieldnames = list(results[0].to_dict().keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in results:
                    writer.writerow(m.to_dict())
        logger.info("Wrote %s", csv_path)

        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_batch(
        self,
        all_messages: List[List[Dict[str, Any]]],
        batch_size: int,
        tool_mode: str,
    ) -> BenchmarkMetrics:
        """Run one (batch_size, tool_mode) configuration."""
        tools = None
        if tool_mode == "with_tools":
            tools = self.config.tool_definitions or DEFAULT_TOOL_DEFINITIONS

        # Start hardware monitors
        gpu_mon = GPUMonitor(interval=self.config.gpu_monitor_interval)
        cpu_mon = CPUMonitor(interval=self.config.cpu_monitor_interval)
        gpu_mon.start()
        cpu_mon.start()

        t_start = time.perf_counter()

        # Send all requests with specified concurrency
        responses = self.client.chat_batch(
            messages_list=all_messages,
            model=self.config.model_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tools=tools,
            concurrency=batch_size,
        )

        t_end = time.perf_counter()
        wall_clock_s = t_end - t_start

        # Collect tool-call latencies (simulated) if in tool mode
        tool_call_latencies_ms: List[float] = []
        if tool_mode == "with_tools":
            for resp in responses:
                if resp.tool_call_count > 0:
                    # Simulate tool execution latency per call
                    for _ in range(resp.tool_call_count):
                        tc_result = simulate_tool_execution(
                            "read_file", {"path": "/tmp/example.py"}
                        )
                        tool_call_latencies_ms.append(tc_result["latency_ms"])

        # Stop monitors
        gpu_stats = gpu_mon.stop()
        cpu_stats = cpu_mon.stop()

        return aggregate_metrics(
            responses=responses,
            batch_size=batch_size,
            tool_mode=tool_mode,
            wall_clock_s=wall_clock_s,
            gpu_avg_util=gpu_stats.avg_gpu_util_pct,
            gpu_max_util=gpu_stats.max_gpu_util_pct,
            cpu_avg_util=cpu_stats.avg_cpu_util_pct,
            cpu_max_util=cpu_stats.max_cpu_util_pct,
            tool_call_latencies_ms=tool_call_latencies_ms or None,
        )

    @staticmethod
    def _print_summary(m: BenchmarkMetrics) -> None:
        """Print a compact one-line summary for a benchmark run."""
        print(
            f"  batch={m.batch_size:<3d}  mode={m.tool_mode:<12s}  "
            f"E2E_avg={m.e2e_latency_avg_ms:>8.1f}ms  "
            f"RPS={m.requests_per_second:>6.2f}  "
            f"TTFT_avg={m.ttft_avg_ms:>7.1f}ms  "
            f"TPOT_avg={m.tpot_avg_ms:>7.1f}ms  "
            f"in_tok={m.total_input_tokens:>7d}  "
            f"out_tok={m.total_output_tokens:>7d}  "
            f"tools={m.total_tool_calls:>3d}  "
            f"GPU={m.avg_gpu_util_pct:>5.1f}%  "
            f"CPU={m.avg_cpu_util_pct:>5.1f}%  "
            f"errs={m.error_count}"
        )
