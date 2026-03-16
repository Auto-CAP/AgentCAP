"""Single-agent benchmark runner with real multi-turn tool execution.

For each task the runner performs a full agentic loop:

    model generates response
       ↓ tool_calls?
    ToolExecutor runs real commands (file I/O, shell, grep)
       ↓ results appended to messages
    model continues … (up to max_turns)

After the loop the runner extracts the model's patch from the final
response and writes SWE-bench-compatible ``predictions.jsonl`` so
results can be evaluated with the official Docker harness::

    python -m swebench.harness.run_evaluation \\
        --predictions_path results/single_agent/predictions.jsonl \\
        --run_id my_run
"""

import csv
import json
import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_cap.benchmarks import load_benchmark
from agent_cap.server.cpu_monitor import CPUMonitor
from agent_cap.server.gpu_monitor import GPUMonitor
from agent_cap.server.streaming_client import StreamingChatClient, StreamingChatResponse
from agent_cap.single_agent.config import SingleAgentBenchConfig
from agent_cap.single_agent.metrics import BenchmarkMetrics, aggregate_metrics
from agent_cap.single_agent.tool_executor import (
    TOOL_DEFINITIONS,
    ToolCallResult,
    ToolExecutor,
)

logger = logging.getLogger("agent_cap.single_agent")


# ------------------------------------------------------------------
# Patch extraction helpers
# ------------------------------------------------------------------


def _extract_patch_from_text(text: str) -> str:
    """Pull the first ```diff block from model output."""
    m = re.search(r"```diff\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(diff --git.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _git_diff(workspace: Path) -> str:
    """Run ``git diff`` in the workspace to capture uncommitted changes."""
    try:
        proc = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(workspace),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(workspace),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass

    return ""


class SingleAgentRunner:
    """Run a single-agent performance benchmark across batch sizes.

    Usage::

        config = SingleAgentBenchConfig.from_yaml("configs/single_agent.yaml")
        runner = SingleAgentRunner(config)
        results, predictions = runner.run()
        runner.save_results(results, predictions)
    """

    def __init__(self, config: SingleAgentBenchConfig) -> None:
        self.config = config
        self.client = StreamingChatClient(base_url=config.base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self, limit: int = 0
    ) -> Tuple[List[BenchmarkMetrics], List[Dict[str, str]]]:
        """Execute the full benchmark sweep.

        Returns:
            (metrics_list, predictions) where predictions is a list of
            SWE-bench-format dicts ready for ``predictions.jsonl``.
        """
        tasks = load_benchmark(self.config.dataset, self.config.dataset_count)
        if limit > 0:
            tasks = tasks[:limit]
        logger.info("Loaded %d tasks from '%s'", len(tasks), self.config.dataset)

        all_messages = [t.messages for t in tasks]
        eval_configs = [t.eval_config or {} for t in tasks]

        results: List[BenchmarkMetrics] = []
        predictions: List[Dict[str, str]] = []

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
                    metrics, preds = self._run_batch(
                        all_messages,
                        eval_configs,
                        batch_size,
                        tool_mode,
                    )
                    results.append(metrics)
                    predictions.extend(preds)
                    self._print_summary(metrics)

        return results, predictions

    def save_results(
        self,
        results: List[BenchmarkMetrics],
        predictions: Optional[List[Dict[str, str]]] = None,
        output_dir: Optional[str] = None,
    ) -> Path:
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # metrics.json
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

        # metrics.csv
        csv_path = out / "metrics.csv"
        if results:
            fieldnames = list(results[0].to_dict().keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in results:
                    writer.writerow(m.to_dict())
        logger.info("Wrote %s", csv_path)

        # predictions.jsonl  (SWE-bench harness format)
        if predictions:
            pred_path = out / "predictions.jsonl"
            with open(pred_path, "w", encoding="utf-8") as f:
                for p in predictions:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            logger.info("Wrote %s  (%d predictions)", pred_path, len(predictions))

        return out

    # ------------------------------------------------------------------
    # Internal – batch orchestration
    # ------------------------------------------------------------------

    def _run_batch(
        self,
        all_messages: List[List[Dict[str, Any]]],
        eval_configs: List[Dict[str, Any]],
        batch_size: int,
        tool_mode: str,
    ) -> Tuple[BenchmarkMetrics, List[Dict[str, str]]]:
        gpu_mon = GPUMonitor(interval=self.config.gpu_monitor_interval)
        cpu_mon = CPUMonitor(interval=self.config.cpu_monitor_interval)
        gpu_mon.start()
        cpu_mon.start()

        t_start = time.perf_counter()

        if tool_mode == "with_tools":
            responses, tc_latencies, preds = self._run_batch_with_tools(
                all_messages,
                eval_configs,
                batch_size,
            )
        else:
            responses = self.client.chat_batch(
                messages_list=all_messages,
                model=self.config.model_id,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                concurrency=batch_size,
            )
            tc_latencies = []
            preds = self._build_predictions(responses, eval_configs)

        wall_clock_s = time.perf_counter() - t_start

        gpu_stats = gpu_mon.stop()
        cpu_stats = cpu_mon.stop()

        metrics = aggregate_metrics(
            responses=responses,
            batch_size=batch_size,
            tool_mode=tool_mode,
            wall_clock_s=wall_clock_s,
            gpu_avg_util=gpu_stats.avg_gpu_util_pct,
            gpu_max_util=gpu_stats.max_gpu_util_pct,
            cpu_avg_util=cpu_stats.avg_cpu_util_pct,
            cpu_max_util=cpu_stats.max_cpu_util_pct,
            tool_call_latencies_ms=tc_latencies or None,
        )
        return metrics, preds

    def _build_predictions(
        self,
        responses: List[StreamingChatResponse],
        eval_configs: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Build SWE-bench prediction dicts from no-tool responses."""
        preds = []
        for resp, ec in zip(responses, eval_configs):
            patch = _extract_patch_from_text(resp.content)
            preds.append(
                {
                    "instance_id": ec.get("instance_id", ""),
                    "model_name_or_path": self.config.model_id,
                    "model_patch": patch,
                }
            )
        return preds

    def _run_batch_with_tools(
        self,
        all_messages: List[List[Dict[str, Any]]],
        eval_configs: List[Dict[str, Any]],
        concurrency: int,
    ) -> Tuple[List[StreamingChatResponse], List[float], List[Dict[str, str]]]:
        """Run all tasks through the agentic loop in parallel."""
        n = len(all_messages)
        all_responses: List[Optional[StreamingChatResponse]] = [None] * n
        all_tc_lats: List[List[float]] = [[] for _ in range(n)]
        all_patches: List[str] = [""] * n

        def _run_one(idx: int, msgs: List[Dict[str, Any]]) -> None:
            resp, lats, patch = self._agentic_loop(list(msgs))
            all_responses[idx] = resp
            all_tc_lats[idx] = lats
            all_patches[idx] = patch

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_run_one, i, msgs): i for i, msgs in enumerate(all_messages)
            }
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    idx = futures[fut]
                    logger.error("Task %d failed: %s", idx, exc)
                    all_responses[idx] = StreamingChatResponse(
                        content="",
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        latency_ms=0,
                        ttft_ms=0,
                        tpot_ms_avg=0,
                        tpot_ms_p99=0,
                        model=self.config.model_id,
                        error=str(exc),
                    )

        flat_lats: List[float] = []
        for lats in all_tc_lats:
            flat_lats.extend(lats)

        preds: List[Dict[str, str]] = []
        for i, ec in enumerate(eval_configs):
            preds.append(
                {
                    "instance_id": ec.get("instance_id", ""),
                    "model_name_or_path": self.config.model_id,
                    "model_patch": all_patches[i],
                }
            )

        return (
            [r for r in all_responses if r is not None],
            flat_lats,
            preds,
        )

    # ------------------------------------------------------------------
    # Core agentic loop (single task)
    # ------------------------------------------------------------------

    def _agentic_loop(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[StreamingChatResponse, List[float], str]:
        """Run model ↔ tool loop, return (response, tool_latencies, patch)."""
        tools = self.config.tool_definitions or TOOL_DEFINITIONS
        executor = ToolExecutor(
            workspace_dir=self.config.workspace_dir,
            shell_timeout=self.config.shell_timeout,
        )

        all_tc_latencies: List[float] = []
        cumulative_input = 0
        cumulative_output = 0
        cumulative_latency = 0.0
        first_ttft: Optional[float] = None
        all_tpot_avgs: List[float] = []
        last_tpot_p99 = 0.0
        total_tool_calls = 0
        final_content = ""

        for turn in range(self.config.max_turns):
            resp = self.client.chat(
                messages=messages,
                model=self.config.model_id,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                tools=tools,
            )

            cumulative_input += resp.input_tokens
            cumulative_output += resp.output_tokens
            cumulative_latency += resp.latency_ms
            if first_ttft is None:
                first_ttft = resp.ttft_ms
            if resp.tpot_ms_avg > 0:
                all_tpot_avgs.append(resp.tpot_ms_avg)
            if resp.tpot_ms_p99 > 0:
                last_tpot_p99 = resp.tpot_ms_p99

            if resp.tool_call_count == 0 or not resp.raw_chunks:
                final_content = resp.content
                break

            pending_calls = self._extract_tool_calls(resp.raw_chunks)

            if not pending_calls:
                final_content = resp.content
                break

            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if resp.content:
                assistant_msg["content"] = resp.content
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in pending_calls
            ]
            messages.append(assistant_msg)

            for tc in pending_calls:
                try:
                    args = json.loads(tc["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc["arguments"]}

                result = executor.execute(
                    tool_name=tc["name"],
                    tool_call_id=tc["id"],
                    arguments=args,
                )
                all_tc_latencies.append(result.latency_ms)
                total_tool_calls += 1

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result.output,
                    }
                )

                logger.debug(
                    "  turn=%d  tool=%s  ok=%s  %.1fms",
                    turn,
                    tc["name"],
                    result.success,
                    result.latency_ms,
                )

        # --- Extract patch ---
        # Priority: git diff in workspace > diff block in final response
        patch = _git_diff(Path(self.config.workspace_dir))
        if not patch:
            patch = _extract_patch_from_text(final_content)

        avg_tpot = sum(all_tpot_avgs) / len(all_tpot_avgs) if all_tpot_avgs else 0.0

        combined = StreamingChatResponse(
            content=final_content,
            input_tokens=cumulative_input,
            output_tokens=cumulative_output,
            total_tokens=cumulative_input + cumulative_output,
            latency_ms=cumulative_latency,
            ttft_ms=first_ttft or 0.0,
            tpot_ms_avg=avg_tpot,
            tpot_ms_p99=last_tpot_p99,
            model=self.config.model_id,
            tool_call_count=total_tool_calls,
        )
        return combined, all_tc_latencies, patch

    @staticmethod
    def _extract_tool_calls(
        raw_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Reassemble tool_calls from streaming delta chunks."""
        fragments: Dict[int, Dict[str, str]] = {}
        for chunk in raw_chunks:
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            tc_list = delta.get("tool_calls")
            if not tc_list:
                continue
            for tc in tc_list:
                idx = tc.get("index", 0)
                if idx not in fragments:
                    fragments[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.get("id"):
                    fragments[idx]["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    fragments[idx]["name"] = fn["name"]
                if fn.get("arguments"):
                    fragments[idx]["arguments"] += fn["arguments"]

        return [v for _, v in sorted(fragments.items()) if v["name"]]

    @staticmethod
    def _print_summary(m: BenchmarkMetrics) -> None:
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
