"""Single-agent benchmark runner with real multi-turn tool execution.

with_tools mode (agentic):
    1. Clone repo at base_commit into workspace
    2. Agent uses tool calls (read/write/shell/grep) to fix the issue
    3. Run fail_to_pass tests directly → resolved or not

no_tools mode (direct patch):
    1. Model generates a patch in one shot
    2. Write predictions.jsonl for offline harness evaluation
"""

import csv
import json
import logging
import os
import re
import shutil
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
from agent_cap.single_agent.local_env import LocalWorkspace

logger = logging.getLogger("agent_cap.single_agent")


# ------------------------------------------------------------------
# Repo + test helpers
# ------------------------------------------------------------------


def _clone_repo(repo: str, base_commit: str, workspace: Path) -> bool:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    repo_url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(
            ["git", "clone", "--depth=50", repo_url, str(workspace)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "checkout", base_commit],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(workspace),
        )
        return True
    except Exception as exc:
        logger.error("Clone failed for %s@%s: %s", repo, base_commit[:12], exc)
        return False


def _apply_test_patch(test_patch: str, workspace: Path) -> bool:
    if not test_patch or not test_patch.strip():
        return True
    try:
        proc = subprocess.run(
            ["git", "apply", "--allow-empty"],
            input=test_patch,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(workspace),
        )
        return proc.returncode == 0
    except Exception:
        return False


def _run_tests(
    fail_to_pass: str, workspace: Path, timeout: int = 120
) -> Dict[str, Any]:
    if not fail_to_pass:
        return {"passed": False, "reason": "no fail_to_pass tests defined"}

    try:
        tests = (
            json.loads(fail_to_pass) if isinstance(fail_to_pass, str) else fail_to_pass
        )
    except json.JSONDecodeError:
        tests = [fail_to_pass]

    passed_count = 0
    total = len(tests)
    details = []

    for test_spec in tests:
        test_file = (
            test_spec.split("::")[0].split(" | ")[0].strip()
            if "::" in test_spec or " | " in test_spec
            else test_spec
        )
        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", test_file, "-x", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workspace),
            )
            ok = proc.returncode == 0
            if ok:
                passed_count += 1
            details.append(
                {
                    "test": test_spec[:100],
                    "passed": ok,
                    "output": (proc.stdout + proc.stderr)[-500:],
                }
            )
        except subprocess.TimeoutExpired:
            details.append(
                {"test": test_spec[:100], "passed": False, "output": "timeout"}
            )
        except Exception as exc:
            details.append(
                {"test": test_spec[:100], "passed": False, "output": str(exc)}
            )

    return {
        "passed": passed_count == total,
        "passed_count": passed_count,
        "total": total,
        "details": details,
    }


def _git_diff(workspace: Path) -> str:
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
    return ""


def _extract_patch_from_text(text: str) -> str:
    m = re.search(r"```diff\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(diff --git.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


class SingleAgentRunner:
    def __init__(self, config: SingleAgentBenchConfig) -> None:
        self.config = config
        self.client = StreamingChatClient(base_url=config.base_url)
        self._resolve_model_id()

    def _resolve_model_id(self) -> None:
        server_id = self.client.get_server_model_id()
        if server_id and server_id != self.config.model_id:
            logger.info(
                "Server model: %s (config had: %s) — using server model",
                server_id,
                self.config.model_id,
            )
            self.config.model_id = server_id
        elif server_id:
            logger.info("Server model confirmed: %s", server_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self, limit: int = 0
    ) -> Tuple[List[BenchmarkMetrics], List[Dict[str, Any]]]:
        tasks = load_benchmark(self.config.dataset, self.config.dataset_count)
        if limit > 0:
            tasks = tasks[:limit]
        logger.info("Loaded %d tasks from '%s'", len(tasks), self.config.dataset)

        all_messages = [t.messages for t in tasks]
        eval_configs = [t.eval_config or {} for t in tasks]

        results: List[BenchmarkMetrics] = []
        task_results: List[Dict[str, Any]] = []

        tool_mode = "with_tools" if self.config.enable_tool_calls else "no_tools"
        logger.info("Mode: %s", tool_mode)

        num_tasks = len(tasks)

        for batch_size in self.config.batch_sizes:
            effective_bs = min(batch_size, num_tasks)
            for rep in range(self.config.repetitions):
                logger.info(
                    "batch_size=%d (effective=%d, tasks=%d)  tool_mode=%s  rep=%d/%d",
                    batch_size,
                    effective_bs,
                    num_tasks,
                    tool_mode,
                    rep + 1,
                    self.config.repetitions,
                )
                metrics, tr = self._run_batch(
                    all_messages,
                    eval_configs,
                    effective_bs,
                    tool_mode,
                )
                metrics.batch_size = batch_size
                for t in tr:
                    t["batch_size"] = batch_size
                results.append(metrics)
                task_results.extend(tr)
                self._print_summary(metrics)

        return results, task_results

    def save_results(
        self,
        results: List[BenchmarkMetrics],
        task_results: Optional[List[Dict[str, Any]]] = None,
        output_dir: Optional[str] = None,
    ) -> Path:
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

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

        csv_path = out / "metrics.csv"
        if results:
            fieldnames = list(results[0].to_dict().keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in results:
                    writer.writerow(m.to_dict())
        logger.info("Wrote %s", csv_path)

        if task_results:
            tr_path = out / "task_results.jsonl"
            with open(tr_path, "w", encoding="utf-8") as f:
                for tr in task_results:
                    f.write(json.dumps(tr, ensure_ascii=False, default=str) + "\n")
            logger.info("Wrote %s (%d entries)", tr_path, len(task_results))

            preds = [
                tr
                for tr in task_results
                if "model_patch" in tr and tr.get("instance_id")
            ]
            if preds:
                pred_path = out / "predictions.jsonl"
                with open(pred_path, "w", encoding="utf-8") as f:
                    for p in preds:
                        f.write(
                            json.dumps(
                                {
                                    "instance_id": p["instance_id"],
                                    "model_name_or_path": p.get(
                                        "model_name_or_path", ""
                                    ),
                                    "model_patch": p["model_patch"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                logger.info("Wrote %s (%d predictions)", pred_path, len(preds))
                print(f"\n  predictions.jsonl: {pred_path}")
                print(
                    "  Evaluate with:\n"
                    "    python -m swebench.harness.run_evaluation \\\n"
                    f"      --predictions_path {pred_path} \\\n"
                    "      --run_id my_run --max_workers 4"
                )

            patches_with_content = sum(1 for p in preds if p.get("model_patch"))
            print(f"  Patches generated: {patches_with_content}/{len(preds)}")

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
    ) -> Tuple[BenchmarkMetrics, List[Dict[str, Any]]]:
        gpu_mon = GPUMonitor(interval=self.config.gpu_monitor_interval)
        cpu_mon = CPUMonitor(interval=self.config.cpu_monitor_interval)
        gpu_mon.start()
        cpu_mon.start()

        metrics_before = self.client.scrape_server_metrics()
        t_start = time.perf_counter()

        if tool_mode == "with_tools":
            responses, tc_latencies, tr = self._run_batch_with_tools(
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
                stop_token_ids=self.config.stop_token_ids,
            )
            tc_latencies = []
            tr = self._build_no_tools_results(responses, eval_configs)

        wall_clock_s = time.perf_counter() - t_start
        metrics_after = self.client.scrape_server_metrics()

        gpu_stats = gpu_mon.stop()
        cpu_stats = cpu_mon.stop()

        server_ttft_ms, server_tpot_ms = self.client.compute_server_tpot(
            metrics_before, metrics_after
        )

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

        if server_tpot_ms > 0:
            metrics.tpot_avg_ms = server_tpot_ms
            metrics.tpot_p99_ms = server_tpot_ms
        if server_ttft_ms > 0 and metrics.ttft_avg_ms == 0:
            metrics.ttft_avg_ms = server_ttft_ms
            metrics.ttft_p99_ms = server_ttft_ms

        return metrics, tr

    def _build_no_tools_results(
        self,
        responses: List[StreamingChatResponse],
        eval_configs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results = []
        for resp, ec in zip(responses, eval_configs):
            patch = _extract_patch_from_text(resp.content)
            results.append(
                {
                    "instance_id": ec.get("instance_id", ""),
                    "model_name_or_path": self.config.model_id,
                    "model_patch": patch,
                    "resolved": None,
                }
            )
        return results

    def _run_batch_with_tools(
        self,
        all_messages: List[List[Dict[str, Any]]],
        eval_configs: List[Dict[str, Any]],
        concurrency: int,
    ) -> Tuple[List[StreamingChatResponse], List[float], List[Dict[str, Any]]]:
        n = len(all_messages)
        all_responses: List[Optional[StreamingChatResponse]] = [None] * n
        all_tc_lats: List[List[float]] = [[] for _ in range(n)]
        all_task_results: List[Dict[str, Any]] = [{}] * n

        def _run_one(idx: int, msgs: List[Dict[str, Any]], ec: Dict[str, Any]) -> None:
            resp, lats, task_result = self._run_single_task(list(msgs), ec)
            all_responses[idx] = resp
            all_tc_lats[idx] = lats
            all_task_results[idx] = task_result

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_run_one, i, msgs, ec): i
                for i, (msgs, ec) in enumerate(zip(all_messages, eval_configs))
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
                    all_task_results[idx] = {
                        "instance_id": eval_configs[idx].get("instance_id", ""),
                        "resolved": False,
                        "error": str(exc),
                    }

        flat_lats: List[float] = []
        for lats in all_tc_lats:
            flat_lats.extend(lats)

        return (
            [r for r in all_responses if r is not None],
            flat_lats,
            all_task_results,
        )

    # ------------------------------------------------------------------
    # Single task: clone → agentic loop → run tests
    # ------------------------------------------------------------------

    def _run_single_task(
        self,
        messages: List[Dict[str, Any]],
        eval_config: Dict[str, Any],
    ) -> Tuple[StreamingChatResponse, List[float], Dict[str, Any]]:
        instance_id = eval_config.get("instance_id", "unknown")
        repo = eval_config.get("repo", "")
        base_commit = eval_config.get("base_commit", "")
        test_patch = eval_config.get("test_patch", "")
        fail_to_pass = eval_config.get(
            "FAIL_TO_PASS", eval_config.get("fail_to_pass", "")
        )

        error_result = lambda err: (
            StreamingChatResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=0,
                ttft_ms=0,
                tpot_ms_avg=0,
                tpot_ms_p99=0,
                model=self.config.model_id,
                error=err,
            ),
            [],
            {
                "instance_id": instance_id,
                "model_name_or_path": self.config.model_id,
                "model_patch": "",
                "error": err,
            },
        )

        ws = LocalWorkspace(eval_config, base_dir=self.config.workspace_dir)

        logger.info("[%s] Setting up local environment...", instance_id[:30])
        if not ws.setup():
            return error_result("environment setup failed")

        try:
            agentic_prompt = (
                f"You are a software engineer. Fix the following issue in the repo "
                f"at {ws.workspace}.\n\n"
                f"{messages[0]['content']}\n\n"
                "Use the available tools (read_file, write_file, run_shell, "
                "search_code) to explore the codebase and make the fix."
            )
            agentic_messages: List[Dict[str, Any]] = [
                {"role": "user", "content": agentic_prompt}
            ]

            logger.info(
                "[%s] Agentic loop (max_turns=%d)",
                instance_id[:30],
                self.config.max_turns,
            )
            resp, tc_lats = self._agentic_loop(agentic_messages, str(ws.workspace))

            patch = ws.get_git_diff()
            logger.info("[%s] Patch: %d chars", instance_id[:30], len(patch))

            test_result = ws.run_tests()
            resolved = test_result.get("passed", False)
            logger.info(
                "[%s] %s (%d/%d tests)",
                instance_id[:30],
                "RESOLVED" if resolved else "FAILED",
                test_result.get("passed_count", 0),
                test_result.get("total", 0),
            )

        finally:
            ws.cleanup()

        return (
            resp,
            tc_lats,
            {
                "instance_id": instance_id,
                "model_name_or_path": self.config.model_id,
                "model_patch": patch,
                "repo": repo,
                "resolved": resolved,
                "test_result": test_result,
                "tool_calls": resp.tool_call_count,
                "total_tokens": resp.total_tokens,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "latency_ms": resp.latency_ms,
                "ttft_ms": resp.ttft_ms,
            },
        )

    def _agentic_loop(
        self,
        messages: List[Dict[str, Any]],
        workspace_dir: str,
    ) -> Tuple[StreamingChatResponse, List[float]]:
        tools = self.config.tool_definitions or TOOL_DEFINITIONS
        executor = ToolExecutor(
            workspace_dir=workspace_dir,
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
                stop_token_ids=self.config.stop_token_ids,
            )

            cumulative_input += resp.input_tokens
            cumulative_output += resp.output_tokens
            cumulative_latency += resp.latency_ms
            print(
                f"    turn={turn}  in_tok={resp.input_tokens}  "
                f"out_tok={resp.output_tokens}  ttft={resp.ttft_ms:.1f}ms  "
                f"tpot={resp.tpot_ms_avg:.2f}ms  latency={resp.latency_ms:.1f}ms  "
                f"tool_calls={resp.tool_call_count}"
            )
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
        return combined, all_tc_latencies

    @staticmethod
    def _extract_tool_calls(
        raw_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
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
