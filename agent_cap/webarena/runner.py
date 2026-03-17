"""WebArena benchmark runner.

Agentic loop: model sees page text → calls browser tools → page updates → repeat.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from agent_cap.server.streaming_client import StreamingChatClient, StreamingChatResponse
from agent_cap.server.cpu_monitor import CPUMonitor
from agent_cap.server.gpu_monitor import GPUMonitor
from agent_cap.webarena.browser_tools import TOOL_DEFINITIONS, BrowserToolExecutor
from agent_cap.webarena.task_loader import WebArenaTask, load_webarena_tasks

logger = logging.getLogger("agent_cap.webarena")


class WebArenaRunner:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_id: str = "default",
        max_turns: int = 10,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        stop_token_ids: Optional[List[int]] = None,
    ):
        self.client = StreamingChatClient(base_url=base_url)
        self.model_id = model_id
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop_token_ids = stop_token_ids

        server_model = self.client.get_server_model_id()
        if server_model and server_model != model_id:
            logger.info("Server model: %s (config had: %s)", server_model, model_id)
            self.model_id = server_model

    def run_task(self, task: WebArenaTask) -> Dict[str, Any]:
        browser = BrowserToolExecutor()
        browser.start()

        try:
            logger.info("[task=%d] %s", task.task_id, task.intent[:80])
            logger.info("[task=%d] start_url: %s", task.task_id, task.start_url)

            if task.start_url:
                browser.execute("goto", "init", {"url": task.start_url})

            page_text = browser.get_page_snapshot()

            system_msg = (
                "You are a web browsing agent. Complete the following task by "
                "interacting with the web page using the provided tools.\n\n"
                "Available tools:\n"
                "- goto(url): Navigate to a URL\n"
                "- click(selector): Click a CSS selector\n"
                "- type_text(selector, text): Type into an input field\n"
                "- scroll(direction): Scroll 'up' or 'down'\n"
                "- get_page_text(): Get visible page text\n"
                "- get_page_url(): Get current URL\n\n"
                "After completing the task, stop calling tools."
            )
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        f"Task: {task.intent}\n\n"
                        f"Current URL: {task.start_url}\n\n"
                        f"Page content:\n{page_text[:4000]}"
                    ),
                },
            ]

            cumulative_input = 0
            cumulative_output = 0
            cumulative_latency = 0.0
            first_ttft = 0.0
            total_tool_calls = 0
            tool_latencies: List[float] = []

            gpu_mon = GPUMonitor(interval=0.5)
            cpu_mon = CPUMonitor(interval=0.5)
            gpu_mon.start()
            cpu_mon.start()

            t_start = time.perf_counter()

            for turn in range(self.max_turns):
                resp = self.client.chat(
                    messages=messages,
                    model=self.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=TOOL_DEFINITIONS,
                    stop_token_ids=self.stop_token_ids,
                )

                cumulative_input += resp.input_tokens
                cumulative_output += resp.output_tokens
                cumulative_latency += resp.latency_ms
                if first_ttft == 0.0:
                    first_ttft = resp.ttft_ms

                print(
                    f"    turn={turn}  in_tok={resp.input_tokens}  "
                    f"out_tok={resp.output_tokens}  ttft={resp.ttft_ms:.1f}ms  "
                    f"latency={resp.latency_ms:.1f}ms  tool_calls={resp.tool_call_count}"
                )

                if resp.tool_call_count == 0:
                    break

                pending = self._extract_tool_calls(resp.raw_chunks)
                if not pending:
                    break

                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                if resp.content:
                    assistant_msg["content"] = resp.content
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in pending
                ]
                messages.append(assistant_msg)

                for tc in pending:
                    try:
                        args = json.loads(tc["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": tc["arguments"]}

                    result = browser.execute(tc["name"], tc["id"], args)
                    tool_latencies.append(result.latency_ms)
                    total_tool_calls += 1

                    tool_output = result.output
                    if tc["name"] in ("click", "goto", "type_text"):
                        page_text = browser.get_page_snapshot()
                        tool_output += (
                            f"\n\nPage content after action:\n{page_text[:4000]}"
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_output,
                        }
                    )

            wall_clock_s = time.perf_counter() - t_start
            gpu_stats = gpu_mon.stop()
            cpu_stats = cpu_mon.stop()

        finally:
            browser.stop()

        return {
            "task_id": task.task_id,
            "intent": task.intent,
            "total_tokens": cumulative_input + cumulative_output,
            "input_tokens": cumulative_input,
            "output_tokens": cumulative_output,
            "latency_ms": cumulative_latency,
            "wall_clock_s": wall_clock_s,
            "ttft_ms": first_ttft,
            "tool_calls": total_tool_calls,
            "avg_tool_latency_ms": (
                sum(tool_latencies) / len(tool_latencies) if tool_latencies else 0
            ),
            "gpu_util_pct": gpu_stats.avg_gpu_util_pct,
            "cpu_util_pct": cpu_stats.avg_cpu_util_pct,
        }

    def run_tasks(self, tasks: List[WebArenaTask]) -> List[Dict[str, Any]]:
        results = []
        for i, task in enumerate(tasks):
            logger.info("Running task %d/%d", i + 1, len(tasks))
            try:
                result = self.run_task(task)
                results.append(result)
                print(
                    f"  task={task.task_id}  tools={result['tool_calls']}  "
                    f"E2E={result['wall_clock_s']:.1f}s  "
                    f"tokens={result['total_tokens']}  "
                    f"GPU={result['gpu_util_pct']:.1f}%"
                )
            except Exception as exc:
                logger.error("Task %d failed: %s", task.task_id, exc)
                results.append({"task_id": task.task_id, "error": str(exc)})
        return results

    @staticmethod
    def _extract_tool_calls(raw_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
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
