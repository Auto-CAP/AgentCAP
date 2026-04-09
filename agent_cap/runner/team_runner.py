from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import aiohttp
from agent_cap.runner.llm_client import (
    _chat_with_fallback,
    _clean_tool_args,
    _extract_cached_tokens,
    _extract_thinking_content,
    _to_int,
)
from agent_cap.runner.tool_backends import (
    MCPToolBackend,
    SWEBenchToolBackend,
    ToolBackend,
)
from agent_cap.runner.unified_runner import (
    UnifiedTask,
    _load_dataset_tasks,
    collect_hardware_info,
    count_tool_calls,
    extract_final_assistant_text,
    flatten_tool_payload,
)

try:
    import psutil
except ImportError:
    psutil = None

from agent_cap.server.gpu_monitor import GPUMetricsSummary, GPUMonitor


THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_STRATEGY_REGISTRY: Dict[str, "DelegationStrategy"] = {}


def register_strategy(name: str, strategy: "DelegationStrategy") -> None:
    """Register a delegation strategy by name for TeamRunner resolution."""
    key = str(name).strip()
    if not key:
        raise ValueError("strategy name must be non-empty")
    _STRATEGY_REGISTRY[key] = strategy


@dataclass
class ModelEndpoint:
    """One model endpoint that can be assigned any role."""

    name: str
    base_url: str
    api_key: str = ""
    is_local: bool = False
    openrouter_provider: str = ""
    max_tokens: int = 8192
    temperature: float = 0.0
    use_streaming: bool = True


@dataclass
class TeamConfig:
    """Configuration for a multi-model team experiment."""

    strategy: str
    models: Dict[str, ModelEndpoint]
    dataset: str
    backend: str = "mcp"
    swebench_runtime: str = "docker"
    mcp_server_url: str = "http://localhost:1984"
    max_turns: int = 15
    output_root: Path = Path("results")
    enabled_tools: Sequence[str] = field(default_factory=list)


@dataclass
class RoleMetrics:
    """Token and timing metrics for one role in the team."""

    model_name: str
    role: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    prefill_time_s: float
    decode_time_s: float
    num_requests: int


@dataclass
class TeamTaskResult:
    """Result from one task run by the team."""

    task_id: str
    task_name: str
    strategy: str
    role_metrics: Dict[str, RoleMetrics]
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    tool_call_count: int
    e2e_latency_s: float
    output_text: str
    plan_text: str = ""
    errors: List[str] = field(default_factory=list)
    num_requests: int = 0
    total_prefill_time_s: float = 0.0
    total_decode_time_s: float = 0.0
    avg_input_tokens_per_request: float = 0.0
    avg_output_tokens_per_request: float = 0.0
    max_input_tokens_per_request: int = 0
    per_request_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TeamRunResult:
    output_dir: Path
    suffix: str
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    task_results: List[TeamTaskResult]


class DelegationStrategy(ABC):
    """Base class for multi-model delegation strategies."""

    @abstractmethod
    def required_roles(self) -> List[str]:
        """What roles this strategy needs."""

    @abstractmethod
    async def run_task(
        self,
        session: aiohttp.ClientSession,
        models: Dict[str, ModelEndpoint],
        task: UnifiedTask,
        tools: List[Dict[str, Any]],
        backend: ToolBackend,
        max_turns: int,
        traj_dir: Optional[Path] = None,
    ) -> TeamTaskResult:
        """Run one task under this delegation strategy."""


class PlanExecuteStrategy(DelegationStrategy):
    """Phase 1: planner generates plan. Phase 2: executor follows plan with tools."""

    PLAN_SYSTEM_PROMPT = (
        "You are an expert planning assistant. Given a task, create a clear, "
        "specific, step-by-step plan that another AI agent can follow.\n\n"
        "The executor agent has access to tools but may have limited reasoning, "
        "so your plan must be detailed and unambiguous.\n\n"
        "Output a numbered list of concrete steps. Each step should specify:\n"
        "- What tool to call (if applicable)\n"
        "- What arguments to use\n"
        "- What to do with the result\n\n"
        "Do NOT execute the task yourself. Only produce the plan."
    )

    EXEC_SYSTEM_PROMPT = (
        "You have been given a task and a step-by-step plan created by a planning agent. "
        "Follow the plan carefully and execute each step using the available tools. "
        "If a step fails, try to recover and continue with the remaining steps."
    )

    def required_roles(self) -> List[str]:
        return ["planner", "executor"]

    @staticmethod
    def _extract_message_text(message: Dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                        continue
                    block_content = block.get("content")
                    if isinstance(block_content, str) and block_content:
                        parts.append(block_content)
            return "\n".join(parts)
        return ""

    @staticmethod
    def _init_role_metrics(models: Dict[str, ModelEndpoint]) -> Dict[str, RoleMetrics]:
        role_metrics: Dict[str, RoleMetrics] = {}
        for role, endpoint in models.items():
            role_metrics[role] = RoleMetrics(
                model_name=endpoint.name,
                role=role,
                input_tokens=0,
                output_tokens=0,
                cached_tokens=0,
                prefill_time_s=0.0,
                decode_time_s=0.0,
                num_requests=0,
            )
        return role_metrics

    @staticmethod
    def _append_request_detail(
        per_request_details: List[Dict[str, Any]],
        role: str,
        request_index: int,
        in_tok: int,
        out_tok: int,
        cached_tok: int,
        ttft_s: float,
        decode_s: float,
        num_tool_calls: int,
        has_tool_calls: bool,
    ) -> None:
        tpot = decode_s / out_tok if out_tok > 0 else 0.0
        throughput = out_tok / decode_s if decode_s > 0 else 0.0
        per_request_details.append(
            {
                "request_index": request_index,
                "role": role,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cached_tokens": cached_tok,
                "prefill_time_s": round(ttft_s, 6),
                "decode_time_s": round(decode_s, 6),
                "tpot_s": round(tpot, 6),
                "output_throughput_tok_s": round(throughput, 2),
                "has_tool_calls": has_tool_calls,
                "num_tool_calls": num_tool_calls,
            }
        )

    async def run_task(
        self,
        session: aiohttp.ClientSession,
        models: Dict[str, ModelEndpoint],
        task: UnifiedTask,
        tools: List[Dict[str, Any]],
        backend: ToolBackend,
        max_turns: int,
        traj_dir: Optional[Path] = None,
    ) -> TeamTaskResult:
        planner = models["planner"]
        executor = models["executor"]
        role_metrics = self._init_role_metrics(models)
        per_request_details: List[Dict[str, Any]] = []
        errors: List[str] = []
        all_input_tokens: List[int] = []
        request_index = 0
        start = time.perf_counter()

        task_dir: Optional[Path] = None
        if traj_dir is not None:
            task_dir = traj_dir
            task_dir.mkdir(parents=True, exist_ok=True)

        user_prompt = ""
        if task.messages:
            user_prompt = str(task.messages[-1].get("content", ""))

        plan_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if task_dir is not None:
            plan_request_data = {
                "turn": "plan",
                "timestamp": datetime.now().isoformat(),
                "model": planner.name,
                "messages": plan_messages,
                "tools_count": 0,
                "temperature": planner.temperature,
                "max_tokens": planner.max_tokens,
            }
            (task_dir / "plan_request.json").write_text(
                json.dumps(plan_request_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        try:
            plan_timed = await _chat_with_fallback(
                session=session,
                base_url=planner.base_url,
                api_key=planner.api_key,
                model=planner.name,
                messages=plan_messages,
                tools=None,
                max_tokens=planner.max_tokens,
                temperature=planner.temperature,
                openrouter_provider=planner.openrouter_provider,
                use_streaming=planner.use_streaming,
                errors=errors,
            )
        except Exception as exc:
            errors.append(f"planner_request_failed: {exc}")
            return TeamTaskResult(
                task_id=task.task_id,
                task_name=task.task_name,
                strategy="plan-execute",
                role_metrics=role_metrics,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cached_tokens=0,
                tool_call_count=0,
                e2e_latency_s=time.perf_counter() - start,
                output_text="",
                plan_text="",
                errors=errors,
                num_requests=0,
                total_prefill_time_s=0.0,
                total_decode_time_s=0.0,
                avg_input_tokens_per_request=0.0,
                avg_output_tokens_per_request=0.0,
                max_input_tokens_per_request=0,
                per_request_details=per_request_details,
            )

        plan_result = plan_timed.response_json
        plan_usage = plan_result.get("usage") or {}
        plan_in_tok = _to_int(plan_usage.get("prompt_tokens", plan_timed.input_tokens))
        plan_out_tok = _to_int(
            plan_usage.get("completion_tokens", plan_timed.output_tokens)
        )
        plan_cached_tok = _to_int(
            plan_usage.get("cached_tokens", plan_timed.cached_tokens)
        )
        if plan_cached_tok == 0:
            plan_cached_tok = _extract_cached_tokens(plan_usage)

        role_metrics["planner"].input_tokens += plan_in_tok
        role_metrics["planner"].output_tokens += plan_out_tok
        role_metrics["planner"].cached_tokens += plan_cached_tok
        role_metrics["planner"].prefill_time_s += plan_timed.ttft_seconds
        role_metrics["planner"].decode_time_s += plan_timed.decode_seconds
        role_metrics["planner"].num_requests += 1
        all_input_tokens.append(plan_in_tok)

        plan_choices = plan_result.get("choices") or []
        plan_message = plan_choices[0].get("message", {}) if plan_choices else {}
        raw_plan_text = self._extract_message_text(plan_message)
        plan_text = THINK_RE.sub("", raw_plan_text).strip()

        self._append_request_detail(
            per_request_details=per_request_details,
            role="planner",
            request_index=request_index,
            in_tok=plan_in_tok,
            out_tok=plan_out_tok,
            cached_tok=plan_cached_tok,
            ttft_s=plan_timed.ttft_seconds,
            decode_s=plan_timed.decode_seconds,
            num_tool_calls=0,
            has_tool_calls=False,
        )
        request_index += 1

        if task_dir is not None:
            plan_response_data = {
                "turn": "plan",
                "timestamp": datetime.now().isoformat(),
                "raw_response": plan_result,
                "thinking_content": _extract_thinking_content(plan_message),
                "content": raw_plan_text,
                "tool_calls": plan_message.get("tool_calls") or [],
                "usage": plan_usage,
                "cached_tokens": plan_cached_tok,
                "ttft_s": plan_timed.ttft_seconds if plan_timed.is_streaming else 0.0,
                "decode_s": (
                    plan_timed.decode_seconds if plan_timed.is_streaming else 0.0
                ),
            }
            (task_dir / "plan_response.json").write_text(
                json.dumps(plan_response_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (task_dir / "plan.txt").write_text(raw_plan_text, encoding="utf-8")

        exec_prompt = (
            f"TASK: {user_prompt}\n\nPLAN:\n{plan_text}\n\nExecute the plan now."
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.EXEC_SYSTEM_PROMPT},
            {"role": "user", "content": exec_prompt},
        ]

        tool_schemas: Dict[str, Dict[str, Any]] = {}
        for tool in tools:
            fn = tool.get("function", {})
            if fn.get("name"):
                tool_schemas[str(fn["name"])] = fn.get("parameters", {})

        executed_turns = 0
        for turn_index in range(max_turns):
            executed_turns = turn_index + 1
            if task_dir is not None:
                request_data = {
                    "turn": turn_index,
                    "timestamp": datetime.now().isoformat(),
                    "model": executor.name,
                    "messages": messages,
                    "tools_count": len(tools) if tools else 0,
                    "temperature": executor.temperature,
                    "max_tokens": executor.max_tokens,
                }
                (task_dir / f"turn_{turn_index:03d}_request.json").write_text(
                    json.dumps(request_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            try:
                timed = await _chat_with_fallback(
                    session=session,
                    base_url=executor.base_url,
                    api_key=executor.api_key,
                    model=executor.name,
                    messages=messages,
                    tools=tools,
                    max_tokens=executor.max_tokens,
                    temperature=executor.temperature,
                    openrouter_provider=executor.openrouter_provider,
                    use_streaming=executor.use_streaming,
                    errors=errors,
                )
            except Exception as exc:
                errors.append(f"executor_request_failed: {exc}")
                break

            result = timed.response_json
            usage = result.get("usage") or {}
            in_tok = _to_int(usage.get("prompt_tokens", timed.input_tokens))
            out_tok = _to_int(usage.get("completion_tokens", timed.output_tokens))
            cached_tok = _to_int(usage.get("cached_tokens", timed.cached_tokens))
            if cached_tok == 0:
                cached_tok = _extract_cached_tokens(usage)

            role_metrics["executor"].input_tokens += in_tok
            role_metrics["executor"].output_tokens += out_tok
            role_metrics["executor"].cached_tokens += cached_tok
            role_metrics["executor"].prefill_time_s += timed.ttft_seconds
            role_metrics["executor"].decode_time_s += timed.decode_seconds
            role_metrics["executor"].num_requests += 1
            all_input_tokens.append(in_tok)

            choices = result.get("choices") or []
            if not choices:
                errors.append("executor returned empty choices")
                break
            assistant = choices[0].get("message") or {}
            if "role" not in assistant:
                assistant["role"] = "assistant"
            tool_calls = assistant.get("tool_calls") or []

            if task_dir is not None:
                response_data = {
                    "turn": turn_index,
                    "timestamp": datetime.now().isoformat(),
                    "raw_response": result,
                    "thinking_content": _extract_thinking_content(assistant),
                    "content": assistant.get("content", ""),
                    "tool_calls": tool_calls,
                    "usage": usage,
                    "cached_tokens": cached_tok,
                    "ttft_s": timed.ttft_seconds if timed.is_streaming else 0.0,
                    "decode_s": timed.decode_seconds if timed.is_streaming else 0.0,
                }
                (task_dir / f"turn_{turn_index:03d}_response.json").write_text(
                    json.dumps(response_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            self._append_request_detail(
                per_request_details=per_request_details,
                role="executor",
                request_index=request_index,
                in_tok=in_tok,
                out_tok=out_tok,
                cached_tok=cached_tok,
                ttft_s=timed.ttft_seconds,
                decode_s=timed.decode_seconds,
                num_tool_calls=len(tool_calls),
                has_tool_calls=len(tool_calls) > 0,
            )
            request_index += 1

            messages.append(assistant)
            if not tool_calls:
                break

            tool_calls_log: List[Dict[str, Any]] = []
            tool_results_log: List[Dict[str, Any]] = []
            for tc in tool_calls:
                function = tc.get("function") or {}
                name = str(function.get("name", ""))
                raw_args = function.get("arguments", "{}")
                try:
                    parsed_args = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except json.JSONDecodeError:
                    parsed_args = {}
                cleaned_args = parsed_args
                if isinstance(parsed_args, dict):
                    cleaned_args = _clean_tool_args(name, parsed_args, tool_schemas)

                tool_calls_log.append(
                    {
                        "tool_call_id": tc.get("id", ""),
                        "name": name,
                        "arguments_raw": raw_args,
                        "arguments_parsed": parsed_args,
                        "arguments_cleaned": cleaned_args,
                    }
                )

                tool_start = time.perf_counter()
                is_error = False
                error_msg = None
                try:
                    tool_result = await backend.call_tool(name, cleaned_args)
                except Exception as exc:
                    is_error = True
                    error_msg = str(exc)
                    errors.append(f"{name}: {exc}")
                    tool_result = [{"type": "text", "text": f"ERROR: {exc}"}]

                tool_latency = time.perf_counter() - tool_start
                flattened_result = flatten_tool_payload(tool_result)
                tool_results_log.append(
                    {
                        "tool_call_id": tc.get("id", ""),
                        "name": name,
                        "success": not is_error,
                        "result": flattened_result,
                        "error": error_msg,
                        "latency_s": round(tool_latency, 4),
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": flattened_result,
                    }
                )

            if task_dir is not None:
                (task_dir / f"turn_{turn_index:03d}_tool_calls.json").write_text(
                    json.dumps(tool_calls_log, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (task_dir / f"turn_{turn_index:03d}_tool_results.json").write_text(
                    json.dumps(tool_results_log, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        elapsed = time.perf_counter() - start
        output_text = extract_final_assistant_text(messages)
        tool_call_count = count_tool_calls(messages)

        total_input_tokens = sum(v.input_tokens for v in role_metrics.values())
        total_output_tokens = sum(v.output_tokens for v in role_metrics.values())
        total_cached_tokens = sum(v.cached_tokens for v in role_metrics.values())
        total_prefill_time_s = sum(v.prefill_time_s for v in role_metrics.values())
        total_decode_time_s = sum(v.decode_time_s for v in role_metrics.values())
        total_requests = sum(v.num_requests for v in role_metrics.values())

        if task_dir is not None:
            summary = {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "strategy": "plan-execute",
                "roles": {
                    role: {
                        "model": metric.model_name,
                        "input_tokens": metric.input_tokens,
                        "output_tokens": metric.output_tokens,
                        "cached_tokens": metric.cached_tokens,
                        "prefill_time_s": metric.prefill_time_s,
                        "decode_time_s": metric.decode_time_s,
                        "num_requests": metric.num_requests,
                    }
                    for role, metric in role_metrics.items()
                },
                "plan_text": plan_text,
                "total_turns": executed_turns,
                "total_tool_calls": tool_call_count,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cached_tokens": total_cached_tokens,
                "e2e_latency_s": elapsed,
                "errors": errors,
                "final_output": output_text,
                "tools": tools,
            }
            (task_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return TeamTaskResult(
            task_id=task.task_id,
            task_name=task.task_name,
            strategy="plan-execute",
            role_metrics=role_metrics,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cached_tokens=total_cached_tokens,
            tool_call_count=tool_call_count,
            e2e_latency_s=elapsed,
            output_text=output_text,
            plan_text=plan_text,
            errors=errors,
            num_requests=total_requests,
            total_prefill_time_s=total_prefill_time_s,
            total_decode_time_s=total_decode_time_s,
            avg_input_tokens_per_request=(
                (total_input_tokens / total_requests) if total_requests > 0 else 0.0
            ),
            avg_output_tokens_per_request=(
                (total_output_tokens / total_requests) if total_requests > 0 else 0.0
            ),
            max_input_tokens_per_request=max(all_input_tokens)
            if all_input_tokens
            else 0,
            per_request_details=per_request_details,
        )


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    frac = rank - low
    return sorted_vals[low] * (1.0 - frac) + sorted_vals[high] * frac


def _read_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def compute_team_metrics(
    results: Sequence[TeamTaskResult],
    gpu_stats: GPUMetricsSummary,
    wall_time: float,
    hw_info: Dict[str, Any],
    detailed_results_path: Optional[Path] = None,
    cpu_samples: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    request_rows = _read_jsonl(detailed_results_path)

    e2e_latencies = [r.e2e_latency_s for r in results]
    prefill_times = [float(r.get("prefill_time_s", 0.0)) for r in request_rows]
    decode_times = [float(r.get("decode_time_s", 0.0)) for r in request_rows]
    tpot_vals = [
        float(r.get("tpot_s", 0.0)) for r in request_rows if r.get("tpot_s") is not None
    ]
    throughput_vals = [
        float(r.get("output_throughput_tok_s", 0.0))
        for r in request_rows
        if r.get("output_throughput_tok_s") is not None
    ]

    total_examples = len(results)
    total_requests = sum(r.num_requests for r in results)
    total_input_tokens = sum(r.total_input_tokens for r in results)
    total_output_tokens = sum(r.total_output_tokens for r in results)
    total_cached_tokens = sum(r.total_cached_tokens for r in results)
    total_tool_calls = sum(r.tool_call_count for r in results)
    error_examples = sum(1 for r in results if r.errors)
    completed_examples = total_examples - error_examples

    role_totals: Dict[str, Dict[str, Any]] = {}
    for result in results:
        for role, metrics in result.role_metrics.items():
            if role not in role_totals:
                role_totals[role] = {
                    "model_name": metrics.model_name,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cached_tokens": 0,
                    "total_prefill_time_s": 0.0,
                    "total_decode_time_s": 0.0,
                    "total_requests": 0,
                }
            role_totals[role]["total_input_tokens"] += metrics.input_tokens
            role_totals[role]["total_output_tokens"] += metrics.output_tokens
            role_totals[role]["total_cached_tokens"] += metrics.cached_tokens
            role_totals[role]["total_prefill_time_s"] += metrics.prefill_time_s
            role_totals[role]["total_decode_time_s"] += metrics.decode_time_s
            role_totals[role]["total_requests"] += metrics.num_requests

    for role in role_totals:
        reqs = int(role_totals[role]["total_requests"] or 0)
        in_toks = int(role_totals[role]["total_input_tokens"] or 0)
        out_toks = int(role_totals[role]["total_output_tokens"] or 0)
        cached = int(role_totals[role]["total_cached_tokens"] or 0)
        role_totals[role]["avg_input_tokens_per_request"] = (
            in_toks / reqs if reqs > 0 else 0.0
        )
        role_totals[role]["avg_output_tokens_per_request"] = (
            out_toks / reqs if reqs > 0 else 0.0
        )
        role_totals[role]["cache_hit_rate"] = (cached / in_toks) if in_toks > 0 else 0.0

    cpu_vals = [float(v) for v in (cpu_samples or [])]

    return {
        "performance": {
            "e2e_s": wall_time,
            "avg_e2e_latency_s": _safe_mean(e2e_latencies),
            "p50_e2e_latency_s": _percentile(e2e_latencies, 50),
            "p99_e2e_latency_s": _percentile(e2e_latencies, 99),
            "examples_per_second": (
                (total_examples / wall_time)
                if wall_time > 0 and total_examples > 0
                else 0.0
            ),
            "ttft": _safe_mean(prefill_times),
            "p99_ttft": _percentile(prefill_times, 99),
            "tpot": _safe_mean(tpot_vals),
            "p99_tpot": _percentile(tpot_vals, 99),
            "decode_time_s": _safe_mean(decode_times),
            "p99_decode_time_s": _percentile(decode_times, 99),
            "output_throughput_tok_s": _safe_mean(throughput_vals),
        },
        "agentic": {
            "avg_total_input_tokens": _safe_mean(
                [float(r.total_input_tokens) for r in results]
            ),
            "avg_total_output_tokens": _safe_mean(
                [float(r.total_output_tokens) for r in results]
            ),
            "avg_tool_call_count": _safe_mean(
                [float(r.tool_call_count) for r in results]
            ),
            "avg_num_requests": _safe_mean([float(r.num_requests) for r in results]),
            "avg_input_tokens_per_request": _safe_mean(
                [float(r.avg_input_tokens_per_request) for r in results]
            ),
            "avg_output_tokens_per_request": _safe_mean(
                [float(r.avg_output_tokens_per_request) for r in results]
            ),
            "avg_max_input_tokens_per_request": _safe_mean(
                [float(r.max_input_tokens_per_request) for r in results]
            ),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_tokens": total_cached_tokens,
            "avg_cache_hit_rate": _safe_mean(
                [
                    (r.total_cached_tokens / r.total_input_tokens)
                    for r in results
                    if r.total_input_tokens > 0
                ]
            ),
            "total_requests": total_requests,
            "total_tool_calls": total_tool_calls,
            "per_role": role_totals,
        },
        "quality": {
            "total_examples": total_examples,
            "completed": completed_examples,
            "errors": error_examples,
        },
        "hardware": {
            "gpu_type": hw_info.get("gpu_type", "unknown"),
            "num_gpus": int(hw_info.get("num_gpus", 0) or 0),
            "avg_gpu_utilization_pct": float(gpu_stats.avg_gpu_util_pct),
            "peak_gpu_memory_used_mb": float(gpu_stats.peak_memory_used_mb),
            "avg_cpu_utilization_pct": _safe_mean(cpu_vals),
        },
    }


class TeamRunner:
    """Runs team experiments across tasks."""

    def __init__(self, config: TeamConfig):
        self.config = config
        self._strategy = self._resolve_strategy(config.strategy)

    @staticmethod
    def _resolve_strategy(name: str) -> DelegationStrategy:
        if name not in _STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy: {name}. Available: {list(_STRATEGY_REGISTRY.keys())}"
            )
        return _STRATEGY_REGISTRY[name]

    @staticmethod
    def _normalize_tasks(
        tasks: Sequence[Union[UnifiedTask, Dict[str, Any]]],
    ) -> List[UnifiedTask]:
        return [
            task if isinstance(task, UnifiedTask) else UnifiedTask.from_dict(task)
            for task in tasks
        ]

    def _validate_roles(self) -> None:
        required = self._strategy.required_roles()
        missing = [role for role in required if role not in self.config.models]
        if missing:
            raise ValueError(
                f"Missing model endpoints for required roles: {missing}. "
                f"Configured roles: {list(self.config.models.keys())}"
            )

    def _metadata(self, tasks: Sequence[UnifiedTask], timestamp: str) -> Dict[str, Any]:
        hw = collect_hardware_info()
        return {
            "hardware": hw,
            "model_config": {
                "strategy": self.config.strategy,
                "roles": {
                    role: {
                        "name": endpoint.name,
                        "base_url": endpoint.base_url,
                        "is_local": endpoint.is_local,
                        "openrouter_provider": endpoint.openrouter_provider,
                        "max_tokens": endpoint.max_tokens,
                        "temperature": endpoint.temperature,
                        "use_streaming": endpoint.use_streaming,
                    }
                    for role, endpoint in sorted(self.config.models.items())
                },
            },
            "system_environment": {
                "backend": self.config.backend,
                "swebench_runtime": self.config.swebench_runtime,
                "mcp_server_url": self.config.mcp_server_url,
                "dataset": self.config.dataset,
                "num_examples": len(tasks),
                "max_turns": self.config.max_turns,
                "timestamp": timestamp,
            },
        }

    async def run(self, tasks: Sequence[UnifiedTask]) -> TeamRunResult:
        self._validate_roles()
        normalized_tasks = self._normalize_tasks(tasks)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        team_name = f"team_{self.config.strategy.replace('-', '_')}"
        out_dir = Path(self.config.output_root) / team_name
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{self.config.dataset}_{timestamp}"
        traj_dir = out_dir / f"trajectories_{suffix}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = out_dir / f"metadata_{suffix}.json"
        metrics_path = out_dir / f"metrics_{suffix}.json"
        detailed_results_path = out_dir / f"detailed_results_{suffix}.jsonl"
        output_data_path = out_dir / f"output_data_{suffix}.jsonl"

        metadata = self._metadata(normalized_tasks, timestamp)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

        backend_name = (self.config.backend or "mcp").lower().replace("_", "-")
        runtime_name = (
            (self.config.swebench_runtime or "docker").lower().replace("_", "-")
        )

        gpu_monitor = GPUMonitor(interval=1.0)
        gpu_monitor.start()

        all_results: List[TeamTaskResult] = []
        cpu_samples: List[float] = []
        wall_start = time.perf_counter()
        wall_end = wall_start

        try:
            async with aiohttp.ClientSession() as session:
                if backend_name == "mcp":
                    backend: ToolBackend = MCPToolBackend(
                        session=session,
                        mcp_server_url=self.config.mcp_server_url,
                        enabled_tools=self.config.enabled_tools,
                    )
                elif backend_name in ("swebench", "swe-bench"):
                    if runtime_name not in ("docker", "modal"):
                        raise ValueError(
                            "Unknown swebench runtime: "
                            f"{self.config.swebench_runtime}. Supported: docker, modal"
                        )
                    backend = SWEBenchToolBackend(runtime=runtime_name)
                elif backend_name in ("swebench-docker", "swe-bench-docker"):
                    backend = SWEBenchToolBackend(runtime="docker")
                elif backend_name in ("swebench-modal", "swe-bench-modal"):
                    backend = SWEBenchToolBackend(runtime="modal")
                else:
                    raise ValueError(
                        "Unknown backend: "
                        f"{self.config.backend}. Supported: mcp, swebench-docker, swebench-modal"
                    )

                if backend_name == "mcp":
                    setup_ok = await backend.setup({})
                    if not setup_ok:
                        raise RuntimeError("MCP backend setup failed")
                    shared_tools = await backend.list_tools()
                else:
                    shared_tools = []

                with (
                    detailed_results_path.open("w", encoding="utf-8") as req_f,
                    output_data_path.open("w", encoding="utf-8") as out_f,
                ):
                    wall_start = time.perf_counter()
                    for i, task in enumerate(normalized_tasks):
                        task_dir = traj_dir / f"task_{i:03d}"
                        task_dir.mkdir(parents=True, exist_ok=True)

                        if backend_name != "mcp":
                            task_config = task.eval_config or {}
                            task_setup_ok = await backend.setup(task_config)
                            if not task_setup_ok:
                                result = TeamTaskResult(
                                    task_id=task.task_id,
                                    task_name=task.task_name,
                                    strategy=self.config.strategy,
                                    role_metrics={
                                        role: RoleMetrics(
                                            model_name=endpoint.name,
                                            role=role,
                                            input_tokens=0,
                                            output_tokens=0,
                                            cached_tokens=0,
                                            prefill_time_s=0.0,
                                            decode_time_s=0.0,
                                            num_requests=0,
                                        )
                                        for role, endpoint in self.config.models.items()
                                    },
                                    total_input_tokens=0,
                                    total_output_tokens=0,
                                    total_cached_tokens=0,
                                    tool_call_count=0,
                                    e2e_latency_s=0.0,
                                    output_text="",
                                    plan_text="",
                                    errors=["backend_setup_failed"],
                                )
                                output_data = {
                                    "index": i,
                                    "task_id": result.task_id,
                                    "input_tokens": result.total_input_tokens,
                                    "output_tokens": result.total_output_tokens,
                                    "tool_call_count": result.tool_call_count,
                                    "num_requests": result.num_requests,
                                    "e2e_latency_s": result.e2e_latency_s,
                                    "output_text": result.output_text,
                                    "errors": result.errors,
                                }
                                out_f.write(
                                    json.dumps(output_data, ensure_ascii=False) + "\n"
                                )
                                out_f.flush()
                                all_results.append(result)
                                if psutil is not None:
                                    try:
                                        cpu_samples.append(
                                            float(psutil.cpu_percent(interval=None))
                                        )
                                    except Exception:
                                        pass
                                await backend.teardown()
                                continue
                            shared_tools = await backend.list_tools()

                        try:
                            result = await self._strategy.run_task(
                                session=session,
                                models=self.config.models,
                                task=task,
                                tools=shared_tools,
                                backend=backend,
                                max_turns=self.config.max_turns,
                                traj_dir=task_dir,
                            )
                        except Exception as exc:
                            result = TeamTaskResult(
                                task_id=task.task_id,
                                task_name=task.task_name,
                                strategy=self.config.strategy,
                                role_metrics={
                                    role: RoleMetrics(
                                        model_name=endpoint.name,
                                        role=role,
                                        input_tokens=0,
                                        output_tokens=0,
                                        cached_tokens=0,
                                        prefill_time_s=0.0,
                                        decode_time_s=0.0,
                                        num_requests=0,
                                    )
                                    for role, endpoint in self.config.models.items()
                                },
                                total_input_tokens=0,
                                total_output_tokens=0,
                                total_cached_tokens=0,
                                tool_call_count=0,
                                e2e_latency_s=0.0,
                                output_text="",
                                plan_text="",
                                errors=[f"strategy_failed: {exc}"],
                            )
                        finally:
                            if backend_name != "mcp":
                                try:
                                    patch = await backend.get_patch()
                                    if patch:
                                        (task_dir / "patch.diff").write_text(
                                            patch, encoding="utf-8"
                                        )
                                except Exception:
                                    pass
                                await backend.teardown()

                        result.task_id = task.task_id
                        result.task_name = task.task_name

                        for req_row in result.per_request_details:
                            req_row["example_index"] = i
                            req_row["task_id"] = task.task_id
                            req_f.write(json.dumps(req_row, ensure_ascii=False) + "\n")
                        req_f.flush()

                        output_data = {
                            "index": i,
                            "task_id": result.task_id,
                            "input_tokens": result.total_input_tokens,
                            "output_tokens": result.total_output_tokens,
                            "tool_call_count": result.tool_call_count,
                            "num_requests": result.num_requests,
                            "e2e_latency_s": result.e2e_latency_s,
                            "output_text": result.output_text,
                            "errors": result.errors,
                        }
                        out_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                        out_f.flush()
                        all_results.append(result)

                        if psutil is not None:
                            try:
                                cpu_samples.append(
                                    float(psutil.cpu_percent(interval=None))
                                )
                            except Exception:
                                pass
                    wall_end = time.perf_counter()

                if backend_name == "mcp":
                    await backend.teardown()
        finally:
            gpu_stats = gpu_monitor.stop()

        metrics = compute_team_metrics(
            results=all_results,
            gpu_stats=gpu_stats,
            wall_time=max(0.0, wall_end - wall_start),
            hw_info=metadata.get("hardware", {}),
            detailed_results_path=detailed_results_path,
            cpu_samples=cpu_samples,
        )
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

        return TeamRunResult(
            output_dir=out_dir,
            suffix=suffix,
            metadata=metadata,
            metrics=metrics,
            task_results=all_results,
        )


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    return value


def _to_model_endpoint(raw: Dict[str, Any]) -> ModelEndpoint:
    return ModelEndpoint(
        name=str(raw.get("name", "")),
        base_url=str(raw.get("base_url", "http://localhost:30000")),
        api_key=str(raw.get("api_key", "dummy")),
        is_local=bool(raw.get("is_local", False)),
        openrouter_provider=str(raw.get("openrouter_provider", "")),
        max_tokens=int(raw.get("max_tokens", 8192) or 8192),
        temperature=float(raw.get("temperature", 0.0) or 0.0),
        use_streaming=bool(raw.get("use_streaming", True)),
    )


def _merge_endpoint(
    base: Optional[ModelEndpoint],
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    no_streaming: bool,
) -> ModelEndpoint:
    merged = base or ModelEndpoint(
        name="", base_url="http://localhost:30000", api_key="dummy"
    )
    if model is not None:
        merged.name = model
    if base_url is not None:
        merged.base_url = base_url
    if api_key is not None:
        merged.api_key = api_key
    if no_streaming:
        merged.use_streaming = False
    return merged


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AgentCAP Multi-Model Team Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CLI planner/executor endpoints:
  python -m agent_cap.runner.team_runner \\
    --strategy plan-execute \\
    --dataset mcp-atlas \\
    --planner-model claude-4.6-opus \\
    --planner-base-url https://api.anthropic.com/v1 \\
    --planner-api-key $ANTHROPIC_API_KEY \\
    --executor-model deepseek-chat \\
    --executor-base-url https://api.deepseek.com/v1 \\
    --executor-api-key $DEEPSEEK_API_KEY

  # YAML config (CLI overrides):
  python -m agent_cap.runner.team_runner --config configs/team_experiment.yaml
""",
    )
    parser.add_argument("--config", type=str, help="YAML team config path")

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Delegation strategy (default: plan-execute)",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Tool backend: mcp, swebench-docker, swebench-modal",
    )
    parser.add_argument(
        "--swebench-runtime",
        type=str,
        default=None,
        help="SWE-bench runtime: docker or modal",
    )
    parser.add_argument("--mcp-server-url", type=str, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--enabled-tools", nargs="*", default=None)
    parser.add_argument("--no-streaming", action="store_true")

    parser.add_argument("--planner-model", type=str, default=None)
    parser.add_argument("--planner-base-url", type=str, default=None)
    parser.add_argument("--planner-api-key", type=str, default=None)

    parser.add_argument("--executor-model", type=str, default=None)
    parser.add_argument("--executor-base-url", type=str, default=None)
    parser.add_argument("--executor-api-key", type=str, default=None)

    args = parser.parse_args()

    file_cfg: Dict[str, Any] = {}
    if args.config:
        try:
            import yaml
        except Exception as exc:
            parser.error(f"failed to import yaml for --config: {exc}")
        with open(args.config, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                parser.error("--config YAML must deserialize to a dictionary")
            file_cfg = _expand_env_vars(loaded)

    def resolve(cli_val: Any, yaml_key: str, default: Any = None) -> Any:
        if cli_val is not None:
            return cli_val
        if yaml_key in file_cfg:
            return file_cfg[yaml_key]
        return default

    models_cfg_raw = file_cfg.get("models") or {}
    if not isinstance(models_cfg_raw, dict):
        parser.error("config field 'models' must be a dictionary")
    models: Dict[str, ModelEndpoint] = {}
    for role, raw_cfg in models_cfg_raw.items():
        if isinstance(raw_cfg, dict):
            models[str(role)] = _to_model_endpoint(raw_cfg)

    models["planner"] = _merge_endpoint(
        models.get("planner"),
        model=args.planner_model,
        base_url=args.planner_base_url,
        api_key=args.planner_api_key,
        no_streaming=args.no_streaming,
    )
    models["executor"] = _merge_endpoint(
        models.get("executor"),
        model=args.executor_model,
        base_url=args.executor_base_url,
        api_key=args.executor_api_key,
        no_streaming=args.no_streaming,
    )

    strategy_name = resolve(args.strategy, "strategy", "plan-execute")
    dataset_name = resolve(args.dataset, "dataset", "")
    if not dataset_name:
        parser.error("--dataset is required (or set dataset in config)")

    required_roles = TeamRunner._resolve_strategy(strategy_name).required_roles()
    for role in required_roles:
        endpoint = models.get(role)
        if endpoint is None or not endpoint.name:
            parser.error(
                f"Missing model for required role '{role}'. "
                f"Set via --{role}-model or config models.{role}.name"
            )

    config = TeamConfig(
        strategy=str(strategy_name),
        models=models,
        dataset=str(dataset_name),
        backend=str(resolve(args.backend, "backend", "mcp")),
        swebench_runtime=str(
            resolve(args.swebench_runtime, "swebench_runtime", "docker")
        ),
        mcp_server_url=str(
            resolve(args.mcp_server_url, "mcp_server_url", "http://localhost:1984")
        ),
        max_turns=int(resolve(args.max_turns, "max_turns", 15)),
        output_root=Path(
            resolve(
                args.output_dir, "output_dir", file_cfg.get("output_root", "results")
            )
        ),
        enabled_tools=resolve(args.enabled_tools, "enabled_tools", []),
    )

    num_tasks = int(resolve(args.num_tasks, "num_tasks", 0) or 0)
    tasks = _load_dataset_tasks(config.dataset, num_tasks)

    import asyncio

    runner = TeamRunner(config)
    result = asyncio.run(runner.run(tasks))
    print(f"\nDone. Output directory: {result.output_dir}")
    print(f"  metadata:         metadata_{result.suffix}.json")
    print(f"  metrics:          metrics_{result.suffix}.json")
    print(f"  detailed_results: detailed_results_{result.suffix}.jsonl")
    print(f"  output_data:      output_data_{result.suffix}.jsonl")


__all__ = [
    "ModelEndpoint",
    "TeamConfig",
    "RoleMetrics",
    "DelegationStrategy",
    "PlanExecuteStrategy",
    "TeamTaskResult",
    "TeamRunResult",
    "TeamRunner",
    "compute_team_metrics",
    "register_strategy",
    "main",
]


register_strategy("plan-execute", PlanExecuteStrategy())


if __name__ == "__main__":
    main()
