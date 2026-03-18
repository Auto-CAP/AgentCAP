import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_cap.core.agentic_loop import LoopResult, run_agentic_loop
from agent_cap.core.evaluator import Evaluator, EvalResult
from agent_cap.core.tool_backend import ToolBackend
from agent_cap.server.streaming_client import StreamingChatClient

THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

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


@dataclass
class PlanExecResult:
    plan: str
    response: str
    plan_input_tokens: int
    plan_output_tokens: int
    plan_latency_ms: float
    plan_ttft_ms: float
    exec_input_tokens: int
    exec_output_tokens: int
    exec_latency_ms: float
    exec_ttft_ms: float
    exec_tpot_ms_avg: float
    tool_calls: int
    tool_latencies_ms: List[float]
    eval: Optional[EvalResult] = None
    errors: List[str] = field(default_factory=list)


def run_plan_execute(
    planner_client: StreamingChatClient,
    executor_client: StreamingChatClient,
    backend: ToolBackend,
    evaluator: Optional[Evaluator],
    task_config: Dict[str, Any],
    prompt: str,
    planner_model: str = "default",
    executor_model: str = "default",
    max_plan_tokens: int = 4096,
    max_exec_turns: int = 20,
    max_exec_tokens: int = 16384,
    temperature: float = 0.0,
    stop_token_ids: Optional[List[int]] = None,
) -> PlanExecResult:

    # Phase 1: Plan
    plan_messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    plan_resp = planner_client.chat(
        messages=plan_messages,
        model=planner_model,
        temperature=temperature,
        max_tokens=max_plan_tokens,
    )
    plan_text = THINK_RE.sub("", plan_resp.content).strip()
    print(
        f"  [PLAN] {plan_resp.input_tokens} in / {plan_resp.output_tokens} out / {plan_resp.latency_ms:.0f}ms"
    )
    print(f"    {plan_text[:200]}...")

    # Phase 2: Execute
    exec_prompt = f"TASK: {prompt}\n\nPLAN:\n{plan_text}\n\nExecute the plan now."
    exec_messages = [
        {"role": "system", "content": EXEC_SYSTEM_PROMPT},
        {"role": "user", "content": exec_prompt},
    ]

    exec_result = run_agentic_loop(
        client=executor_client,
        backend=backend,
        messages=exec_messages,
        model=executor_model,
        max_turns=max_exec_turns,
        max_tokens=max_exec_tokens,
        temperature=temperature,
        stop_token_ids=stop_token_ids,
    )

    patch = backend.get_patch()

    eval_result = None
    if evaluator:
        eval_result = evaluator.evaluate(task_config, backend)

    return PlanExecResult(
        plan=plan_text,
        response=exec_result.response,
        plan_input_tokens=plan_resp.input_tokens,
        plan_output_tokens=plan_resp.output_tokens,
        plan_latency_ms=plan_resp.latency_ms,
        plan_ttft_ms=plan_resp.ttft_ms,
        exec_input_tokens=exec_result.input_tokens,
        exec_output_tokens=exec_result.output_tokens,
        exec_latency_ms=exec_result.latency_ms,
        exec_ttft_ms=exec_result.ttft_ms,
        exec_tpot_ms_avg=exec_result.tpot_ms_avg,
        tool_calls=exec_result.tool_calls,
        tool_latencies_ms=exec_result.tool_latencies_ms,
        eval=eval_result,
        errors=exec_result.errors,
    )
