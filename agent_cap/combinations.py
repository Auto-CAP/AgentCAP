import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agent_cap.server.client import ChatClient, ChatResponse

logger = logging.getLogger("agent_cap.combinations")


@dataclass
class StepRecord:
    step_name: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    output_text: str


@dataclass
class CombinationResult:
    strategy: str
    final_output: str
    steps: List[StepRecord] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    task_success: Optional[bool] = None
    quality_score: Optional[float] = None
    eval_explanation: str = ""


def _evaluate_output(
    output_text: str,
    eval_config: Optional[Dict],
) -> Tuple[Optional[bool], Optional[float], str]:
    if not eval_config:
        return None, None, ""

    from agent_cap.evaluator import EvalConfig, evaluate

    cfg = EvalConfig.from_dict(eval_config)
    result = evaluate(output_text, cfg)
    return result.task_success, result.quality_score, result.explanation


def _get_task_text(messages: List[Dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _record_step(step_name: str, model_id: str, response: ChatResponse) -> StepRecord:
    return StepRecord(
        step_name=step_name,
        model_id=model_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        output_text=response.content,
    )


def _totals(steps: List[StepRecord]) -> Tuple[int, int, float]:
    total_input_tokens = sum(step.input_tokens for step in steps)
    total_output_tokens = sum(step.output_tokens for step in steps)
    total_latency_ms = sum(step.latency_ms for step in steps)
    return total_input_tokens, total_output_tokens, total_latency_ms


def run_single_pass(
    messages: List[Dict],
    client: ChatClient,
    model_id: str,
    eval_config: Optional[Dict],
    max_tokens: int = 8192,
) -> CombinationResult:
    response = client.chat(messages, model=model_id, temperature=0.0, max_tokens=max_tokens)
    step = _record_step("generate", model_id, response)
    task_success, quality_score, eval_explanation = _evaluate_output(response.content, eval_config)
    total_input_tokens, total_output_tokens, total_latency_ms = _totals([step])

    logger.debug(
        "single-pass complete: %s",
        json.dumps({"model": model_id, "steps": 1, "latency_ms": total_latency_ms}),
    )

    return CombinationResult(
        strategy="single-pass",
        final_output=response.content,
        steps=[step],
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=total_latency_ms,
        task_success=task_success,
        quality_score=quality_score,
        eval_explanation=eval_explanation,
    )


def run_cascade(
    messages: List[Dict],
    small_client: ChatClient,
    small_model_id: str,
    large_client: ChatClient,
    large_model_id: str,
    eval_config: Optional[Dict],
    max_tokens: int = 8192,
) -> CombinationResult:
    steps: List[StepRecord] = []

    small_response = small_client.chat(
        messages,
        model=small_model_id,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    small_step = _record_step("small_generate", small_model_id, small_response)
    steps.append(small_step)

    small_success, small_score, small_explanation = _evaluate_output(small_response.content, eval_config)
    if eval_config and small_success is True:
        total_input_tokens, total_output_tokens, total_latency_ms = _totals(steps)
        return CombinationResult(
            strategy="cascade",
            final_output=small_response.content,
            steps=steps,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_latency_ms=total_latency_ms,
            task_success=small_success,
            quality_score=small_score,
            eval_explanation=small_explanation,
        )

    if eval_config:
        logger.debug("cascade fallback to large model: %s -> %s", small_model_id, large_model_id)
    else:
        total_input_tokens, total_output_tokens, total_latency_ms = _totals(steps)
        return CombinationResult(
            strategy="cascade",
            final_output=small_response.content,
            steps=steps,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_latency_ms=total_latency_ms,
            task_success=small_success,
            quality_score=small_score,
            eval_explanation=small_explanation,
        )

    large_response = large_client.chat(
        messages,
        model=large_model_id,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    large_step = _record_step("large_generate", large_model_id, large_response)
    steps.append(large_step)

    large_success, large_score, large_explanation = _evaluate_output(large_response.content, eval_config)
    total_input_tokens, total_output_tokens, total_latency_ms = _totals(steps)
    return CombinationResult(
        strategy="cascade",
        final_output=large_response.content,
        steps=steps,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=total_latency_ms,
        task_success=large_success,
        quality_score=large_score,
        eval_explanation=large_explanation,
    )


def run_self_critique(
    messages: List[Dict],
    client: ChatClient,
    model_id: str,
    eval_config: Optional[Dict],
    max_tokens: int = 8192,
) -> CombinationResult:
    steps: List[StepRecord] = []
    original_task_text = _get_task_text(messages)

    initial_response = client.chat(messages, model=model_id, temperature=0.0, max_tokens=max_tokens)
    steps.append(_record_step("generate", model_id, initial_response))

    critique_messages = [
        {
            "role": "user",
            "content": (
                "Review this code for correctness against the original task. "
                "List specific bugs or issues. Be concise.\n\n"
                f"Original task: {original_task_text}\n\n"
                f"Code to review:\n{initial_response.content}"
            ),
        }
    ]
    critique_response = client.chat(
        critique_messages,
        model=model_id,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    steps.append(_record_step("critique", model_id, critique_response))

    revise_messages = [
        {
            "role": "user",
            "content": (
                "Based on this code review, write an improved version of the code. "
                "Fix all identified issues.\n\n"
                f"Original task: {original_task_text}\n\n"
                f"Review feedback:\n{critique_response.content}\n\n"
                f"Original code:\n{initial_response.content}\n\n"
                "Write the improved code in a ```python block."
            ),
        }
    ]
    revise_response = client.chat(
        revise_messages,
        model=model_id,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    steps.append(_record_step("revise", model_id, revise_response))

    task_success, quality_score, eval_explanation = _evaluate_output(revise_response.content, eval_config)
    total_input_tokens, total_output_tokens, total_latency_ms = _totals(steps)

    return CombinationResult(
        strategy="self-critique",
        final_output=revise_response.content,
        steps=steps,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=total_latency_ms,
        task_success=task_success,
        quality_score=quality_score,
        eval_explanation=eval_explanation,
    )


def run_multi_model_vote(
    messages: List[Dict],
    clients_and_models: List[Tuple[ChatClient, str]],
    eval_config: Optional[Dict],
    max_tokens: int = 8192,
) -> CombinationResult:
    if not clients_and_models:
        raise ValueError("clients_and_models must not be empty")

    steps: List[StepRecord] = []
    evals: List[Tuple[Optional[bool], Optional[float], str]] = []

    for client, model_id in clients_and_models:
        response = client.chat(messages, model=model_id, temperature=0.0, max_tokens=max_tokens)
        steps.append(_record_step("generate", model_id, response))
        evals.append(_evaluate_output(response.content, eval_config))

    selected_index = 0
    if eval_config:
        best_score = max((score if score is not None else float("-inf")) for _, score, _ in evals)
        candidate_indices = [
            idx for idx, (_, score, _) in enumerate(evals) if (score if score is not None else float("-inf")) == best_score
        ]
        passed_index = next((idx for idx in candidate_indices if evals[idx][0] is True), None)
        selected_index = passed_index if passed_index is not None else candidate_indices[0]

    final_step = steps[selected_index]
    task_success, quality_score, eval_explanation = evals[selected_index]
    total_input_tokens, total_output_tokens, total_latency_ms = _totals(steps)

    return CombinationResult(
        strategy="vote",
        final_output=final_step.output_text,
        steps=steps,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=total_latency_ms,
        task_success=task_success,
        quality_score=quality_score,
        eval_explanation=eval_explanation,
    )


def run_generate_verify(
    messages: List[Dict],
    small_client: ChatClient,
    small_model_id: str,
    large_client: ChatClient,
    large_model_id: str,
    eval_config: Optional[Dict],
    max_tokens: int = 8192,
) -> CombinationResult:
    steps: List[StepRecord] = []
    original_task_text = _get_task_text(messages)

    small_response = small_client.chat(
        messages,
        model=small_model_id,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    steps.append(_record_step("small_generate", small_model_id, small_response))

    verify_messages = [
        {
            "role": "user",
            "content": (
                "You are a code reviewer. Does this code correctly implement the requirements?\n\n"
                f"Requirements: {original_task_text}\n\n"
                f"Code:\n{small_response.content}\n\n"
                "Respond with VERDICT: PASS or VERDICT: FAIL, then explain briefly."
            ),
        }
    ]
    verify_response = large_client.chat(
        verify_messages,
        model=large_model_id,
        temperature=0.0,
        max_tokens=2048,
    )
    steps.append(_record_step("verify", large_model_id, verify_response))

    verify_upper = verify_response.content.upper()
    verdict_pass = "VERDICT: PASS" in verify_upper
    verdict_fail = "VERDICT: FAIL" in verify_upper

    if verdict_pass and not verdict_fail:
        final_output = small_response.content
    else:
        large_response = large_client.chat(
            messages,
            model=large_model_id,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        steps.append(_record_step("large_generate", large_model_id, large_response))
        final_output = large_response.content

    task_success, quality_score, eval_explanation = _evaluate_output(final_output, eval_config)
    total_input_tokens, total_output_tokens, total_latency_ms = _totals(steps)

    return CombinationResult(
        strategy="generate-verify",
        final_output=final_output,
        steps=steps,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=total_latency_ms,
        task_success=task_success,
        quality_score=quality_score,
        eval_explanation=eval_explanation,
    )
