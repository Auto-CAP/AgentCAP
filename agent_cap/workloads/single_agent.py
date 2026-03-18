from typing import Any, Dict, List, Optional

from agent_cap.core.agentic_loop import LoopResult, run_agentic_loop
from agent_cap.core.evaluator import Evaluator, EvalResult
from agent_cap.core.tool_backend import ToolBackend
from agent_cap.server.streaming_client import StreamingChatClient

SYSTEM_PROMPT = (
    "You are an expert software engineer tasked with fixing bugs in code repositories. "
    "You have access to tools: read_file, write_file, run_shell, search_code.\n\n"
    "YOUR WORKFLOW:\n"
    "1. READ: Use search_code and read_file to find the relevant code (1-3 calls max)\n"
    "2. FIX: Use write_file to modify the source files that need changing\n"
    "3. VERIFY: Use run_shell to run tests and confirm your fix\n\n"
    "CRITICAL: You MUST call write_file to fix the code. Reading alone is NOT enough.\n"
    "Do NOT spend more than 3 turns just reading. Start writing fixes.\n"
)


def run_single_agent(
    client: StreamingChatClient,
    backend: ToolBackend,
    evaluator: Optional[Evaluator],
    task_config: Dict[str, Any],
    prompt: str,
    model: str = "default",
    max_turns: int = 10,
    max_tokens: int = 16384,
    temperature: float = 0.0,
    stop_token_ids: Optional[List[int]] = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    loop_result = run_agentic_loop(
        client=client,
        backend=backend,
        messages=messages,
        model=model,
        max_turns=max_turns,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_token_ids=stop_token_ids,
    )

    patch = backend.get_patch()

    eval_result = None
    if evaluator:
        eval_result = evaluator.evaluate(task_config, backend)

    return {
        "response": loop_result.response,
        "patch": patch,
        "tool_calls": loop_result.tool_calls,
        "input_tokens": loop_result.input_tokens,
        "output_tokens": loop_result.output_tokens,
        "latency_ms": loop_result.latency_ms,
        "ttft_ms": loop_result.ttft_ms,
        "tpot_ms_avg": loop_result.tpot_ms_avg,
        "tpot_ms_p99": loop_result.tpot_ms_p99,
        "eval": eval_result,
        "errors": loop_result.errors,
    }
