import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_cap.core.tool_backend import ToolBackend, ToolResult
from agent_cap.server.streaming_client import StreamingChatClient, StreamingChatResponse


@dataclass
class LoopResult:
    response: str
    messages: List[Dict[str, Any]]
    tool_calls: int
    tool_latencies_ms: List[float]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float
    tpot_ms_avg: float
    tpot_ms_p99: float
    errors: List[str] = field(default_factory=list)


def run_agentic_loop(
    client: StreamingChatClient,
    backend: ToolBackend,
    messages: List[Dict[str, Any]],
    model: str = "default",
    max_turns: int = 10,
    max_tokens: int = 16384,
    temperature: float = 0.0,
    stop_token_ids: Optional[List[int]] = None,
) -> LoopResult:
    tools = backend.get_tool_definitions()
    total_input = 0
    total_output = 0
    total_latency = 0.0
    first_ttft: Optional[float] = None
    tpot_avgs: List[float] = []
    last_tpot_p99 = 0.0
    total_tool_calls = 0
    tool_latencies: List[float] = []
    final_content = ""
    errors: List[str] = []

    for turn in range(max_turns):
        resp = client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools if tools else None,
            stop_token_ids=stop_token_ids,
        )

        total_input += resp.input_tokens
        total_output += resp.output_tokens
        total_latency += resp.latency_ms
        if first_ttft is None:
            first_ttft = resp.ttft_ms
        if resp.tpot_ms_avg > 0:
            tpot_avgs.append(resp.tpot_ms_avg)
        if resp.tpot_ms_p99 > 0:
            last_tpot_p99 = resp.tpot_ms_p99

        print(
            f"    turn={turn}  in_tok={resp.input_tokens}  "
            f"out_tok={resp.output_tokens}  ttft={resp.ttft_ms:.1f}ms  "
            f"tpot={resp.tpot_ms_avg:.2f}ms  latency={resp.latency_ms:.1f}ms  "
            f"tool_calls={resp.tool_call_count}"
        )

        if resp.tool_call_count == 0 or not resp.raw_chunks:
            final_content = resp.content
            break

        pending = _extract_tool_calls(resp.raw_chunks)
        if not pending:
            final_content = resp.content
            break

        assistant_msg: Dict[str, Any] = {"role": "assistant"}
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

            result = backend.execute(tc["name"], tc["id"], arguments=args)
            tool_latencies.append(result.latency_ms)
            total_tool_calls += 1

            out_preview = (
                result.output[:150].replace("\n", "\\n") if result.output else "(empty)"
            )
            # print(
            #     f"      -> {tc['name']}({json.dumps(args)[:80]})  "
            #     f"ok={result.success}  {result.latency_ms:.0f}ms  "
            #     f"out={out_preview}"
            # )

            ######################################################################################################
            out_preview = (
                result.output[:150].replace("\n", "\\n") if result.output else "(empty)"
            )

            log_line = (
                f"      -> {tc['name']}({json.dumps(args)[:80]})  "
                f"ok={result.success}  {result.latency_ms:.0f}ms  "
                f"out={out_preview}"
            )

            if result.output and result.output.startswith("[WARN]"):
                input_code = args.get("code", args.get("raw", ""))
                log_line += (
                    f"\n         full_input_code:\n"
                    f"{input_code}"
                )

            print(log_line)
            ######################################################################################################
            
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result.output,
                }
            )

    avg_tpot = sum(tpot_avgs) / len(tpot_avgs) if tpot_avgs else 0.0

    return LoopResult(
        response=final_content,
        messages=messages,
        tool_calls=total_tool_calls,
        tool_latencies_ms=tool_latencies,
        input_tokens=total_input,
        output_tokens=total_output,
        latency_ms=total_latency,
        ttft_ms=first_ttft or 0.0,
        tpot_ms_avg=avg_tpot,
        tpot_ms_p99=last_tpot_p99,
        errors=errors,
    )


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
