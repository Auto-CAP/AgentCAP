from __future__ import annotations

import argparse
import ast
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import aiohttp

from agent_cap.evaluators.gtfa_eval import GTFAEvaluator
from agent_cap.runner.llm_client import _chat_with_fallback
from agent_cap.runner.tool_backends import MCPToolBackend
from agent_cap.runner.unified_runner import _load_dataset_tasks


def _parse_claims(raw: object) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            try:
                raw = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                raw = [raw] if raw.strip() else []
    if not isinstance(raw, list):
        raw = [str(raw)] if str(raw).strip() else []
    return [str(c).strip() for c in raw if str(c).strip()]


def _load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and row.get("task_id"):
                done[str(row["task_id"])] = row
    return done


async def _run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_data_path = out_dir / f"output-data_{suffix}.jsonl"
    results_path = out_dir / "results.jsonl"
    tasks = _load_dataset_tasks(args.dataset, int(args.num_tasks or 0))
    done = _load_done(results_path)
    evaluator = GTFAEvaluator()

    async with aiohttp.ClientSession() as session:
        backend = MCPToolBackend(session=session, mcp_server_url=args.mcp_server_url)
        if not await backend.setup({}):
            raise RuntimeError("MCP backend setup failed")
        all_tools = await backend.list_tools()
        with (
            output_data_path.open("w", encoding="utf-8") as out_f,
            results_path.open("a", encoding="utf-8") as res_f,
        ):
            for i, task in enumerate(tasks):
                row = done.get(task.task_id)
                if row is None:
                    tools = all_tools
                    if task.enabled_tools is not None:
                        allow = {str(t) for t in task.enabled_tools if str(t).strip()}
                        tools = [
                            t
                            for t in all_tools
                            if t.get("function", {}).get("name", "") in allow
                        ]
                    messages = [dict(m) for m in task.messages]
                    total_in = total_out = tool_calls = reqs = 0
                    output_text, score, passed = "", 0.0, False
                    details = {"evaluator": "gtfa"}
                    errors: list[str] = []
                    start = time.perf_counter()
                    for turn in range(args.max_turns):
                        reqs = turn + 1
                        timed = None
                        for retry in range(3):
                            try:
                                timed = await _chat_with_fallback(
                                    session=session,
                                    base_url=args.base_url,
                                    api_key=args.api_key,
                                    model=args.model,
                                    messages=messages,
                                    tools=tools,
                                    max_tokens=8192,
                                    temperature=0.0,
                                    openrouter_provider="",
                                    use_streaming=True,
                                    errors=errors,
                                    parallel_tool_calls=True,
                                )
                                break
                            except Exception as exc:
                                errors.append(f"llm_retry_{retry + 1}_failed: {exc}")
                                if retry < 2:
                                    await asyncio.sleep(2**retry)
                        if timed is None:
                            errors.append("llm_request_failed")
                            break
                        usage = timed.response_json.get("usage") or {}
                        total_in += int(
                            usage.get("prompt_tokens", timed.input_tokens) or 0
                        )
                        total_out += int(
                            usage.get("completion_tokens", timed.output_tokens) or 0
                        )
                        msg = (timed.response_json.get("choices") or [{}])[0].get(
                            "message"
                        ) or {}
                        calls = msg.get("tool_calls") or []
                        tool_calls += len(calls)
                        messages.append(
                            {
                                "role": "assistant",
                                "content": msg.get("content"),
                                "tool_calls": calls,
                            }
                        )
                        if not calls:
                            output_text = str(msg.get("content", "") or "")
                            break
                        for tc in calls:
                            fn = tc.get("function") or {}
                            name = str(fn.get("name", ""))
                            try:
                                tool_args = json.loads(fn.get("arguments", "{}"))
                            except Exception:
                                tool_args = {}
                            try:
                                tool_result = await backend.call_tool(
                                    name,
                                    tool_args if isinstance(tool_args, dict) else {},
                                )
                            except Exception as exc:
                                errors.append(f"{name}: {exc}")
                                tool_result = {"error": str(exc)}
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.get("id", ""),
                                    "content": str(tool_result)[:4000],
                                }
                            )
                    claims = _parse_claims(
                        (task.eval_config or {}).get("gtfa_claims", [])
                    )
                    if output_text.strip() and claims:
                        try:
                            ev = evaluator.evaluate(
                                {"gtfa_claims": claims, "response": output_text}, None
                            )
                            score = float(getattr(ev, "score", 0.0) or 0.0)
                            passed = bool(getattr(ev, "passed", False))
                            d = getattr(ev, "details", None)
                            details = (
                                d
                                if isinstance(d, dict)
                                else ({"details": d} if d is not None else details)
                            )
                            if "evaluator" not in details:
                                details["evaluator"] = "gtfa"
                        except Exception as exc:
                            errors.append(f"gtfa_eval_failed: {exc}")
                    row = {
                        "task_id": task.task_id,
                        "input_tokens": total_in,
                        "output_tokens": total_out,
                        "tool_call_count": tool_calls,
                        "num_requests": reqs,
                        "e2e_latency_s": time.perf_counter() - start,
                        "output_text": output_text,
                        "errors": errors,
                        "eval_passed": passed,
                        "eval_score": score,
                        "eval_details": details,
                    }
                    res_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    res_f.flush()
                    done[task.task_id] = row
                out = {
                    "index": i,
                    "task_id": task.task_id,
                    "input_tokens": int(row.get("input_tokens", 0) or 0),
                    "output_tokens": int(row.get("output_tokens", 0) or 0),
                    "tool_call_count": int(row.get("tool_call_count", 0) or 0),
                    "num_requests": int(row.get("num_requests", 0) or 0),
                    "e2e_latency_s": float(row.get("e2e_latency_s", 0.0) or 0.0),
                    "output_text": str(row.get("output_text", "") or ""),
                    "errors": row.get("errors", []) or [],
                    "eval_passed": bool(row.get("eval_passed", False)),
                    "eval_score": float(row.get("eval_score", 0.0) or 0.0),
                    "eval_details": row.get("eval_details") or {"evaluator": "gtfa"},
                }
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                out_f.flush()
                print(
                    f"[{i}] {task.task_id}: {out['e2e_latency_s']:.2f}s, turns={out['num_requests']}, tool_calls={out['tool_call_count']}, score={out['eval_score']:.3f}, passed={out['eval_passed']}"
                )
        await backend.teardown()

    ordered = [
        done.get(
            t.task_id, {"output_text": "", "eval_score": 0.0, "eval_passed": False}
        )
        for t in tasks
    ]
    acc = sum(
        0.0
        if not str(r.get("output_text", "") or "").strip()
        else float(r.get("eval_score", 0.0) or 0.0)
        for r in ordered
    ) / max(len(ordered), 1)
    total = len(ordered)
    task_coverage = (
        (sum(1 for r in ordered if bool(r.get("eval_passed", False))) / total)
        if total
        else 0.0
    )
    print(
        f"FINAL: acc={acc:.3f}, task_coverage={task_coverage:.3f}, evaluated={total}/{total}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="AgentCAP single-agent MCP runner")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--dataset", default="mcp-atlas")
    p.add_argument("--num-tasks", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mcp-server-url", required=True)
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
