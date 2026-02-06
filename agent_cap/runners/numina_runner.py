import argparse
import json
import time
import statistics
from pathlib import Path
from typing import Optional, Dict, Any, Type
import importlib

from openai import OpenAI

from agent_cap import Tracer, StepType
from agent_cap.configs.cap_config import CAPConfig
from agent_cap.data_loader.numinamath_loader import NuminaMathLoader
from agent_cap.tools.calculator import calculator
from agent_cap.tools.schema import calculator_tool_schema
from agent_cap.utils.acc_metrics import extract_answer, compute_exact_match

from transformers import AutoTokenizer


# loader registry / dynamic import
LOADER_REGISTRY: Dict[str, Type] = {
    "numina": NuminaMathLoader,
}

def load_loader_by_spec(spec: str) -> Type:
    """
    spec format: "module:ClassName"
    e.g. "data_loader.numinamath_loader:NuminaMathLoader"
    """
    if ":" not in spec:
        raise ValueError(f"--loader_spec must be 'module:ClassName', got {spec!r}")
    mod, cls = spec.split(":", 1)
    m = importlib.import_module(mod)
    c = getattr(m, cls, None)
    if c is None:
        raise ValueError(f"Cannot find class {cls!r} in module {mod!r}")
    return c


# helpers: eval + stats
def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return float(xs[f] * (c - k) + xs[c] * (k - f))

def summarize(xs):
    if not xs:
        return {"n": 0, "mean": 0, "stdev": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
    return {
        "n": len(xs),
        "mean": float(statistics.mean(xs)),
        "stdev": float(statistics.pstdev(xs) if len(xs) > 1 else 0.0),
        "p50": float(percentile(xs, 50)),
        "p95": float(percentile(xs, 95)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }

def get_usage(resp) -> Dict[str, Optional[int]]:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "total_tokens": getattr(u, "total_tokens", None),
    }

def tool_content(tool_out: dict) -> str:
    if tool_out.get("status") == "ok":
        return tool_out.get("result_str", "")
    return f"ERROR: {tool_out.get('error')}"


def build_prompt(question: str, allow_tool: bool, tool_policy: str = "auto") -> str:
    base = (
        "Solve this math problem step by step.\n"
        "\n"
        "IMPORTANT - Output Rules (STRICT):\n"
        "1) Your final output MUST contain EXACTLY ONE occurrence of \\boxed{...}.\n"
        "2) The VERY LAST LINE must be exactly: \\boxed{YOUR_COMPLETE_ANSWER}\n"
        "3) Do NOT output multiple boxes. Do NOT include 'a)'/'b)' in the final line.\n"
        "4) Do NOT write anything after the final boxed line.\n"
        "\n"
        "Final Answer Format:\n"
        "Your last line must be ONLY the boxed answer, e.g.:\n"
        "\\boxed{(-\\infty, \\frac{5}{4}]}\n"
        "\\boxed{\\frac{2}{9} \\le x^4 + y^4 \\le 8}\n"
        "\\boxed{42}\n"
        "\n"
        "Requirements:\n"
        "- The content inside \\boxed{} must be COMPLETE (full intervals, full inequalities, etc.)\n"
        "- Use \\frac instead of \\dfrac for fractions\n"
        "\n"
        "If the problem has multiple parts (a/b/c), answer ONLY ONE part and output ONLY ONE final boxed answer.\n"
    )

    if allow_tool:
        base += "\nYou may use the calculator tool if it helps.\n"
        if tool_policy == "required":
            base += 'You MUST call the calculator tool at least once. If you do not need it, call it with expr="0".\n'

    base += f"\nProblem: {question}"
    return base


def maybe_sleep_ms(ms: float, enable_sleep: bool) -> float:
    if (not enable_sleep) or ms <= 0:
        return 0.0
    time.sleep(ms / 1000.0)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", type=str, required=True)
    ap.add_argument("--model_id", type=str, required=True)

    ap.add_argument("--mode", choices=["no-tool", "tool"], default="no-tool")
    ap.add_argument(
        "--tool_policy",
        choices=["auto", "required", "none"],
        default="auto",
        help="Tool policy: auto=may call tool; required=must call tool; none=never call tool",
    )
    ap.add_argument("--n", type=int, default=100)

    ap.add_argument("--max_tokens", type=int, default=16384)  
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--summary_path", type=str, default=None)

    # Skeleton controls
    ap.add_argument("--enable_skeleton_sleep", action="store_true")
    ap.add_argument("--planning_ms", type=float, default=80.0)
    ap.add_argument("--retrieval_ms", type=float, default=200.0)
    ap.add_argument("--embedding_ms", type=float, default=70.0)
    ap.add_argument("--prefill_ms", type=float, default=0.0)
    ap.add_argument("--code_ms", type=float, default=160.0)
    ap.add_argument("--reasoning_ms", type=float, default=180.0)

    # Metrics controls
    ap.add_argument("--enable_metrics", action="store_true")
    ap.add_argument("--debug", action="store_true", help="Print debug info for failed samples")

    ap.add_argument("--dataset", type=str, default="numina", choices=sorted(LOADER_REGISTRY.keys()))
    ap.add_argument("--loader_spec", type=str, default=None)  
    ap.add_argument("--dataset_split", type=str, default="test")

    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if args.mode == "no-tool" and args.tool_policy != "none":
        raise ValueError("In no-tool mode, please use --tool_policy none for a strict baseline.")
    if args.mode == "tool" and args.tool_policy == "none":
        raise ValueError("tool_policy=none conflicts with mode=tool. Use --mode no-tool instead.")

    out_dir = Path(args.out_dir or f"baselines/{args.dataset}_{args.mode}_skeleton_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path or f"baselines/{args.dataset}_{args.mode}_skeleton_summary.json")

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")


    if args.loader_spec:
        LoaderCls = load_loader_by_spec(args.loader_spec)
        dataset_name_for_cfg = args.dataset  # for bookkeeping
    else:
        LoaderCls = LOADER_REGISTRY[args.dataset]
        dataset_name_for_cfg = args.dataset

    cfg = CAPConfig(
        dataset_names=[dataset_name_for_cfg],
        metrics=["em"],
        model_id=args.model_id,
        dataset_split=args.dataset_split,
    )
    loader = LoaderCls(cfg)
    xs = loader.get_input()[: args.n]
    ys = loader.get_target()[: args.n]

    dataset_for_eval = "numinamath" if args.dataset.lower() == "numina" else args.dataset.lower()

    ems = []
    e2e_lats = []
    tool_called = 0
    tool_calls_total = 0
    trace_files = []

    # for summary token stats
    question_tokens_list = []
    output_tokens_list = []
    
    truncated_count = 0

    for i, (q, gold) in enumerate(zip(xs, ys)):
        tracer = Tracer(name=f"{args.dataset}-{args.mode}-skeleton")
        tracer.start()

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_tool_calls = 0

        with tracer.step("agent_run", StepType.OTHER) as top:
            top.metadata["sample_id"] = i
            top.metadata["mode"] = args.mode
            top.metadata["dataset"] = args.dataset

            # skeleton
            with tracer.step("planning", StepType.PLANNING) as step:
                step.metadata["slept_ms"] = maybe_sleep_ms(args.planning_ms, args.enable_skeleton_sleep)

            with tracer.step("retrieval", StepType.RETRIEVAL) as step:
                step.metadata["slept_ms"] = maybe_sleep_ms(args.retrieval_ms, args.enable_skeleton_sleep)

            with tracer.step("embedding", StepType.EMBEDDING) as step:
                step.metadata["slept_ms"] = maybe_sleep_ms(args.embedding_ms, args.enable_skeleton_sleep)
                step.metadata["embedding_dim"] = 768

            with tracer.step("llm_prefill", StepType.PREFILL) as step:
                step.metadata["slept_ms"] = maybe_sleep_ms(args.prefill_ms, args.enable_skeleton_sleep)


            top.metadata["question_tokens"] = int(len(tok.encode(q, add_special_tokens=False)))

            prompt = build_prompt(q, allow_tool=(args.mode == "tool"), tool_policy=args.tool_policy)
            if i == 0:
                print("\n========== PROMPT SENT TO MODEL ==========")
                print(prompt)
                print("==========================================\n")

            messages = [{"role": "user", "content": prompt}]

            # resolve tool_choice
            if args.tool_policy == "required":
                tool_choice = {"type": "function", "function": {"name": "calculator"}}
            elif args.tool_policy == "auto":
                tool_choice = "auto"
            else:
                tool_choice = "none"

            if args.mode == "no-tool":
                with tracer.step("llm_decode", StepType.DECODE) as step:
                    t0 = time.perf_counter()
                    resp = client.chat.completions.create(
                        model=args.model_id,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    t1 = time.perf_counter()

                    msg = resp.choices[0].message

                    # strict no-tool assertion 
                    if getattr(msg, "tool_calls", None):
                        raise RuntimeError("no-tool mode violation: model returned tool_calls unexpectedly")

                    out = msg.content or ""
                
                    finish_reason = resp.choices[0].finish_reason
                    step.metadata["finish_reason"] = finish_reason
                    if finish_reason == "length":
                        step.metadata["truncated"] = True
                        truncated_count += 1
                    
                    u = get_usage(resp)
                    step.metadata.update(u)
                    step.metadata["elapsed_ms"] = (t1 - t0) * 1000.0

                    #  extract + exact match 
                    pred = extract_answer(out, dataset_for_eval)
                    m = compute_exact_match([pred], [gold])
                    em_val = 1 if m.get("correct", 0) == 1 else 0

                    step.metadata["raw_output"] = out
                    step.metadata["pred"] = pred
                    step.metadata["gold"] = gold
                    step.metadata["em"] = em_val
                    step.metadata["no_answer"] = m.get("no_answer", 0)
                    
                    if args.debug and (i < 5 or em_val == 0):
                        print(f"\n{'='*70}")
                        print(f"Sample {i} | EM={em_val} | Truncated={finish_reason == 'length'}")
                        print(f"GOLD: {gold}")
                        print(f"PRED: {pred}")
                        print(f"RAW OUTPUT (last 800 chars):")
                        print(out[-800:] if len(out) > 800 else out)
                        print(f"{'='*70}\n")

                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = step.metadata.get(k)
                    if isinstance(v, int):
                        if k == "prompt_tokens":
                            total_prompt_tokens += v
                        elif k == "completion_tokens":
                            total_completion_tokens += v
                        else:
                            total_tokens += v

                top.metadata["pred"] = step.metadata["pred"]
                top.metadata["gold"] = gold
                top.metadata["em"] = int(step.metadata["em"])
                top.metadata["no_answer"] = int(step.metadata.get("no_answer", 0))

            else:
                with tracer.step("llm_decode_1", StepType.DECODE) as step:
                    t0 = time.perf_counter()
                    resp1 = client.chat.completions.create(
                        model=args.model_id,
                        messages=messages,
                        tools=calculator_tool_schema,
                        tool_choice=tool_choice,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    t1 = time.perf_counter()

                    msg1 = resp1.choices[0].message
                    
                    finish_reason = resp1.choices[0].finish_reason
                    step.metadata["finish_reason"] = finish_reason
                    if finish_reason == "length":
                        step.metadata["truncated"] = True
                        truncated_count += 1
                    
                    u1 = get_usage(resp1)
                    step.metadata.update(u1)
                    step.metadata["elapsed_ms"] = (t1 - t0) * 1000.0
                    step.metadata["tool_calls_n"] = len(getattr(msg1, "tool_calls", None) or [])

                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = step.metadata.get(k)
                    if isinstance(v, int):
                        if k == "prompt_tokens":
                            total_prompt_tokens += v
                        elif k == "completion_tokens":
                            total_completion_tokens += v
                        else:
                            total_tokens += v

                tool_calls = getattr(resp1.choices[0].message, "tool_calls", None) or []
                total_tool_calls += len(tool_calls)

                if args.tool_policy == "required" and len(tool_calls) == 0:
                    # raise RuntimeError("tool_policy=required but model returned no tool_calls")
                    
                    # 1) record in metadata
                    step.metadata["required_but_no_tool_calls"] = True   #  llm_decode_1‘s step
                    top.metadata["required_but_no_tool_calls"] = True

                    # 2) fallback: use first output as final answer and continue
                    final_out = resp1.choices[0].message.content or ""
                    

                if tool_calls:
                    tool_called += 1
                    tool_calls_total += len(tool_calls)

                    with tracer.step("tool_call_1", StepType.TOOL_CALLING) as step:
                        step.metadata["tool_calls_n"] = len(tool_calls)
                        step.metadata["tool_name"] = "calculator"

                        t_tool0 = time.perf_counter()
                        messages.append({
                            "role": "assistant",
                            "content": resp1.choices[0].message.content,
                            "tool_calls": tool_calls,
                        })

                        for tc in tool_calls:
                            args_json = json.loads(tc.function.arguments)
                            tool_out = calculator(args_json.get("expr", ""))
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_content(tool_out),
                            })

                        t_tool1 = time.perf_counter()
                        step.metadata["tool_exec_elapsed_ms"] = (t_tool1 - t_tool0) * 1000.0

                    with tracer.step("llm_decode_2", StepType.DECODE) as step:
                        t2_0 = time.perf_counter()
                        resp2 = client.chat.completions.create(
                            model=args.model_id,
                            messages=messages,
                            tools=calculator_tool_schema,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                        t2_1 = time.perf_counter()

                        out = resp2.choices[0].message.content or ""
                        
                        finish_reason2 = resp2.choices[0].finish_reason
                        step.metadata["finish_reason"] = finish_reason2
                        if finish_reason2 == "length":
                            step.metadata["truncated"] = True
                        
                        u2 = get_usage(resp2)
                        step.metadata.update(u2)
                        step.metadata["elapsed_ms"] = (t2_1 - t2_0) * 1000.0

                    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        v = step.metadata.get(k)
                        if isinstance(v, int):
                            if k == "prompt_tokens":
                                total_prompt_tokens += v
                            elif k == "completion_tokens":
                                total_completion_tokens += v
                            else:
                                total_tokens += v
                    final_out = out
                else:
                    final_out = resp1.choices[0].message.content or ""

                pred = extract_answer(final_out, dataset_for_eval)
                m = compute_exact_match([pred], [gold])
                em_val = 1 if m.get("correct", 0) == 1 else 0

                top.metadata["raw_output"] = final_out
                top.metadata["pred"] = pred
                top.metadata["gold"] = gold
                top.metadata["em"] = em_val
                top.metadata["no_answer"] = int(m.get("no_answer", 0))
                
                if args.debug and (i < 5 or em_val == 0):
                    print(f"\n{'='*70}")
                    print(f"Sample {i} | EM={em_val}")
                    print(f"GOLD: {gold}")
                    print(f"PRED: {pred}")
                    print(f"RAW OUTPUT (last 800 chars):")
                    print(final_out[-800:] if len(final_out) > 800 else final_out)
                    print(f"{'='*70}\n")

            with tracer.step("code_execution", StepType.CODE_EXECUTION) as step:
                step.metadata["slept_ms"] = maybe_sleep_ms(args.code_ms, args.enable_skeleton_sleep)
                step.metadata["language"] = "python"

            with tracer.step("reasoning", StepType.REASONING) as step:
                step.metadata["slept_ms"] = maybe_sleep_ms(args.reasoning_ms, args.enable_skeleton_sleep)

            top.metadata["total_prompt_tokens"] = total_prompt_tokens if total_prompt_tokens > 0 else None
            top.metadata["total_completion_tokens"] = total_completion_tokens if total_completion_tokens > 0 else None
            top.metadata["total_tokens"] = total_tokens if total_tokens > 0 else None
            top.metadata["total_tool_calls"] = total_tool_calls
            top.metadata["input_tokens_prompt"] = total_prompt_tokens if total_prompt_tokens > 0 else None
            top.metadata["output_tokens"] = total_completion_tokens if total_completion_tokens > 0 else None

        trace = tracer.stop()

        if args.enable_metrics:
            question_tokens_list.append(int(top.metadata.get("question_tokens") or 0))
            output_tokens_list.append(int(top.metadata.get("output_tokens") or 0))

        p = out_dir / f"trace_{i+1:03d}.json"
        p.write_text(trace.to_json(indent=2), encoding="utf-8")
        trace_files.append(str(p))

        if args.enable_metrics:
            ems.append(int(top.metadata.get("em", 0)))
            e2e_lats.append(float(trace.total_duration_ms))

    if args.enable_metrics:
        n = len(xs)
        summary = {
            "workload": f"{args.dataset} split={args.dataset_split} (n={n}) | mode={args.mode} | skeleton={'sleep' if args.enable_skeleton_sleep else '0ms'}",
            "base_url": args.base_url,
            "model_id": args.model_id,
            "dataset": args.dataset,
            "dataset_split": args.dataset_split,
            "mode": args.mode,
            "n": n,
            "em": {"mean": float(sum(ems) / n), "correct": int(sum(ems))},
            "latency_ms": summarize(e2e_lats),
            "tool_call_rate": float(tool_called / n) if args.mode == "tool" else 0.0,
            "tool_calls_total": int(tool_calls_total) if args.mode == "tool" else 0,
            "truncated_count": truncated_count,  
            "tokens": {
                "avg_question_tokens": float(sum(question_tokens_list) / len(question_tokens_list)) if question_tokens_list else 0.0,
                "avg_output_tokens": float(sum(output_tokens_list) / len(output_tokens_list)) if output_tokens_list else 0.0,
                "max_output_tokens": int(max(output_tokens_list)) if output_tokens_list else 0,
            },
            "trace_dir": str(out_dir),
            "trace_files": trace_files[:5] + (["..."] if len(trace_files) > 5 else []),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    else:
        print(f"[OK] saved {len(trace_files)} traces -> {out_dir}")
        print("[OK] metrics disabled (no summary computed)")

if __name__ == "__main__":
    main()