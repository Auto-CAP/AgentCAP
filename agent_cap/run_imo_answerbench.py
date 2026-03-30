import argparse
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from math_verify import parse, verify

from agent_cap.benchmarks import load_benchmark
from agent_cap.backends.math_python_backend import MathPythonBackend
from agent_cap.core.agentic_loop import run_agentic_loop
from agent_cap.server.streaming_client import StreamingChatClient
from openai_harmony import HarmonyEncodingName, load_harmony_encoding


SYSTEM_PROMPT = """
You are an expert mathematical problem solver with access to a Python execution tool.

Your goal is to solve the problem rigorously and correctly.

Rules:
- Justify important steps.
- Use the python tool for calculations, symbolic checks, sanity checks, and small experiments when useful.
- Keep Python usage focused and relevant.
- If you use the python tool, incorporate its output into your reasoning.
- Put your final answer in \\boxed{...}.
""".strip()


class CFG:
    system_prompt = """You are an elite mathematical problem solver with expertise at the International Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through rigorous mathematical reasoning.

# General rules:
- Every claim must be logically justified.
- Do not guess, speculate, or rely on unproven intuition.
- If a full solution is not possible, only present results that can be proven rigorously.
- Do not include failed attempts, scratch work, or informal commentary.

# General Problem-Solving Approachs:
1. UNDERSTAND: Carefully read and rephrase the problem in your own words. Identify what is given, what needs to be found, and any constraints.Pay attention to all mathematical symbols and words, and understand what they mean.
2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, techniques, patterns, or analogous problems. Don't commit to one approach immediately.
3. PLAN: Select the most promising approach and outline key steps before executing.
4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.
5. VERIFY: Check your answer by substituting back, testing edge cases, or using alternative methods. Ensure logical consistency throughout.

# General Mathematical Reasoning Principles:
- Break complex problems into smaller, manageable sub-problems
- Look for patterns, symmetries, and special cases that provide insight
- Use concrete examples to build intuition before generalizing
- When attempting to generalizing concrete examples into conclusions, always try a few examples first.Do not settle on the first example studied.
- If the problem involves multiple cases, reason through each one of them carefully. Do not take shortcuts.Do not assume the results from previous cases generalize to the next one.
- Consider extreme cases and boundary conditions
- If stuck, try working backwards from the desired result
- Be willing to restart with a different approach if needed
- For induction arguments, always check the base case and make sure it holds.

# Verification Requirements:
- Cross-check arithmetic and algebraic manipulations
- Verify that your solution satisfies all problem constraints
- Test your answer with simple cases or special values when possible
- Ensure dimensional consistency and reasonableness of the result

Before finalizing your answer, carefully review the solution and ensure:
- Logical correctness of every step.
- Compliance with all instructions.
- Clarity, structure, and mathematical rigor.

# Output Format:
### Summary
- **Verdict**: State whether a complete solution is obtained or only a partial solution.
- **Result**:
  - If complete: clearly state the final answer with a short summary. The final answer must be a non-negative integer between 0 and 99999. Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}
  - If partial: state the main rigorously proven results.

Think step-by-step and show your complete reasoning process. Quality of reasoning is as important as the final answer.

# Combinatorics Specific Hints:
- If the problem is a combinatorics problem involving tiling and lower or upper bounds, look for a few invariants. Do not settle on the first one.
- If the problem require representing it as a graph, be careful of hidden assumptions and restrictions. Do not miss them.
- If the problem involves tiling or colouring a square grid, think very carefully about what happens along the diagonal.
- If the problem involves a turn based game, think about the optimal strategy for each of the players.

# Algebra Specific Hints:
- The word "between" means not-enclusive of both endpoints of an interval.
- If this problem requires proving an inequality, recall some common equalities in Olympiads:
* triangle inequality
* Vasc's inequality
* Muirhead inequality
* Cauchy Schwarz inequality
* Rearrangement inequailty
* QM-AM-GM-HM inequality

- If the problem involves functional equation, a common approach is to rewrite the equations and try to find relations between function values.

# Geometry Specific Hints:
- If the problem is a geometry problem and the conclusion cannot be deduced directly, try drawing additional points, lines, and circles.
"""

    tool_prompt = (
        'Use this tool to execute Python code for:\n'
        '- Complex calculations that would be error-prone by hand\n'
        '- Numerical verification of analytical results\n'
        '- Generating examples or testing conjectures\n'
        '- Visualizing problem structure when helpful\n'
        '- Brute-force verification for small cases\n\n'
        'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
        'Always use print() to display results. Write clear, well-commented code.\n\n'
        "Remember: Code should support your mathematical reasoning, not replace it. "
        "Explain what you're computing and why before running code."
    )

    preference_prompt = (
        'You have access to `math`, `numpy`, and `sympy` for:\n\n'
        '# Symbolic Computation (sympy):\n'
        '- Algebraic manipulation and simplification\n'
        '- Solving equations and systems of equations\n'
        '- Symbolic differentiation and integration\n'
        '- Number theory functions (primes, divisors, modular arithmetic)\n'
        '- Polynomial operations and factorization\n'
        '- Working with mathematical expressions symbolically\n\n'
        '# Numerical Computation (numpy):\n'
        '- Array operations and linear algebra\n'
        '- Efficient numerical calculations for large datasets\n'
        '- Matrix operations and eigenvalue problems\n'
        '- Statistical computations\n\n'
        '# Mathematical Functions (math):\n'
        '- Standard mathematical functions (trig, log, exp)\n'
        '- Constants like pi and e\n'
        '- Basic operations for single values\n\n'
        'Best Practices:\n'
        '- Use sympy for exact symbolic answers when possible\n'
        '- Use numpy for numerical verification and large-scale computation\n'
        '- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n'
        '- Document your computational strategy clearly\n'
        '- Validate computational results against known cases or theoretical bounds'
    )


@dataclass
class RuntimeConfig:
    served_model_name: str
    model_path: str
    port: int
    seed: int
    kv_cache_dtype: str
    dtype: str
    stream_interval: int
    context_tokens: int
    batch_size: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    server_timeout: int
    preload_workers: int


class VLLMInfraGPTOSS:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.port = cfg.port
        self.base_url = f"http://127.0.0.1:{cfg.port}"
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self.server_process: Optional[subprocess.Popen] = None
        self.log_file = None
        self.client: Optional[StreamingChatClient] = None

    def start(self) -> None:
        self._preload_model_weights()
        self.server_process = self._start_server()
        self._wait_for_server()
        self.client = StreamingChatClient(base_url=self.base_url)

    def stop(self) -> None:
        if self.server_process is not None and self.server_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                except Exception:
                    pass

        if self.log_file is not None:
            try:
                self.log_file.close()
            except Exception:
                pass

    def _preload_model_weights(self) -> None:
        if not os.path.isdir(self.cfg.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.cfg.model_path}")

        print(f"Loading model weights from {self.cfg.model_path} into OS Page Cache...")
        start_time = time.time()

        files_to_load: List[str] = []
        total_size = 0

        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)

        def _read_file(path: str) -> None:
            with open(path, "rb") as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass

        with ThreadPoolExecutor(max_workers=self.cfg.preload_workers) as executor:
            list(executor.map(_read_file, files_to_load))

        elapsed = time.time() - start_time
        print(f"Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n")

    def _start_server(self) -> subprocess.Popen:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--seed",
            str(self.cfg.seed),
            "--model",
            self.cfg.model_path,
            "--served-model-name",
            self.cfg.served_model_name,
            "--tensor-parallel-size",
            str(self.cfg.tensor_parallel_size),
            "--max-num-seqs",
            str(self.cfg.batch_size),
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.cfg.port),
            "--dtype",
            self.cfg.dtype,
            "--kv-cache-dtype",
            self.cfg.kv_cache_dtype,
            "--max-model-len",
            str(self.cfg.context_tokens),
            "--stream-interval",
            str(self.cfg.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
            "--enable-prefix-caching",
        ]

        self.log_file = open("vllm_server.log", "w")
        print("Launching vLLM:")
        print(" ".join(cmd))
        return subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _wait_for_server(self) -> None:
        print("Waiting for vLLM server...")
        start_time = time.time()
        models_url = f"{self.base_url}/v1/models"

        for _ in range(self.cfg.server_timeout):
            if _ % 100 == 0:
                print(f'waiting for server to start: poll count={_}')
            if self.server_process is None:
                raise RuntimeError("Server process was not created.")

            return_code = self.server_process.poll()
            if return_code is not None:
                self.log_file.flush()
                with open("vllm_server.log", "r") as log_file:
                    logs = log_file.read()
                raise RuntimeError(f"Server died with code {return_code}. Full logs:\n{logs}\n")

            try:
                req = urllib.request.Request(models_url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - start_time
                        print(f"Server is ready (took {elapsed:.2f} seconds).\n")
                        return
            except Exception:
                time.sleep(1)

        raise RuntimeError("Server failed to start (timeout).\n")


def last_boxed_only_string(text: str) -> Optional[str]:
    positions = [m.start() for m in re.finditer(r"\\boxed\b", text)]
    if not positions:
        return None

    for start in reversed(positions):
        i = start + len(r"\boxed")
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            continue

        depth = 0
        j = i
        while j < len(text):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:j + 1]
            j += 1

    return None


def remove_boxed(boxed_str: str) -> str:
    boxed_str = boxed_str.strip()
    if not boxed_str.startswith(r"\boxed"):
        return boxed_str

    i = len(r"\boxed")
    while i < len(boxed_str) and boxed_str[i].isspace():
        i += 1
    if i >= len(boxed_str) or boxed_str[i] != "{":
        return boxed_str

    return boxed_str[i + 1:-1].strip()


def extract_last_boxed_content(text: str) -> Optional[str]:
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def is_equiv(str1: str, str2: str, verbose: bool = False) -> bool:
    if "$" not in str1:
        str1 = "$" + str1 + "$"
    if "$" not in str2:
        str2 = "$" + str2 + "$"

    gold = parse(str2)
    pred = parse(str1)
    return verify(gold, pred)


def compute_score(solution_str: str, ground_truth: str) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(e)

    return retval


def solve_one_task(
    task,
    client: StreamingChatClient,
    model: str,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    startup_timeout: float,
    exec_timeout: float,
    preload: str,
    auto_print_last_expr: bool,
    stop_token_ids: Optional[List[int]],
) -> Dict[str, Any]:
    backend = MathPythonBackend(
        startup_timeout=startup_timeout,
        exec_timeout=exec_timeout,
        preload=preload,
        auto_print_last_expr=auto_print_last_expr,
    )

    backend.setup(task.eval_config or {})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *task.messages,
    ]

    try:
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
    finally:
        backend.teardown()

    response_text = loop_result.response or ""
    boxed_answer = extract_last_boxed_content(response_text)
    expected = (task.eval_config or {}).get("expected", None)
    score = compute_score(response_text, expected) if expected is not None else 0.0

    return {
        "task_id": task.id,
        "task_name": task.name,
        "category": task.category,
        "expected": expected,
        "predicted": boxed_answer,
        "score": score,
        "correct": score >= 1.0,
        "response": response_text,
        "tool_calls": loop_result.tool_calls,
        "tool_latencies_ms": list(loop_result.tool_latencies_ms),
        "input_tokens": loop_result.input_tokens,
        "output_tokens": loop_result.output_tokens,
        "latency_ms": loop_result.latency_ms,
        "ttft_ms": loop_result.ttft_ms,
        "tpot_ms_avg": loop_result.tpot_ms_avg,
        "tpot_ms_p99": loop_result.tpot_ms_p99,
        "errors": list(loop_result.errors),
    }


def print_task_result(index: int, total: int, result: Dict[str, Any]) -> None:
    status = "✅" if result["correct"] else "❌"
    print("\n" + "=" * 100)
    print(f"[{index}/{total}] {result['task_id']}  {status}")
    print(f"Name: {result['task_name']}")
    print(f"Expected:  {result['expected']}")
    print(f"Predicted: {result['predicted']}")
    print(f"Score:     {result['score']:.1f}")
    print(
        f"Tokens in/out: {result['input_tokens']}/{result['output_tokens']} | "
        f"Latency: {result['latency_ms']:.1f} ms | "
        f"TTFT: {result['ttft_ms']:.1f} ms | "
        f"TPOT(avg): {result['tpot_ms_avg']:.1f} ms | "
        f"Python calls: {result['tool_calls']}"
    )
    if result["errors"]:
        print(f"Errors: {result['errors']}")
    print("\nResponse preview:")
    print(result["response"])


def print_summary(results: List[Dict[str, Any]], wall_time_s: float) -> None:
    total = len(results)
    total_score = sum(float(r["score"]) for r in results)
    accuracy = 100.0 * total_score / total if total else 0.0

    total_tool_calls = sum(int(r["tool_calls"]) for r in results)
    avg_in = statistics.mean([r["input_tokens"] for r in results]) if results else 0.0
    avg_out = statistics.mean([r["output_tokens"] for r in results]) if results else 0.0
    avg_latency = statistics.mean([r["latency_ms"] for r in results]) if results else 0.0
    avg_ttft = statistics.mean([r["ttft_ms"] for r in results]) if results else 0.0
    avg_tpot = statistics.mean([r["tpot_ms_avg"] for r in results]) if results else 0.0

    answer_counter = Counter(r["predicted"] for r in results if r["predicted"] is not None)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"Tasks solved:        {total}")
    print(f"Total score:         {total_score:.1f}")
    print(f"Average score:       {accuracy:.1f}%")
    print(f"Wall time:           {wall_time_s:.2f}s")
    print(f"Avg input tokens:    {avg_in:.1f}")
    print(f"Avg output tokens:   {avg_out:.1f}")
    print(f"Avg latency:         {avg_latency:.1f} ms")
    print(f"Avg TTFT:            {avg_ttft:.1f} ms")
    print(f"Avg TPOT:            {avg_tpot:.1f} ms")
    print(f"Total python calls:  {total_tool_calls}")

    if answer_counter:
        print("\nMost common predicted answers:")
        for ans, cnt in answer_counter.most_common(10):
            print(f"  {ans}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser()

    # Benchmark / agent settings
    parser.add_argument("--model", type=str, default="gpt-oss")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--exec-timeout", type=float, default=5.0)
    parser.add_argument(
        "--preload",
        type=str,
        default="minimal",
        choices=["none", "minimal", "full"],
    )
    parser.add_argument("--auto-print-last-expr", action="store_true")

    # vLLM server launch settings
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", type=str, default="gpt-oss")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--kv-cache-dtype", type=str, default="fp8_e4m3")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--stream-interval", type=int, default=200)
    parser.add_argument("--context-tokens", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--server-timeout", type=int, default=3600)
    parser.add_argument("--preload-workers", type=int, default=8)

    args = parser.parse_args()
    t0 = time.time()

    runtime_cfg = RuntimeConfig(
        served_model_name=args.served_model_name,
        model_path=args.model_path,
        port=args.port,
        seed=args.seed,
        kv_cache_dtype=args.kv_cache_dtype,
        dtype=args.dtype,
        stream_interval=args.stream_interval,
        context_tokens=args.context_tokens,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        server_timeout=args.server_timeout,
        preload_workers=args.preload_workers,
    )

    infra = VLLMInfraGPTOSS(runtime_cfg)
    results: List[Dict[str, Any]] = []

    try:
        infra.start()
        if infra.client is None:
            raise RuntimeError("Streaming client was not initialized.")

        tasks = load_benchmark("imo_answerbench", num_tasks=args.num_tasks, seed=args.seed)

        print(f"Loaded {len(tasks)} IMO AnswerBench tasks")
        print(f"Server: {infra.base_url}")
        print(f"Model:  {args.model}")

        for i, task in enumerate(tasks, start=1):
            result = solve_one_task(
                task=task,
                client=infra.client,
                model=args.model,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                startup_timeout=args.startup_timeout,
                exec_timeout=args.exec_timeout,
                preload=args.preload,
                auto_print_last_expr=args.auto_print_last_expr,
                stop_token_ids=infra.stop_token_ids,
            )
            results.append(result)
            print_task_result(i, len(tasks), result)

            running_score = sum(float(r["score"]) for r in results)
            running_avg = 100.0 * running_score / len(results)
            print(f"\nRunning average score: {running_score:.1f}/{len(results)} ({running_avg:.1f}%)")

        wall_time_s = time.time() - t0
        print_summary(results, wall_time_s)

    finally:
        infra.stop()


if __name__ == "__main__":
    main()