import argparse
import asyncio
import json
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
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any, Dict, List, Optional
from google import genai

import aiohttp
from math_verify import parse, verify

from agent_cap.benchmarks import load_benchmark
from agent_cap.backends.math_python_backend import MathPythonBackend
from agent_cap.runner.tool_backends import ToolBackend
from agent_cap.runner.unified_runner import UnifiedTask, run_single_example
from openai_harmony import HarmonyEncodingName, load_harmony_encoding


SYSTEM_PROMPT = (
"""You are an elite mathematical problem solver with expertise at the International Mathematical Olympiad (IMO) level.

# Output Format:
### Summary
  - Clearly state the final answer with a short summary. The final answer must be a non-negative integer between 0 and 99999. Place your final numerical answer inside \boxed{}, e.g., \boxed{42}.
  - You must always provide a summary of the solution, in addition to the final answer.

"""
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


class AsyncMathPythonBackend(ToolBackend):
    def __init__(
        self,
        startup_timeout: float = 30.0,
        exec_timeout: float = 5.0,
        preload: str = "minimal",
        auto_print_last_expr: bool = True,
    ):
        self._backend = MathPythonBackend(
            startup_timeout=startup_timeout,
            exec_timeout=exec_timeout,
            preload=preload,
            auto_print_last_expr=auto_print_last_expr,
        )

    async def setup(self, task_config: Dict[str, Any]) -> bool:
        return await asyncio.to_thread(self._backend.setup, task_config)

    async def list_tools(self) -> List[Dict[str, Any]]:
        return self._backend.get_tool_definitions()

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        result = await asyncio.to_thread(self._backend.execute, name, "call", arguments)
        if result.success:
            return [{"type": "text", "text": result.output}]
        raise RuntimeError(result.output)

    async def teardown(self) -> None:
        await asyncio.to_thread(self._backend.teardown)


class GeminiEquivalenceJudge:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client()

    def _extract_json_bool(self, text: str) -> Optional[bool]:
        text = text.strip()

        # Try direct JSON first
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "equivalent" in data:
                return bool(data["equivalent"])
        except Exception:
            pass

        # Try to find a JSON object inside the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict) and "equivalent" in data:
                    return bool(data["equivalent"])
            except Exception:
                pass

        # Fallback: simple heuristic
        lowered = text.lower()
        if '"equivalent": true' in lowered or lowered.startswith("yes"):
            return True
        if '"equivalent": false' in lowered or lowered.startswith("no"):
            return False

        return None

    def judge_equivalence(self, predicted: Optional[str], expected: Optional[str]) -> Dict[str, Any]:
        if predicted is None or expected is None:
            return {
                "equivalent": False,
                "raw_response": "Missing predicted or expected value.",
            }

        prompt = f"""You are a strict mathematical answer equivalence judge.

Determine whether the following two final answers are mathematically equivalent.

Rules:
- Focus only on whether the predicted answer and expected answer represent the same mathematical value.
- Ignore formatting differences like whitespace, commas, LaTeX wrappers, or extra prose.
- If they represent the same integer or the same mathematical expression/value, return equivalent=true.
- If they do not represent the same value, return equivalent=false.
- Return ONLY valid JSON with this exact schema:
{{"equivalent": true_or_false, "reason": "short reason"}}

Predicted answer:
{predicted}

Expected answer:
{expected}
"""

        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        text = getattr(resp, "text", None) or getattr(resp, "output_text", None) or str(resp)
        equivalent = self._extract_json_bool(text)

        return {
            "equivalent": bool(equivalent) if equivalent is not None else False,
            "raw_response": text,
        }

    async def judge_equivalence_async(
        self, predicted: Optional[str], expected: Optional[str]
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self.judge_equivalence, predicted, expected)
    

async def apply_gemini_judgment(
    result: Dict[str, Any],
    judge: GeminiEquivalenceJudge,
    max_retries: int = 5,
) -> Dict[str, Any]:
    predicted = result.get("predicted")
    expected = result.get("expected")

    # Preserve original rule-based scoring
    result["rule_score"] = result["score"]
    result["rule_correct"] = result["correct"]

    last_raw_response = None
    gemini_equivalent = False
    gemini_attempts = 0

    for attempt in range(1, max_retries + 1):
        gemini_attempts = attempt
        try:
            gemini_eval = await judge.judge_equivalence_async(predicted, expected)
            last_raw_response = gemini_eval.get("raw_response")
            gemini_equivalent = bool(gemini_eval.get("equivalent", False))

            if gemini_equivalent:
                break
        except Exception as exc:
            last_raw_response = f"Gemini judge attempt {attempt} failed: {exc}"

    result["gemini_equivalent"] = gemini_equivalent
    result["gemini_judge_response"] = last_raw_response
    result["gemini_attempts"] = gemini_attempts

    # After up to max_retries attempts, mark wrong if Gemini never said equivalent
    result["score"] = 1.0 if gemini_equivalent else 0.0
    result["correct"] = gemini_equivalent

    return result
    


class VLLMInfraGPTOSS:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.port = cfg.port
        self.base_url = f"http://127.0.0.1:{cfg.port}/v1"
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self.server_process: Optional[subprocess.Popen] = None
        self.log_file = None

    def start(self) -> None:
        self._preload_model_weights()
        self.server_process = self._start_server()
        self._wait_for_server()

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
        print(
            f"Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n"
        )

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
            "--tool-call-parser",
            "openai",
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
        models_url = f"{self.base_url}/models"

        for i in range(self.cfg.server_timeout):
            if i % 100 == 0:
                print(f"waiting for server to start: poll count={i}")
            if self.server_process is None:
                raise RuntimeError("Server process was not created.")

            return_code = self.server_process.poll()
            if return_code is not None:
                self.log_file.flush()
                with open("vllm_server.log", "r") as log_file:
                    logs = log_file.read()
                raise RuntimeError(
                    f"Server died with code {return_code}. Full logs:\n{logs}\n"
                )

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
                    return text[start : j + 1]
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

    return boxed_str[i + 1 : -1].strip()


def extract_last_boxed_content(text: str) -> Optional[str]:
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def is_equiv(str1: str, str2: str, verbose: bool = False) -> bool:
    del verbose
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
    except Exception as exc:
        print(exc)

    return retval


def _build_result_dict(task: Any, example_result: Any) -> Dict[str, Any]:
    response_text = example_result.output_text or ""
    boxed_answer = extract_last_boxed_content(response_text)
    expected = (task.eval_config or {}).get("expected")
    score = compute_score(response_text, expected) if expected is not None else 0.0

    avg_ttft_ms = 0.0
    if example_result.num_requests > 0:
        avg_ttft_ms = 1000.0 * (
            example_result.total_prefill_time_s / example_result.num_requests
        )

    avg_tpot_ms = 0.0
    if example_result.total_output_tokens > 0:
        avg_tpot_ms = (
            1000.0
            * example_result.total_decode_time_s
            / example_result.total_output_tokens
        )

    return {
        "task_id": task.id,
        "task_name": task.name,
        "category": task.category,
        "expected": expected,
        "predicted": boxed_answer,
        "score": score,
        "correct": score >= 1.0,
        "response": response_text,
        "tool_calls": example_result.tool_call_count,
        "tool_latencies_ms": [],
        "input_tokens": example_result.total_input_tokens,
        "output_tokens": example_result.total_output_tokens,
        "latency_ms": example_result.e2e_latency_s * 1000.0,
        "ttft_ms": avg_ttft_ms,
        "tpot_ms_avg": avg_tpot_ms,
        "tpot_ms_p99": 0.0,
        "errors": list(example_result.errors),
    }


async def solve_one_task(
    task: Any,
    session: aiohttp.ClientSession,
    model: str,
    base_url: str,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    startup_timeout: float,
    exec_timeout: float,
    preload: str,
    auto_print_last_expr: bool,
) -> Dict[str, Any]:
    backend = AsyncMathPythonBackend(
        startup_timeout=startup_timeout,
        exec_timeout=exec_timeout,
        preload=preload,
        auto_print_last_expr=auto_print_last_expr,
    )
    task_config = task.eval_config or {}
    await backend.setup(task_config)

    unified_task = UnifiedTask(
        task_id=task.id,
        task_name=task.name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *task.messages,
        ],
        eval_config=task_config,
    )

    try:
        tools = await backend.list_tools()
        with TemporaryFile(mode="w+") as request_details_file:
            example_result = await run_single_example(
                session=session,
                base_url=base_url,
                api_key="dummy",
                model=model,
                task=unified_task,
                tools=tools,
                backend=backend,
                max_turns=max_turns,
                max_tokens=max_tokens,
                temperature=temperature,
                openrouter_provider="",
                example_index=0,
                request_details_file=request_details_file,
                use_streaming=True,
                traj_dir=None,
            )
    finally:
        await backend.teardown()

    return _build_result_dict(task, example_result)


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


async def async_main(args: argparse.Namespace) -> List[Dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=1)

    gemini_judge = GeminiEquivalenceJudge(model_name=args.gemini_model)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = load_benchmark("imo_answerbench", num_tasks=args.num_tasks, seed=args.seed)
        print(f"Loaded {len(tasks)} IMO AnswerBench tasks")

        results: List[Dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            result = await solve_one_task(
                task=task,
                session=session,
                model=args.model,
                base_url=f"http://127.0.0.1:{args.port}/v1",
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                startup_timeout=args.startup_timeout,
                exec_timeout=args.exec_timeout,
                preload=args.preload,
                auto_print_last_expr=args.auto_print_last_expr,
            )

            result = await apply_gemini_judgment(result, gemini_judge, max_retries=5) # 5 max retries, unless move off free API key
            print(f"[GEMINI RESPONSE (line 607 run_imo_answerbench2)]: {result["gemini_judge_response"]}")

            results.append(result)
            print_task_result(index, len(tasks), result)
        return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="gpt-oss")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=128)
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
    parser.add_argument("--gemini-model", type=str, default="gemini-3.1-flash-lite-preview")

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
    try:
        infra.start()
        results = asyncio.run(async_main(args))
    finally:
        infra.stop()

    print_summary(results, time.time() - t0)


if __name__ == "__main__":
    main()
