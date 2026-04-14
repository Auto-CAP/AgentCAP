import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from math_verify import parse, verify
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

from agent_cap.benchmarks import load_benchmark
from agent_cap.runner.unified_runner import (
    UnifiedConfig,
    UnifiedTask,
    compute_aggregated_metrics,
    run_experiment,
)


SYSTEM_PROMPT = """You are an elite mathematical problem solver with expertise at the International Mathematical Olympiad (IMO) level.

# Output Format
- Provide a brief summary of the solution.
- Then state the final mathematical answer clearly.
- Put the final answer inside \\boxed{...}.
- The final answer may be an integer, fraction, expression, tuple, sequence, set, or other mathematical object, depending on the problem.
- Do not put anything except the final answer inside the final \\boxed{...}.
"""


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
                with open("vllm_server.log", "r", encoding="utf-8") as log_file:
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


class GeminiEquivalenceJudge:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client()

    def _extract_json_bool(self, text: str) -> Optional[bool]:
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "equivalent" in data:
                return bool(data["equivalent"])
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict) and "equivalent" in data:
                    return bool(data["equivalent"])
            except Exception:
                pass

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
    result["score"] = 1.0 if gemini_equivalent else 0.0
    result["correct"] = gemini_equivalent
    return result


def build_unified_tasks(num_tasks: int, seed: int) -> List[UnifiedTask]:
    tasks = load_benchmark("imo_answerbench", num_tasks=num_tasks, seed=seed)
    unified_tasks: List[UnifiedTask] = []
    for task in tasks:
        unified_tasks.append(
            UnifiedTask(
                task_id=task.id,
                task_name=task.name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *task.messages,
                ],
                eval_config=task.eval_config,
            )
        )
    return unified_tasks


def build_result_dict(task: UnifiedTask, example_result: Any) -> Dict[str, Any]:
    response_text = example_result.output_text or ""
    expected = (task.eval_config or {}).get("expected")
    predicted = extract_last_boxed_content(response_text)
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
        "task_id": task.task_id,
        "task_name": task.task_name,
        "expected": expected,
        "predicted": predicted,
        "score": score,
        "correct": score >= 1.0,
        "response": response_text,
        "tool_calls": example_result.tool_call_count,
        "input_tokens": example_result.total_input_tokens,
        "output_tokens": example_result.total_output_tokens,
        "latency_ms": example_result.e2e_latency_s * 1000.0,
        "ttft_ms": avg_ttft_ms,
        "tpot_ms_avg": avg_tpot_ms,
        "tpot_ms_p99": 0.0,
        "num_requests": example_result.num_requests,
        "max_input_tokens_per_request": example_result.max_input_tokens_per_request,
        "errors": list(example_result.errors),
    }


def update_output_data_file(output_data_path: Path, judged_results: List[Dict[str, Any]]) -> None:
    with output_data_path.open("w", encoding="utf-8") as f:
        for index, result in enumerate(judged_results):
            row = {
                "index": index,
                "task_id": result["task_id"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "tool_call_count": result["tool_calls"],
                "num_requests": result["num_requests"],
                "e2e_latency_s": result["latency_ms"] / 1000.0,
                "output_text": result["response"],
                "predicted": result["predicted"],
                "expected": result["expected"],
                "score": result["score"],
                "correct": result["correct"],
                "rule_score": result.get("rule_score"),
                "rule_correct": result.get("rule_correct"),
                "gemini_equivalent": result.get("gemini_equivalent"),
                "gemini_attempts": result.get("gemini_attempts"),
                "gemini_judge_response": result.get("gemini_judge_response"),
                "errors": result["errors"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_metrics_file(
    metrics_path: Path,
    base_metrics: Dict[str, Any],
    judged_results: List[Dict[str, Any]],
    gemini_model: str,
) -> None:
    total_examples = len(judged_results)
    acc = (
        float(sum(float(r["score"]) for r in judged_results)) / total_examples
        if total_examples > 0
        else 0.0
    )
    claim_coverage = (
        float(sum(1 for r in judged_results if r.get("predicted") is not None)) / total_examples
        if total_examples > 0
        else 0.0
    )

    metrics = dict(base_metrics)
    metrics["quality"] = {
        "acc": acc,
        "claim_coverage": claim_coverage,
        "eval_judge": f"google/{gemini_model}",
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


def print_summary(results: List[Dict[str, Any]], wall_time_s: float) -> None:
    total = len(results)
    total_score = sum(float(r["score"]) for r in results)
    accuracy = 100.0 * total_score / total if total else 0.0

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"Tasks solved:        {total}")
    print(f"Total score:         {total_score:.1f}")
    print(f"Average score:       {accuracy:.1f}%")
    print(f"Wall time:           {wall_time_s:.2f}s")


async def async_main(args: argparse.Namespace) -> None:
    unified_tasks = build_unified_tasks(args.num_tasks, args.seed)
    config = UnifiedConfig(
        model_name=args.model,
        serving_engine="vllm",
        base_url=f"http://127.0.0.1:{args.port}/v1",
        dataset="imo_answerbench",
        mcp_server_url="",
        backend="math-python",
        swebench_runtime="docker",
        api_key="dummy",
        api_provider="",
        openrouter_provider_pin="",
        openrouter_provider="",
        is_local=True,
        precision=args.dtype,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enabled_tools=["python"],
        output_root=Path(args.output_root),
        use_streaming=True,
    )

    run_result = await run_experiment(config, unified_tasks)

    judged_results: List[Dict[str, Any]] = []
    gemini_judge = GeminiEquivalenceJudge(model_name=args.gemini_model)

    for task, example_result in zip(unified_tasks, run_result.example_results):
        result = build_result_dict(task, example_result)
        result = await apply_gemini_judgment(result, gemini_judge, max_retries=5)
        judged_results.append(result)

    output_dir = run_result.output_dir
    suffix = run_result.suffix
    metrics_path = output_dir / f"metrics_{suffix}.json"
    output_data_path = output_dir / f"output_data_{suffix}.jsonl"

    update_output_data_file(output_data_path, judged_results)
    update_metrics_file(metrics_path, run_result.metrics, judged_results, args.gemini_model)

    print(f"\nDone. Output directory: {output_dir}")
    print(f"  metadata:         metadata_{suffix}.json")
    print(f"  metrics:          metrics_{suffix}.json")
    print(f"  detailed_results: detailed_results_{suffix}.jsonl")
    print(f"  output_data:      output_data_{suffix}.jsonl")
    print_summary(judged_results, float(run_result.metrics["performance"]["e2e_s"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-oss")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)

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
    parser.add_argument("--output-root", type=str, default="results")

    args = parser.parse_args()

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
        asyncio.run(async_main(args))
    finally:
        infra.stop()


if __name__ == "__main__":
    main()
