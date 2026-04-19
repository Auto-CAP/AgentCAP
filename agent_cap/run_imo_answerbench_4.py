import argparse
import asyncio
import contextlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
import aiohttp
from google import genai
from math_verify import parse, verify
from openai import OpenAI

from agent_cap.benchmarks import load_benchmark
from agent_cap.backends.math_python_backend import MathPythonBackend
from agent_cap.runner.unified_runner import collect_hardware_info
from openai_harmony import HarmonyEncodingName, load_harmony_encoding, Conversation, Role, Message



SYSTEM_PROMPT = """You are an elite mathematical problem solver with expertise at the International Mathematical Olympiad (IMO) level.

# Output Format
- Provide a brief summary of the solution.
- Then state the final mathematical answer clearly.
- Put the final answer inside \\boxed{...}.
- The final answer may be an integer, fraction, expression, tuple, sequence, set, or other mathematical object, depending on the problem.
- Do not put anything except the final answer inside the final \\boxed{...}.
"""

JUDGE_PROMPT = """You are a strict mathematical answer equivalence judge.

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


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _find_config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        if key in config:
            return config[key]
        for value in config.values():
            found = _find_config_value(value, key)
            if found is not None:
                return found
    elif isinstance(config, list):
        for item in config:
            found = _find_config_value(item, key)
            if found is not None:
                return found
    return None


def infer_model_precision(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "unknown"

    quant_method = _find_config_value(model_config, "quant_method")
    if quant_method is not None:
        quant_method_str = str(quant_method)
        if re.search(r"\d", quant_method_str):
            return quant_method_str

    dtype = _find_config_value(model_config, "dtype")
    if dtype is not None:
        return str(dtype)

    return "unknown"


def initialize_output_files(args: argparse.Namespace) -> Dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    hw_info = collect_hardware_info()

    model_name = Path(args.model_path).name
    dataset_name = "imo_answerbench"
    gpu_shortform = str(hw_info.get("gpu_type", "unknown")).replace(" ", "-")
    number_of_gpus = int(hw_info.get("num_gpus", 0))

    results_dir = (
        Path("/llm-cache-pvc/outputs/TEAS_Development_Results_Private/agentic_results/eidf/vllm")
        / model_name
        / dataset_name
        / f"{gpu_shortform}-x-{number_of_gpus}"
        / "batch-size-default"
        / timestamp
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    detailed_results_path = results_dir / f"detailed-results_imo-answerbench_{timestamp}.jsonl"
    metadata_path = results_dir / f"metadata_imo-answerbench_{timestamp}.json"
    metrics_path = results_dir / f"metrics_imo-answerbench_{timestamp}.json"
    output_data_path = results_dir / f"output-data_imo-answerbench_{timestamp}.jsonl"

    detailed_results_path.touch()
    output_data_path.touch()

    metadata = {
        "hardware": hw_info,
        "model_config": {
            "model_name": _env_str("MODEL_NAME_FOR_METADATA", args.model_path),
            "precision": infer_model_precision(args.model_path),
        },
        "system_environment": {
            "inference_engine": _env_str("INFERENCE_ENGINE", "vllm"),
            "is_local": _env_bool("IS_LOCAL", True),
            "dataset": dataset_name,
            "num_examples": args.num_tasks,
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timestamp": timestamp,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "dataset": dataset_name,
                "num_examples": args.num_tasks,
                "status": "initialized",
            },
            f,
            indent=4,
        )

    print(f"Created results directory:      {results_dir}")
    print(f"Created detailed results file: {detailed_results_path}")
    print(f"Created metadata file:         {metadata_path}")
    print(f"Created metrics file:          {metrics_path}")
    print(f"Created output data file:      {output_data_path}")

    return {
        "timestamp": timestamp,
        "results_dir": str(results_dir),
        "detailed_results_path": str(detailed_results_path),
        "metadata_path": str(metadata_path),
        "metrics_path": str(metrics_path),
        "output_data_path": str(output_data_path),
    }


def _safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_sum(values: List[float]) -> float:
    return float(sum(values)) if values else 0.0


def _p99(values: List[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=100, method="inclusive")[98])


def write_metrics_file(
    results: List[Dict[str, Any]],
    wall_time_s: float,
    output_paths: Dict[str, str],
    args: argparse.Namespace,
) -> None:
    total_examples = len(results)

    latencies_s = [float(r["latency_ms"]) / 1000.0 for r in results]
    ttft_s = [float(r["ttft_ms"]) / 1000.0 for r in results]
    tpot_s = [float(r["tpot_ms_avg"]) / 1000.0 for r in results]

    input_tokens_list = [int(r["input_tokens"]) for r in results]
    output_tokens_list = [int(r["output_tokens"]) for r in results]
    tool_calls_list = [int(r["tool_calls"]) for r in results]

    total_input_tokens = int(sum(input_tokens_list))
    total_output_tokens = int(sum(output_tokens_list))
    total_tool_calls = int(sum(tool_calls_list))

    num_requests_list = [1 for _ in results]
    total_requests = int(sum(num_requests_list))

    input_tokens_per_request = []
    output_tokens_per_request = []
    max_input_tokens_per_request_list = []

    for r in results:
        reqs = 1
        total_in = int(r["input_tokens"])
        total_out = int(r["output_tokens"])

        input_tokens_per_request.append(total_in / reqs)
        output_tokens_per_request.append(total_out / reqs)
        max_input_tokens_per_request_list.append(float(total_in))

    decode_time_s_list = [
        (float(r["tpot_ms_avg"]) / 1000.0) * int(r["output_tokens"])
        for r in results
    ]
    total_decode_time_s = float(sum(decode_time_s_list))

    acc = (
        float(sum(float(r["score"]) for r in results)) / total_examples
        if total_examples > 0
        else 0.0
    )

    metrics = {
        "performance": {
            "e2e_s": float(wall_time_s),
            "avg_e2e_latency_s": _safe_mean(latencies_s),
            "p50_e2e_latency_s": float(statistics.median(latencies_s)) if latencies_s else 0.0,
            "p99_e2e_latency_s": _p99(latencies_s),
            "examples_per_second": (float(total_examples) / wall_time_s) if wall_time_s > 0 else 0.0,
            "ttft": _safe_mean(ttft_s),
            "p99_ttft": _p99(ttft_s),
            "tpot": _safe_mean(tpot_s),
            "p99_tpot": _p99(tpot_s),
            "decode_time_s": total_decode_time_s,
            "p99_decode_time_s": _p99(decode_time_s_list),
            "output_throughput_tok_s": (float(total_output_tokens) / total_decode_time_s)
            if total_decode_time_s > 0
            else 0.0,
        },
        "agentic": {
            "avg_total_input_tokens": _safe_mean([float(x) for x in input_tokens_list]),
            "avg_total_output_tokens": _safe_mean([float(x) for x in output_tokens_list]),
            "avg_tool_call_count": _safe_mean([float(x) for x in tool_calls_list]),
            "avg_num_requests": _safe_mean([float(x) for x in num_requests_list]),
            "avg_input_tokens_per_request": _safe_mean(input_tokens_per_request),
            "avg_output_tokens_per_request": _safe_mean(output_tokens_per_request),
            "avg_max_input_tokens_per_request": _safe_mean(max_input_tokens_per_request_list),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_tokens": 0,
            "avg_cache_hit_rate": 0.0,
            "total_requests": total_requests,
            "total_tool_calls": total_tool_calls,
        },
        "quality": {
            "acc": acc,
            "claim_coverage": "",
            "eval_judge": args.judge_model,
        },
        "hardware": {
            "gpu_type": _env_str("GPU_TYPE", "unknown"),
            "num_gpus": _env_int("NUM_GPUS", args.tensor_parallel_size),
            "avg_gpu_utilization_pct": "",
            "peak_gpu_memory_used_mb": "",
            "avg_cpu_utilization_pct": "",
        },
    }

    metrics_path = output_paths["metrics_path"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Wrote metrics file: {metrics_path}")

def append_output_data_row(
    result: Dict[str, Any],
    index: int,
    output_data_path: str,
) -> None:
    row = {
        "index": index - 1,
        "task_id": result["task_id"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "tool_call_count": result["tool_calls"],
        "num_requests": result["num_requests"],
        "e2e_latency_s": float(result["latency_ms"]) / 1000.0,
        "output_text": result["response"],
        "errors": result["errors"],
        "eval_passed": result["judge_equivalent"],
        "eval_score": result["score"],
        "eval_details": result["judge_response"],
    }

    with open(output_data_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def append_detailed_result_rows(
    detailed_rows: List[Dict[str, Any]],
    detailed_results_path: str,
) -> None:
    with open(detailed_results_path, "a", encoding="utf-8") as f:
        for row in detailed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")




def _task_message_to_harmony(msg: Dict[str, Any]) -> Message:
    role = msg["role"]
    content = msg["content"]

    if role == "system":
        return Message.from_role_and_content(Role.SYSTEM, content)
    if role == "user":
        return Message.from_role_and_content(Role.USER, content)
    if role == "assistant":
        return Message.from_role_and_content(Role.ASSISTANT, content)

    raise ValueError(f"Unsupported benchmark message role: {role}")


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


class SyncMathPythonBackend:
    """
    Thin synchronous wrapper around MathPythonBackend so the custom Harmony loop
    can call it directly in the same style as your AIMO3 notebook.
    """

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

    def setup(self, task_config: Dict[str, Any]) -> bool:
        return self._backend.setup(task_config)

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._backend.get_tool_definitions()

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        result = self._backend.execute(name, "call", arguments)
        if result.success:
            return result.output
        raise RuntimeError(result.output)

    def teardown(self) -> None:
        self._backend.teardown()


class OpenRouterEquivalenceJudge:
    def __init__(
        self,
        model_name: str = "openrouter/elephant-alpha",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY or pass api_key explicitly."
            )

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

    def judge_equivalence(
        self,
        predicted: Optional[str],
        expected: Optional[str],
    ) -> Dict[str, Any]:
        if predicted is None or expected is None:
            return {
                "equivalent": False,
                "raw_response": "Missing predicted or expected value.",
                "status_code": None,
                "response_json": None,
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

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # "HTTP-Referer": "<YOUR_SITE_URL>",
                # "X-OpenRouter-Title": "<YOUR_SITE_NAME>",
            },
            data=json.dumps(
                {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "temperature": 0.0,
                }
            ),
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()
        text = data["choices"][0]["message"]["content"]
        equivalent = self._extract_json_bool(text)

        return {
            "equivalent": bool(equivalent) if equivalent is not None else False,
            "raw_response": text,
            "status_code": response.status_code,
            "response_json": data,
        }

    async def judge_equivalence_async(
        self,
        predicted: Optional[str],
        expected: Optional[str],
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self.judge_equivalence, predicted, expected)


async def apply_llm_judgment(
    result: Dict[str, Any],
    judge,
    max_retries: int = 5,
) -> Dict[str, Any]:
    predicted = result.get("predicted")
    expected = result.get("expected")

    result["rule_score"] = result["score"]
    result["rule_correct"] = result["correct"]

    last_raw_response = None
    judge_equivalent = False
    judge_attempts = 0

    for attempt in range(1, max_retries + 1):
        judge_attempts = attempt
        try:
            judge_eval = await judge.judge_equivalence_async(predicted, expected)
            last_raw_response = judge_eval.get("raw_response")
            judge_equivalent = bool(judge_eval.get("equivalent", False))

            if judge_equivalent:
                break
        except Exception as exc:
            last_raw_response = f"Judge attempt {attempt} failed: {exc}"

    result["judge_equivalent"] = judge_equivalent
    result["judge_response"] = last_raw_response
    result["judge_attempts"] = judge_attempts

    result["score"] = 1.0 if judge_equivalent else 0.0
    result["correct"] = judge_equivalent

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
            "--trust_remote_code",
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
                with open("vllm_server.log", "r", encoding="utf-8", errors="ignore") as log_file:
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

def compute_score(
    solution_str: str,
    ground_truth: str,
    judge: "OpenRouterEquivalenceJudge",
) -> tuple[float, Optional[str], Dict[str, Any]]:
    """
    Extract predicted answer from solution_str, then ask the OpenRouter judge
    whether it is equivalent to ground_truth.

    Returns:
        score: 1.0 if equivalent else 0.0
        predicted: extracted predicted answer string, or None
        judge_result: raw judge metadata dict
    """
    predicted = _scan_for_answer(solution_str)

    if predicted is None or ground_truth is None:
        return 0.0, predicted, {
            "equivalent": False,
            "raw_response": "Missing predicted or expected value.",
            "status_code": None,
            "response_json": None,
        }

    try:
        judge_result = judge.judge_equivalence(predicted, ground_truth)
        score = 1.0 if judge_result.get("equivalent", False) else 0.0
        return score, predicted, judge_result
    except Exception as exc:
        return 0.0, predicted, {
            "equivalent": False,
            "raw_response": f"Judge failed: {type(exc).__name__}: {exc}",
            "status_code": None,
            "response_json": None,
        }
    
def compute_score_math_verify(solution_str: str, ground_truth: str) -> float:
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


def _scan_for_answer(text: str) -> Optional[str]:
    """
    Your AIMO3-style answer scan, but returning string content rather than forcing int,
    because IMO AnswerBench answers can be non-integers.
    """
    boxed_content = extract_last_boxed_content(text)
    if boxed_content is not None:
        return boxed_content.strip()

    matches = re.findall(r'final\s+answer\s+is\s*(.+)', text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    bold_matches = re.findall(r'(?:\*\*|__)\s*(.+?)\s*(?:\*\*|__)', text)
    if bold_matches:
        return bold_matches[-1].strip()

    return None


def _build_tool_response_message(tool_name: str, tool_output: str) -> Dict[str, Any]:
    """
    OpenAI-style dict message. This is only used to bootstrap the conversation.
    After the first assistant completion, the Harmony parser returns native message objects.
    """
    return {
        "role": "tool",
        "name": tool_name,
        "content": [{"type": "text", "text": tool_output}],
    }


def _extract_tool_call(last_message: Any) -> tuple[str, Dict[str, Any]]:
    """
    Best-effort extraction of tool name + args from a Harmony assistant message.
    You may need to tweak this depending on your openai_harmony object structure.
    """
    recipient = getattr(last_message, "recipient", None)
    if recipient is None:
        raise ValueError("Assistant tool-call message missing recipient.")

    # Common case: structured arguments already parsed.
    if hasattr(last_message, "arguments"):
        args = getattr(last_message, "arguments")
        if isinstance(args, dict):
            return recipient, args

    # Fallback: content[0].text contains JSON arguments.
    content = getattr(last_message, "content", None) or []
    if content:
        maybe_text = getattr(content[0], "text", None)
        if maybe_text:
            try:
                return recipient, json.loads(maybe_text)
            except Exception:
                pass

    # Another fallback: raw text body
    raw_text = str(last_message)
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        try:
            return recipient, json.loads(json_match.group(0))
        except Exception:
            pass

    raise ValueError(f"Could not extract tool arguments from message: {last_message!r}")


def _append_tool_response_to_conversation(
    conversation: Any,
    tool_output: str,
    request_message: Any,
) -> None:
    tool_message = Message(
        author="tool",
        channel=getattr(request_message, "channel", None),
        content=[{"type": "text", "text": tool_output}],
    )

    if hasattr(conversation, "messages") and isinstance(conversation.messages, list):
        conversation.messages.append(tool_message)
        return

    raise TypeError("Could not append tool response to Harmony conversation.")


def run_harmony_attempt(
    *,
    task: Any,
    example_index: int,
    client: OpenAI,
    model: str,
    encoding: Any,
    stop_token_ids: List[int],
    max_turns: int,
    max_tokens: int,
    temperature: float,
    startup_timeout: float,
    exec_timeout: float,
    preload: str,
    auto_print_last_expr: bool,
    seed: int,
    judge: OpenRouterEquivalenceJudge,
) -> Dict[str, Any]:
    """
    Single-attempt loop:
    - render conversation with Harmony encoding
    - call /completions with token ids
    - parse assistant messages from completion tokens
    - intercept python tool calls
    - continue until final channel or turn limit
    """

    backend = SyncMathPythonBackend(
        startup_timeout=startup_timeout,
        exec_timeout=exec_timeout,
        preload=preload,
        auto_print_last_expr=auto_print_last_expr,
    )

    t0 = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    total_decode_time_s = 0.0
    total_prefill_time_s = 0.0
    tool_call_count = 0
    errors: List[str] = []
    response_text = ""
    final_answer = None
    num_requests = 0
    detailed_rows: List[Dict[str, Any]] = []
    backend.setup(task.eval_config or {})

    try:
        initial_messages = [
            Message.from_role_and_content(Role.SYSTEM, SYSTEM_PROMPT),
            *[_task_message_to_harmony(m) for m in task.messages],
        ]

        conversation = Conversation.from_messages(initial_messages)

        for turn_idx in range(max_turns):
            prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
            total_input_tokens += len(prompt_ids)

            remaining_ctx = min(max_tokens, 131072) - len(prompt_ids)
            if remaining_ctx <= 0:
                errors.append("No remaining token budget.")
                break

            stream_start = time.time()
            first_token_time = None
            token_buffer: List[int] = []
            text_chunks: List[str] = []

            extra = {
                "stop_token_ids": stop_token_ids,
                "return_token_ids": True,
            }

            num_requests += 1

            stream = client.completions.create(
                model=model,
                prompt=prompt_ids,
                max_tokens=remaining_ctx,
                temperature=temperature,
                seed=seed,
                stream=True,
                extra_body=extra,
            )

            cached_tokens_this_request = 0

            try:
                for chunk in stream:
                    choice = chunk.choices[0]

                    if first_token_time is None:
                        first_token_time = time.time()

                    new_tokens = getattr(choice, "token_ids", None) or []
                    new_text = getattr(choice, "text", "") or ""

                    if new_tokens:
                        token_buffer.extend(new_tokens)
                        total_output_tokens += len(new_tokens)

                    if new_text:
                        text_chunks.append(new_text)

                    # Best-effort extraction of cached tokens if vLLM/OpenAI-compatible
                    # usage metadata is present on the streamed chunk.
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                        if prompt_tokens_details is not None:
                            maybe_cached = getattr(prompt_tokens_details, "cached_tokens", None)
                            if maybe_cached is not None:
                                cached_tokens_this_request = int(maybe_cached)

                stream_end = time.time()

            finally:
                with contextlib.suppress(Exception):
                    stream.close()

            if first_token_time is None:
                total_prefill_time_s += 0.0
                total_decode_time_s += 0.0
                errors.append("Model returned no streamed tokens.")
                break

            total_prefill_time_s += max(0.0, first_token_time - stream_start)
            total_decode_time_s += max(0.0, stream_end - first_token_time)

            input_tokens_this_request = len(prompt_ids)
            output_tokens_this_request = len(token_buffer)
            prefill_time_s_this_request = max(0.0, first_token_time - stream_start)
            decode_time_s_this_request = max(0.0, stream_end - first_token_time)
            tpot_s_this_request = (
                decode_time_s_this_request / output_tokens_this_request
                if output_tokens_this_request > 0
                else 0.0
            )
            output_throughput_tok_s_this_request = (
                output_tokens_this_request / decode_time_s_this_request
                if decode_time_s_this_request > 0
                else 0.0
            )

            if not token_buffer:
                errors.append("Empty token buffer.")
                break

            parsed_messages = encoding.parse_messages_from_completion_tokens(
                token_buffer,
                Role.ASSISTANT,
            )

            if not parsed_messages:
                response_text = "".join(text_chunks)
                final_answer = _scan_for_answer(response_text)
                break

            if hasattr(conversation, "messages"):
                conversation.messages.extend(parsed_messages)

            last_message = parsed_messages[-1]

            if getattr(last_message, "channel", None) == "final":
                content = getattr(last_message, "content", None) or []
                if content and getattr(content[0], "text", None) is not None:
                    response_text = content[0].text
                else:
                    response_text = "".join(text_chunks)

                final_answer = _scan_for_answer(response_text)
                break

            recipient = getattr(last_message, "recipient", None)
            has_tool_calls_this_request = recipient is not None and "python" in str(recipient)
            num_tool_calls_this_request = 1 if has_tool_calls_this_request else 0

            detailed_rows.append(
                {
                    "example_index": example_index,
                    "request_index": num_requests - 1,
                    "input_tokens": input_tokens_this_request,
                    "output_tokens": output_tokens_this_request,
                    "cached_tokens": cached_tokens_this_request,
                    "prefill_time_s": prefill_time_s_this_request,
                    "decode_time_s": decode_time_s_this_request,
                    "tpot_s": tpot_s_this_request,
                    "output_throughput_tok_s": output_throughput_tok_s_this_request,
                    "has_tool_calls": has_tool_calls_this_request,
                    "num_tool_calls": num_tool_calls_this_request,
                }
            )

            if recipient is not None and "python" in str(recipient):
                try:
                    tool_name, tool_args = _extract_tool_call(last_message)
                    tool_call_count += 1
                    tool_output = backend.execute_tool(tool_name, tool_args)
                    _append_tool_response_to_conversation(
                        conversation,
                        tool_output,
                        last_message,
                    )
                except Exception as exc:
                    errors.append(f"Tool execution failed: {type(exc).__name__}: {exc}")
                    break
            else:
                content = getattr(last_message, "content", None) or []
                if content and getattr(content[0], "text", None) is not None:
                    response_text = content[0].text
                    final_answer = _scan_for_answer(response_text)
                    if final_answer is not None:
                        break

        if not response_text:
            response_text = ""

        expected = (task.eval_config or {}).get("expected")
        score, extracted_predicted, judge_result = compute_score(
            response_text,
            expected,
            judge,
        )

        avg_ttft_ms = 1000.0 * total_prefill_time_s
        avg_tpot_ms = (
            1000.0 * total_decode_time_s / total_output_tokens
            if total_output_tokens > 0
            else 0.0
        )

        return {
            "task_id": task.id,
            "task_name": task.name,
            "category": task.category,
            "expected": expected,
            "predicted": extracted_predicted,
            "score": score,
            "correct": score >= 1.0,
            "response": response_text,
            "tool_calls": tool_call_count,
            "num_requests": num_requests,
            "tool_latencies_ms": [],
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "latency_ms": (time.time() - t0) * 1000.0,
            "ttft_ms": avg_ttft_ms,
            "tpot_ms_avg": avg_tpot_ms,
            "tpot_ms_p99": 0.0,
            "errors": errors,
            "judge_equivalent": judge_result.get("equivalent", False),
            "judge_response": judge_result.get("raw_response"),
            "judge_status_code": judge_result.get("status_code"),
            "judge_attempts": 1,
            "detailed_rows": detailed_rows,
        }
    
    finally:
        with contextlib.suppress(Exception):
            backend.teardown()


async def solve_one_task(
    task: Any,
    example_index: int,
    model: str,
    base_url: str,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    startup_timeout: float,
    exec_timeout: float,
    preload: str,
    auto_print_last_expr: bool,
    seed: int,
    judge: OpenRouterEquivalenceJudge,
) -> Dict[str, Any]:
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    client = OpenAI(
        base_url=base_url,
        api_key="dummy",
        timeout=600,
    )

    return await asyncio.to_thread(
        run_harmony_attempt,
        task=task,
        example_index=example_index,
        client=client,
        model=model,
        encoding=encoding,
        stop_token_ids=stop_token_ids,
        max_turns=max_turns,
        max_tokens=max_tokens,
        temperature=temperature,
        startup_timeout=startup_timeout,
        exec_timeout=exec_timeout,
        preload=preload,
        auto_print_last_expr=auto_print_last_expr,
        seed=seed,
        judge=judge,
    )


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


async def async_main(
    args: argparse.Namespace,
    output_paths: Dict[str, str],
) -> List[Dict[str, Any]]:
    judge = OpenRouterEquivalenceJudge(model_name=args.judge_model)

    tasks = load_benchmark("imo_answerbench", num_tasks=args.num_tasks, seed=args.seed)
    print(f"Loaded {len(tasks)} IMO AnswerBench tasks")

    results: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        result = await solve_one_task(
            task=task,
            example_index=index - 1,
            model=args.model,
            base_url=f"http://127.0.0.1:{args.port}/v1",
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            startup_timeout=args.startup_timeout,
            exec_timeout=args.exec_timeout,
            preload=args.preload,
            auto_print_last_expr=args.auto_print_last_expr,
            seed=args.seed + index,
            judge=judge,
        )
        print(f'[JUDGE RESPONSE]: {result["judge_response"]}')

        results.append(result)
        append_detailed_result_rows(result.get("detailed_rows", []), output_paths["detailed_results_path"])
        append_output_data_row(result, index, output_paths["output_data_path"])
        print_task_result(index, len(tasks), result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="gpt-oss")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=131072)
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
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--context-tokens", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--server-timeout", type=int, default=3600)
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument("--judge-model", type=str, default="openrouter/elephant-alpha")

    args = parser.parse_args()
    output_paths = initialize_output_files(args)
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
        results = asyncio.run(async_main(args, output_paths))
    finally:
        infra.stop()

    wall_time_s = time.time() - t0
    print_summary(results, wall_time_s)
    write_metrics_file(results, wall_time_s, output_paths, args)


if __name__ == "__main__":
    main()
