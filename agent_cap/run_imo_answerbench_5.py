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
import uuid
# from google import genai
from math_verify import parse, verify
from openai import OpenAI
from huggingface_hub import snapshot_download

from agent_cap.benchmarks import load_benchmark
from agent_cap.backends.math_python_backend import MathPythonBackend
from agent_cap.runner.unified_runner import collect_hardware_info
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Role,
    Message,
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    TextContent,
)



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

def collect_hardware_info_rocm_fallback() -> Dict[str, Any]:
    try:
        hw_info = collect_hardware_info()
        if hw_info.get("gpu_type") not in (None, "", "unknown"):
            return hw_info
    except Exception:
        hw_info = {}

    rocr_visible = os.getenv("ROCR_VISIBLE_DEVICES", "")
    hip_visible = os.getenv("HIP_VISIBLE_DEVICES", "")

    gpu_type = "unknown"
    num_gpus = 0

    try:
        proc = subprocess.run(
            ["rocminfo"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        text = proc.stdout

        # Common on MI350/MI355 systems: gfx950.
        if "gfx950" in text:
            gpu_type = "AMD Instinct MI35x / gfx950"
        elif "gfx942" in text:
            gpu_type = "AMD Instinct MI300/MI325 / gfx942"
        elif "AMD Instinct" in text:
            gpu_type = "AMD Instinct"

    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["python", "-c", "import torch; print(torch.cuda.device_count())"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        num_gpus = int(proc.stdout.strip().splitlines()[-1])
    except Exception:
        if rocr_visible:
            num_gpus = len([x for x in rocr_visible.split(",") if x.strip()])
        else:
            num_gpus = 0

    hw_info.update(
        {
            "gpu_type": hw_info.get("gpu_type") or gpu_type,
            "num_gpus": hw_info.get("num_gpus") or num_gpus,
            "rocr_visible_devices": rocr_visible,
            "hip_visible_devices": hip_visible,
        }
    )

    return hw_info


def initialize_output_files(args: argparse.Namespace) -> Dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print('collecting hardware info...', flush=True)
    hw_info = collect_hardware_info_rocm_fallback()
    print('successfully collected hardware info.', flush=True)
    model_name = Path(args.model_path).name
    dataset_name = "imo_answerbench"
    gpu_shortform = str(hw_info.get("gpu_type", "unknown")).replace(" ", "-")
    number_of_gpus = int(hw_info.get("num_gpus", 0))

    output_root = Path(
        _env_str(
            "OUTPUT_ROOT",
            "/workspace/outputs/TEAS_Development_Results_Private/agentic_results/amd/sglang",
        )
    )

    results_dir = (
        output_root
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
            "inference_engine": _env_str("INFERENCE_ENGINE", "sglang"),
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
    kv_cache_dtype: Optional[str]
    dtype: Optional[str]
    context_tokens: int
    batch_size: int
    mem_fraction_static: float
    tensor_parallel_size: int
    server_timeout: int
    preload_workers: int

    # SGLang-specific
    host: str = "127.0.0.1"
    log_level: str = "warning"
    chunked_prefill_size: Optional[int] = 16384
    cuda_graph_max_bs: Optional[int] = 8
    enable_torch_compile: bool = False
    allow_auto_truncate: bool = True
    top_p: float = 1.0

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

        # AIMO3-style: feed tool failure back to the model as tool output
        output = result.output or "Unknown tool error."
        if not str(output).startswith("[ERROR]"):
            output = f"[ERROR] {output}"
        return output

    def teardown(self) -> None:
        self._backend.teardown()


class OpenRouterEquivalenceJudge:
    def __init__(
        self,
        model_name: str = "openrouter/elephant-alpha",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 10,
        backoff_start_s: float = 20.0,
        backoff_cap_s: float = 60.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_start_s = backoff_start_s
        self.backoff_cap_s = backoff_cap_s

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

    def _compute_backoff_seconds(self, attempt: int) -> float:
        """
        Linear backoff:
        attempt=1 -> 20s
        attempt=2 -> 40s
        attempt>=3 -> 60s (cap)
        """
        return min(self.backoff_start_s * attempt, self.backoff_cap_s)

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

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
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

            except requests.exceptions.RequestException as exc:
                last_error = exc
                status_code = None
                response_json = None

                if getattr(exc, "response", None) is not None:
                    status_code = exc.response.status_code
                    try:
                        response_json = exc.response.json()
                    except Exception:
                        response_json = None

                if attempt < self.max_retries:
                    sleep_s = self._compute_backoff_seconds(attempt)
                    print(
                        f"[OpenRouterEquivalenceJudge] attempt {attempt}/{self.max_retries} failed: "
                        f"{type(exc).__name__}: {exc}. Retrying in {sleep_s:.1f}s...",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                    continue

                return {
                    "equivalent": False,
                    "raw_response": (
                        f"Judge failed after {self.max_retries} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "status_code": status_code,
                    "response_json": response_json,
                }

            except Exception as exc:
                last_error = exc

                if attempt < self.max_retries:
                    sleep_s = self._compute_backoff_seconds(attempt)
                    print(
                        f"[OpenRouterEquivalenceJudge] attempt {attempt}/{self.max_retries} failed: "
                        f"{type(exc).__name__}: {exc}. Retrying in {sleep_s:.1f}s...",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                    continue

                return {
                    "equivalent": False,
                    "raw_response": (
                        f"Judge failed after {self.max_retries} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "status_code": None,
                    "response_json": None,
                }

        return {
            "equivalent": False,
            "raw_response": (
                f"Judge failed after {self.max_retries} attempts: "
                f"{type(last_error).__name__}: {last_error}"
                if last_error is not None
                else "Judge failed for unknown reason."
            ),
            "status_code": None,
            "response_json": None,
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

def resolve_model_path(model_path: str, local_model_root: Optional[str] = None) -> str:
    """
    If model_path is an existing local directory, return it unchanged.
    Otherwise, treat it as a Hugging Face repo id and download it locally.

    Examples:
      - /workspace/models/unsloth/gpt-oss-120b -> use as-is if it exists
      - unsloth/gpt-oss-120b                  -> download to /workspace/models/unsloth/gpt-oss-120b
    """
    path_obj = Path(model_path)
    if path_obj.is_dir():
        resolved = str(path_obj.resolve())
        print(f"Using local model directory: {resolved}", flush=True)
        return resolved

    if local_model_root is None:
        local_model_root = os.getenv("HF_LOCAL_MODEL_ROOT", "/workspace/models")

    target_dir = Path(local_model_root) / model_path
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    print(
        f"Local model directory not found for '{model_path}'. "
        f"Treating it as a Hugging Face repo id and downloading to {target_dir}",
        flush=True,
    )

    resolved = snapshot_download(
        repo_id=model_path,
        local_dir=str(target_dir),
        token=hf_token,
        resume_download=True,
    )

    print(f"Downloaded model snapshot to: {resolved}", flush=True)
    return resolved

class SGLangInfraGPTOSS:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.port = cfg.port

        client_host = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
        self.native_base_url = f"http://{client_host}:{cfg.port}"
        self.openai_base_url = f"{self.native_base_url}/v1"

        self.server_process: Optional[subprocess.Popen] = None
        self.log_file = None

    def start(self) -> None:
        self._preload_model_weights()
        self.server_process = self._start_server()
        self._wait_for_server()

    def stop(self) -> None:
        if self.server_process is not None and self.server_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGINT)
                self.server_process.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                    self.server_process.wait(timeout=5)
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
            raise FileNotFoundError(
                f"Resolved model path does not exist or is not a directory: {self.cfg.model_path}"
            )

        print(f"Loading model weights from {self.cfg.model_path} into OS Page Cache...", flush=True)
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
            f"Processed {len(files_to_load)} files "
            f"({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n",
            flush=True,
        )

    def _start_server(self) -> subprocess.Popen:
        env = os.environ.copy()
        env["TRANSFORMERS_NO_TF"] = "1"
        env["TRANSFORMERS_NO_FLAX"] = "1"

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.cfg.model_path,
            "--host",
            self.cfg.host,
            "--port",
            str(self.cfg.port),
            "--log-level",
            self.cfg.log_level,
            "--served-model-name",
            self.cfg.served_model_name,
            "--context-length",
            str(self.cfg.context_tokens),
            "--mem-fraction-static",
            str(self.cfg.mem_fraction_static),
            "--tensor-parallel-size",
            str(self.cfg.tensor_parallel_size),
            "--random-seed",
            str(self.cfg.seed),
            "--trust-remote-code",
        ]

        if self.cfg.dtype:
            cmd += ["--dtype", self.cfg.dtype]

        if self.cfg.kv_cache_dtype:
            cmd += ["--kv-cache-dtype", self.cfg.kv_cache_dtype]

        if self.cfg.chunked_prefill_size is not None:
            cmd += ["--chunked-prefill-size", str(self.cfg.chunked_prefill_size)]

        if self.cfg.cuda_graph_max_bs is not None:
            cmd += ["--cuda-graph-max-bs", str(self.cfg.cuda_graph_max_bs)]

        if self.cfg.enable_torch_compile:
            cmd.append("--enable-torch-compile")

        if self.cfg.allow_auto_truncate:
            cmd.append("--allow-auto-truncate")

        self.log_file = open("sglang_server.log", "w", encoding="utf-8")

        print("Launching SGLang:")
        print(" ".join(cmd), flush=True)

        return subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    def _wait_for_server(self) -> None:
        print("Waiting for SGLang server...", flush=True)
        start_time = time.time()
        models_url = f"{self.openai_base_url}/models"

        for i in range(self.cfg.server_timeout):
            if i % 100 == 0:
                print(f"waiting for server to start: poll count={i}", flush=True)

            if self.server_process is None:
                raise RuntimeError("Server process was not created.")

            return_code = self.server_process.poll()
            if return_code is not None:
                if self.log_file is not None:
                    self.log_file.flush()
                with open("sglang_server.log", "r", encoding="utf-8", errors="ignore") as log_file:
                    logs = log_file.read()
                raise RuntimeError(
                    f"SGLang server died with code {return_code}. Full logs:\n{logs}\n"
                )

            try:
                resp = requests.get(models_url, timeout=5)
                if resp.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"SGLang server is ready (took {elapsed:.2f} seconds).\n", flush=True)
                    return
            except Exception:
                time.sleep(1)

        raise RuntimeError("SGLang server failed to start within timeout.")


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
    recipient = getattr(last_message, "recipient", None)
    if recipient is None:
        raise ValueError("Assistant tool-call message missing recipient.")

    content = getattr(last_message, "content", None) or []

    if hasattr(last_message, "arguments"):
        args = getattr(last_message, "arguments")
        if isinstance(args, dict):
            return str(recipient), args

    maybe_text = None
    if content and getattr(content[0], "text", None) is not None:
        maybe_text = content[0].text

    if isinstance(recipient, str) and "python" in recipient:
        if isinstance(maybe_text, str) and maybe_text.strip():
            return "python", {"code": maybe_text}

    if isinstance(maybe_text, str):
        try:
            parsed = json.loads(maybe_text)
            if isinstance(parsed, dict):
                return str(recipient), parsed
        except Exception:
            pass

    raw_text = str(last_message)
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                return str(recipient), parsed
        except Exception:
            pass

    raise ValueError(f"Could not extract tool arguments from message: {last_message!r}")


def _append_tool_response_to_conversation(
    conversation: Any,
    tool_name: str,
    tool_output: str,
    request_message: Any,
) -> None:
    tool_message = (
        Message(
            author=Author(role=Role.TOOL, name=tool_name),
            content=[TextContent(text=tool_output)],
        )
        .with_recipient("assistant")
    )

    channel = getattr(request_message, "channel", None)
    if channel:
        tool_message = tool_message.with_channel(channel)

    if hasattr(conversation, "messages") and isinstance(conversation.messages, list):
        conversation.messages.append(tool_message)
        return

    raise TypeError("Could not append tool response to Harmony conversation.")


def sglang_generate_with_ids(
    *,
    native_base_url: str,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_ids: List[int],
    timeout_s: float = 600.0,
) -> tuple[str, List[int], Dict[str, Any], float]:
    payload = {
        "input_ids": list(map(int, prompt_ids)),
        "rid": str(uuid.uuid4()),
        "sampling_params": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stop_token_ids": list(map(int, stop_token_ids)),

            # Important for Harmony parsing/debugging.
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "no_stop_trim": True,
        },
        "stream": False,
    }

    t0 = time.time()
    response = requests.post(
        f"{native_base_url.rstrip('/')}/generate",
        json=payload,
        timeout=timeout_s,
    )
    elapsed_s = time.time() - t0

    if response.status_code == 400:
        raise RuntimeError(
            "SGLang /generate returned 400.\n"
            f"Response text:\n{response.text}\n\n"
            f"len(input_ids)={len(prompt_ids)}\n"
            f"sampling_params={payload['sampling_params']}"
        )

    response.raise_for_status()
    data = response.json()

    text = data.get("text", "")
    output_ids = data.get("output_ids")
    meta_info = data.get("meta_info", {}) or {}

    if not output_ids:
        raise RuntimeError(
            f"SGLang /generate returned no output_ids. "
            f"keys={list(data.keys())}, meta_info={meta_info!r}"
        )

    output_ids = list(map(int, output_ids))

    # Defensive guard: some SGLang versions historically returned prompt+completion ids.
    # If that happens, strip the prompt prefix.
    if len(output_ids) > len(prompt_ids) and output_ids[: len(prompt_ids)] == prompt_ids:
        output_ids = output_ids[len(prompt_ids) :]

    return text, output_ids, meta_info, elapsed_s

def sglang_generate_with_ids_streaming_deprecated(
    *,
    native_base_url: str,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_ids: List[int],
    timeout_s: float = 600.0,
) -> tuple[str, List[int], Dict[str, Any], float, float]:
    """
    Stream from SGLang native /generate.

    Returns:
        final_text: accumulated decoded text from SGLang
        output_ids: generated token ids from the final/most recent chunk
        meta_info: final/most recent meta_info
        ttft_s: time to first streamed chunk
        elapsed_s: total request wall time
    """
    payload = {
        "input_ids": list(map(int, prompt_ids)),
        "rid": str(uuid.uuid4()),
        "sampling_params": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stop_token_ids": list(map(int, stop_token_ids)),

            # Important for Harmony parsing/debugging.
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "no_stop_trim": True,
        },
        "stream": True,
    }

    t0 = time.time()
    first_chunk_time: Optional[float] = None

    final_text = ""
    output_ids: List[int] = []
    meta_info: Dict[str, Any] = {}

    response = requests.post(
        f"{native_base_url.rstrip('/')}/generate",
        json=payload,
        timeout=timeout_s,
        stream=True,
    )

    elapsed_s = 0.0

    try:
        if response.status_code == 400:
            raise RuntimeError(
                "SGLang /generate returned 400.\n"
                f"Response text:\n{response.text}\n\n"
                f"len(input_ids)={len(prompt_ids)}\n"
                f"sampling_params={payload['sampling_params']}"
            )

        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="replace").strip()

            if not line.startswith("data:"):
                continue

            data_str = line[len("data:") :].strip()

            if data_str == "[DONE]":
                break

            if first_chunk_time is None:
                first_chunk_time = time.time()

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # SGLang streaming chunks usually contain cumulative text.
            if isinstance(data.get("text"), str):
                final_text = data["text"]

            # Depending on SGLang version, output_ids may appear on each chunk
            # or only later. Keep the latest non-empty version.
            maybe_output_ids = data.get("output_ids")
            if maybe_output_ids:
                output_ids = list(map(int, maybe_output_ids))

            maybe_meta = data.get("meta_info")
            if isinstance(maybe_meta, dict):
                meta_info = maybe_meta

        elapsed_s = time.time() - t0

    finally:
        with contextlib.suppress(Exception):
            response.close()

    if first_chunk_time is None:
        ttft_s = 0.0
    else:
        ttft_s = max(0.0, first_chunk_time - t0)

    if not output_ids:
        raise RuntimeError(
            "SGLang streaming /generate returned no output_ids. "
            f"final_text_len={len(final_text)}, meta_info={meta_info!r}"
        )

    # Workaround for SGLang versions where output_ids may contain prompt-prefix tokens.
    # Prefer meta_info['completion_tokens'] when present.
    completion_tokens = meta_info.get("completion_tokens")
    if isinstance(completion_tokens, int) and completion_tokens > 0:
        output_ids = output_ids[-completion_tokens:]
    elif len(output_ids) > len(prompt_ids) and output_ids[: len(prompt_ids)] == prompt_ids:
        output_ids = output_ids[len(prompt_ids) :]

    return final_text, output_ids, meta_info, ttft_s, elapsed_s


def _encode_completion_text_for_harmony(encoding: Any, text: str) -> List[int]:
    """
    Best-effort re-tokenization for the OpenAI-compatible /v1/completions fallback.

    Native /generate is better because it returns output_ids directly.
    This fallback is only used when /generate is unavailable.
    """
    if not text:
        return []

    # openai_harmony encoding may expose encode directly.
    if hasattr(encoding, "encode"):
        try:
            return list(map(int, encoding.encode(text)))
        except TypeError:
            pass
        except Exception:
            pass

    # Some wrappers expose the underlying tokenizer/encoding.
    for attr in ("_encoding", "encoding", "tokenizer"):
        inner = getattr(encoding, attr, None)
        if inner is None:
            continue

        if hasattr(inner, "encode"):
            try:
                return list(map(int, inner.encode(text, allowed_special="all")))
            except TypeError:
                try:
                    return list(map(int, inner.encode(text)))
                except Exception:
                    pass
            except Exception:
                pass

    raise RuntimeError(
        "Could not re-tokenize /v1/completions text with the Harmony encoding. "
        "Native /generate returned 404, and fallback tokenization failed."
    )


def sglang_generate_with_ids_streaming_native_on_path(
    *,
    native_base_url: str,
    endpoint_path: str,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_ids: List[int],
    timeout_s: float = 600.0,
) -> tuple[str, List[int], Dict[str, Any], float, float]:
    payload = {
        "input_ids": list(map(int, prompt_ids)),
        "rid": str(uuid.uuid4()),
        "sampling_params": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stop_token_ids": list(map(int, stop_token_ids)),
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "no_stop_trim": True,
        },
        "stream": True,
    }

    url = f"{native_base_url.rstrip('/')}{endpoint_path}"

    t0 = time.time()
    first_chunk_time: Optional[float] = None

    final_text = ""
    output_ids: List[int] = []
    meta_info: Dict[str, Any] = {}

    response = requests.post(
        url,
        json=payload,
        timeout=timeout_s,
        stream=True,
    )

    elapsed_s = 0.0

    try:
        if response.status_code >= 400:
            raise requests.HTTPError(
                f"{response.status_code} Error for {endpoint_path}: {response.text}",
                response=response,
            )

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="replace").strip()

            # SGLang native streaming is usually SSE-style: data: {...}
            # But keep a plain JSON fallback just in case this endpoint changed format.
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
            else:
                data_str = line

            if data_str == "[DONE]":
                break

            if first_chunk_time is None:
                first_chunk_time = time.time()

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if isinstance(data.get("text"), str):
                final_text = data["text"]

            maybe_output_ids = data.get("output_ids")
            if maybe_output_ids:
                output_ids = list(map(int, maybe_output_ids))

            maybe_meta = data.get("meta_info")
            if isinstance(maybe_meta, dict):
                meta_info = maybe_meta

        elapsed_s = time.time() - t0

    finally:
        with contextlib.suppress(Exception):
            response.close()

    ttft_s = 0.0 if first_chunk_time is None else max(0.0, first_chunk_time - t0)

    if not output_ids:
        raise RuntimeError(
            f"SGLang native endpoint {endpoint_path} returned no output_ids. "
            f"final_text_len={len(final_text)}, meta_info={meta_info!r}"
        )

    completion_tokens = meta_info.get("completion_tokens")
    if isinstance(completion_tokens, int) and completion_tokens > 0:
        output_ids = output_ids[-completion_tokens:]
    elif len(output_ids) > len(prompt_ids) and output_ids[: len(prompt_ids)] == prompt_ids:
        output_ids = output_ids[len(prompt_ids) :]

    return final_text, output_ids, meta_info, ttft_s, elapsed_s


def sglang_generate_with_ids_openai_fallback(
    *,
    native_base_url: str,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    encoding: Any,
    model: str = "gpt-oss",
    timeout_s: float = 600.0,
) -> tuple[str, List[int], Dict[str, Any], float, float]:
    """
    Fallback for SGLang images where native /generate is unavailable.

    Uses OpenAI-compatible /v1/completions. This is less ideal than /generate
    because it may not preserve all Harmony special tokens/tool-call structure.
    """
    url = f"{native_base_url.rstrip('/')}/v1/completions"

    payload = {
        "model": model,
        "prompt": list(map(int, prompt_ids)),
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
        "echo": False,

        # SGLang often accepts extra sampling params even on OpenAI-compatible routes.
        # If this route rejects them in your build, remove these three lines.
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
    }

    t0 = time.time()
    response = requests.post(url, json=payload, timeout=timeout_s)
    elapsed_s = time.time() - t0

    if response.status_code >= 400:
        raise RuntimeError(
            f"SGLang OpenAI fallback /v1/completions returned {response.status_code}.\n"
            f"URL: {url}\n"
            f"Response text:\n{response.text}\n"
            f"len(input_ids)={len(prompt_ids)}"
        )

    data = response.json()
    choice = data["choices"][0]
    text = choice.get("text", "")

    if not isinstance(text, str) or not text:
        raise RuntimeError(
            "SGLang OpenAI fallback returned empty text. "
            f"Response: {json.dumps(data)[:2000]}"
        )

    output_ids = _encode_completion_text_for_harmony(encoding, text)

    meta_info = {
        "api_fallback": "openai_v1_completions",
        "raw_response": data,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(output_ids),
    }

    # Non-streaming fallback cannot measure true TTFT.
    ttft_s = elapsed_s

    return text, output_ids, meta_info, ttft_s, elapsed_s


def sglang_generate_with_ids_streaming(
    *,
    native_base_url: str,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_ids: List[int],
    encoding: Any,
    model: str = "gpt-oss",
    timeout_s: float = 600.0,
) -> tuple[str, List[int], Dict[str, Any], float, float]:
    """
    Try SGLang native endpoints first.

    SGLang 0.5.7 used:
        /generate

    This ROCm 0.5.10 image exposes:
        /inference/v1/generate

    If both native endpoints fail, fall back to:
        /v1/completions
    """
    native_endpoint_paths = [
        "/generate",
        "/inference/v1/generate",
    ]

    last_native_error: Optional[Exception] = None

    for endpoint_path in native_endpoint_paths:
        try:
            print(f"[SGLang] Trying native endpoint: {endpoint_path}", flush=True)

            return sglang_generate_with_ids_streaming_native_on_path(
                native_base_url=native_base_url,
                endpoint_path=endpoint_path,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=stop_token_ids,
                timeout_s=timeout_s,
            )

        except requests.HTTPError as exc:
            last_native_error = exc
            status_code = exc.response.status_code if exc.response is not None else None

            print(
                f"[SGLang] Native endpoint {endpoint_path} failed with HTTP {status_code}.",
                flush=True,
            )

            # 404 means this endpoint path is absent; try the next native endpoint.
            if status_code == 404:
                continue

            # Other HTTP errors mean route exists but payload/schema may be wrong.
            # Still try the next endpoint before giving up.
            continue

        except RuntimeError as exc:
            last_native_error = exc

            print(
                f"[SGLang] Native endpoint {endpoint_path} failed: {exc}",
                flush=True,
            )

            # If endpoint exists but does not return output_ids, try the next path.
            continue

    print(
        "[SGLang] All native endpoints failed. "
        "Falling back to /v1/completions.",
        flush=True,
    )

    try:
        return sglang_generate_with_ids_openai_fallback(
            native_base_url=native_base_url,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            encoding=encoding,
            model=model,
            timeout_s=timeout_s,
        )
    except Exception as fallback_exc:
        raise RuntimeError(
            "All SGLang generation endpoints failed. "
            f"Last native error: {type(last_native_error).__name__}: {last_native_error}. "
            f"OpenAI fallback error: {type(fallback_exc).__name__}: {fallback_exc}"
        )


def run_harmony_attempt(
    *,
    task: Any,
    example_index: int,
    native_base_url: str,
    model: str,
    top_p: float,
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
        tool_config = ToolNamespaceConfig(
            name="python",
            description=backend.list_tools()[0]["function"]["description"] if backend.list_tools() else (
                "Use this tool to execute Python code for calculations, verification, "
                "examples, and small brute-force checks. Always use print() to show results."
            ),
            tools=[],
        )

        system_content = (
            SystemContent.new()
            .with_model_identity(SYSTEM_PROMPT)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

        initial_messages = [
            Message.from_role_and_content(Role.SYSTEM, system_content),
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



            request_start = time.time()
            token_buffer: List[int] = []
            text_chunks: List[str] = []

            num_requests += 1

########### ###############################################
            try:
                raw_text, output_ids, meta_info, ttft_s_this_request, request_elapsed_s = (
                    sglang_generate_with_ids_streaming(
                        native_base_url=native_base_url,
                        prompt_ids=prompt_ids,
                        max_new_tokens=remaining_ctx,
                        temperature=temperature,
                        top_p=top_p,
                        stop_token_ids=stop_token_ids,
                        encoding=encoding,
                        model=model,
                        timeout_s=600.0,
                    )
                )
            except Exception as exc:
                errors.append(f"SGLang streaming generation failed: {type(exc).__name__}: {exc}")
                break

            token_buffer = output_ids
            total_output_tokens += len(token_buffer)

            if raw_text:
                text_chunks.append(raw_text)

            # With streaming, TTFT is measurable.
            prefill_time_s_this_request = float(ttft_s_this_request)

            # Decode time excludes time-to-first-token.
            decode_time_s_this_request = max(
                0.0,
                float(request_elapsed_s) - float(ttft_s_this_request),
            )
########### ###############################################
            total_prefill_time_s += prefill_time_s_this_request
            total_decode_time_s += decode_time_s_this_request

            cached_tokens_this_request = int(meta_info.get("cached_tokens", 0) or 0)
            input_tokens_this_request = len(prompt_ids)
            output_tokens_this_request = len(token_buffer)

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

            if getattr(last_message, "channel", None) == "final":
                content = getattr(last_message, "content", None) or []
                if content and getattr(content[0], "text", None) is not None:
                    response_text = content[0].text
                else:
                    response_text = "".join(text_chunks)

                final_answer = _scan_for_answer(response_text)
                # print('[run_imo_answerbench_4.py][line 1135] === Printing Response Text...')
                # print(response_text)
                break

            recipient = getattr(last_message, "recipient", None)
            has_tool_calls_this_request = recipient is not None and "python" in str(recipient)
            num_tool_calls_this_request = 1 if has_tool_calls_this_request else 0

            if recipient is not None and "python" in str(recipient):
                try:
                    tool_name, tool_args = _extract_tool_call(last_message)
                    tool_call_count += 1
                    tool_output = backend.execute_tool(tool_name, tool_args)

                    print(
                        f'[run_imo_answerbench_4.py] python tool called {tool_call_count} times',
                        flush=True
                    )

                    if "[ERROR] Execution timed out" in tool_output:
                        errors.append("Python tool timeout")
                    elif tool_output.startswith("[ERROR]") or "Traceback" in tool_output or "Error:" in tool_output:
                        errors.append("Python tool error")

                    _append_tool_response_to_conversation(
                        conversation,
                        tool_name,
                        tool_output,
                        last_message,
                    )

                    # continue the loop; do NOT break on tool failure
                    continue


                except Exception as exc:
                    errors.append(f"Tool plumbing failed: {type(exc).__name__}: {exc}")
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
    native_base_url: str,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    startup_timeout: float,
    exec_timeout: float,
    preload: str,
    auto_print_last_expr: bool,
    seed: int,
    judge: OpenRouterEquivalenceJudge,
) -> Dict[str, Any]:
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    return await asyncio.to_thread(
        run_harmony_attempt,
        task=task,
        example_index=example_index,
        native_base_url=native_base_url,
        model=model,
        top_p=top_p,
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
    print("\n" + "==============================================================================================================================")
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
    print(result["response"], flush=True)


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

def probe_sglang_endpoints(base_url: str) -> None:
    """
    Probe the live SGLang server and print available/likely endpoints.

    This is intended for debugging SGLang version/API differences.
    It does not run the benchmark.
    """
    base_url = base_url.rstrip("/")

    print("\n" + "=" * 100, flush=True)
    print("SGLang endpoint probe", flush=True)
    print("=" * 100, flush=True)
    print(f"Base URL: {base_url}", flush=True)

    def _preview(text: str, limit: int = 800) -> str:
        text = text.replace("\n", "\\n")
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _request(method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> None:
        url = f"{base_url}{path}"
        try:
            response = requests.request(
                method,
                url,
                json=json_body,
                timeout=10,
            )
            content_type = response.headers.get("content-type", "")
            print(
                f"{method:7s} {path:30s} -> "
                f"{response.status_code} {response.reason} "
                f"content-type={content_type}",
                flush=True,
            )

            body = response.text or ""
            if body:
                print(f"    body: {_preview(body)}", flush=True)

        except Exception as exc:
            print(
                f"{method:7s} {path:30s} -> "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    print("\n--- Basic GET probes ---", flush=True)
    for path in [
        "/",
        "/health",
        "/health_generate",
        "/v1/models",
        "/get_model_info",
        "/docs",
        "/openapi.json",
    ]:
        _request("GET", path)

    print("\n--- OPTIONS probes for likely generation routes ---", flush=True)
    for path in [
        "/generate",
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/responses",
    ]:
        _request("OPTIONS", path)

    print("\n--- Empty POST probes for likely generation routes ---", flush=True)
    print(
        "Note: 400/422 usually means the route exists but the payload is invalid; "
        "404 means the route is probably absent.",
        flush=True,
    )
    for path in [
        "/generate",
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/responses",
    ]:
        _request("POST", path, json_body={})

    print("\n--- Parsed OpenAPI paths, if available ---", flush=True)
    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=10)
        if response.status_code == 200:
            data = response.json()
            paths = sorted(data.get("paths", {}).keys())
            if paths:
                print("Available paths from /openapi.json:", flush=True)
                for path in paths:
                    methods = sorted(data.get("paths", {}).get(path, {}).keys())
                    print(f"  {path}    methods={methods}", flush=True)
            else:
                print("/openapi.json exists but contains no paths.", flush=True)
        else:
            print(
                f"/openapi.json unavailable: {response.status_code} {response.reason}",
                flush=True,
            )
    except Exception as exc:
        print(f"Could not parse /openapi.json: {type(exc).__name__}: {exc}", flush=True)

    print("=" * 100 + "\n", flush=True)


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
            native_base_url=f"http://127.0.0.1:{args.port}",
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            startup_timeout=args.startup_timeout,
            exec_timeout=args.exec_timeout,
            preload=args.preload,
            auto_print_last_expr=args.auto_print_last_expr,
            seed=args.seed + index,
            judge=judge,
        )


        results.append(result)
        append_detailed_result_rows(result.get("detailed_rows", []), output_paths["detailed_results_path"])
        append_output_data_row(result, index, output_paths["output_data_path"])
        print_task_result(index, len(tasks), result)
        print(f'[JUDGE RESPONSE]: {result["judge_response"]}', flush=True)

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

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--log-level", type=str, default="warning")
    parser.add_argument("--top-p", type=float, default=1.0)

    # SGLang memory/scheduling args
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--chunked-prefill-size", type=int, default=16384)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--allow-auto-truncate", action="store_true", default=True)

    parser.add_argument("--probe-sglang-endpoints-and-exit", action="store_true")

    args = parser.parse_args()
    if args.mem_fraction_static is None:
        # Reuse the old vLLM knob as the default SGLang memory fraction.
        args.mem_fraction_static = args.gpu_memory_utilization
    args.model_path = resolve_model_path(args.model_path)
    output_paths = initialize_output_files(args)
    t0 = time.time()

    runtime_cfg = RuntimeConfig(
        served_model_name=args.served_model_name,
        model_path=args.model_path,
        port=args.port,
        seed=args.seed,
        kv_cache_dtype=args.kv_cache_dtype,
        dtype=args.dtype,
        context_tokens=args.context_tokens,
        batch_size=args.batch_size,
        mem_fraction_static=args.mem_fraction_static,
        tensor_parallel_size=args.tensor_parallel_size,
        server_timeout=args.server_timeout,
        preload_workers=args.preload_workers,
        host=args.host,
        log_level=args.log_level,
        chunked_prefill_size=args.chunked_prefill_size,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        enable_torch_compile=args.enable_torch_compile,
        allow_auto_truncate=args.allow_auto_truncate,
        top_p=args.top_p,
    )

    infra = SGLangInfraGPTOSS(runtime_cfg)
    try:
        infra.start()

        if args.probe_sglang_endpoints_and_exit:
            probe_sglang_endpoints(f"http://127.0.0.1:{args.port}") 
            return

        results = asyncio.run(async_main(args, output_paths))
    finally:
        infra.stop()

    wall_time_s = time.time() - t0
    print_summary(results, wall_time_s)
    write_metrics_file(results, wall_time_s, output_paths, args)

if __name__ == "__main__":
    main()
