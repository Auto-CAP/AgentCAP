import argparse
import asyncio
import contextlib
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from huggingface_hub import snapshot_download
from math_verify import parse, verify
from openai import OpenAI

from agent_cap.benchmarks import load_benchmark
from agent_cap.backends.math_python_backend import MathPythonBackend
from agent_cap.runner.unified_runner import collect_hardware_info


SYSTEM_PROMPT = """You are an elite mathematical problem solver with expertise at the International Mathematical Olympiad (IMO) level.

# Output Format
- Provide a brief summary of the solution.
- Then state the final mathematical answer clearly.
- Put the final answer inside \\boxed{...}.
- The final answer may be an integer, fraction, expression, tuple, sequence, set, or other mathematical object, depending on the problem.
- Do not put anything except the final answer inside the final \\boxed{...}.
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

    print("collecting hardware info...", flush=True)
    hw_info = collect_hardware_info_rocm_fallback()
    print("successfully collected hardware info.", flush=True)

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
            "top_p": args.top_p,
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
    cached_tokens_list = [int(r.get("total_cached_tokens", 0)) for r in results]

    total_input_tokens = int(sum(input_tokens_list))
    total_output_tokens = int(sum(output_tokens_list))
    total_tool_calls = int(sum(tool_calls_list))
    total_cached_tokens = int(sum(cached_tokens_list))

    num_requests_list = [int(r.get("num_requests", 1)) for r in results]
    total_requests = int(sum(num_requests_list))

    input_tokens_per_request = []
    output_tokens_per_request = []
    max_input_tokens_per_request_list = []

    for r in results:
        reqs = max(1, int(r.get("num_requests", 1)))
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

    avg_cache_hit_rate = (
        float(total_cached_tokens) / float(total_input_tokens)
        if total_input_tokens > 0
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
            "total_cached_tokens": total_cached_tokens,
            "avg_cache_hit_rate": avg_cache_hit_rate,
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


def append_output_data_row(result: Dict[str, Any], index: int, output_data_path: str) -> None:
    row = {
        "index": index - 1,
        "task_id": result["task_id"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "tool_call_count": result["tool_calls"],
        "num_requests": result["num_requests"],
        "e2e_latency_s": float(result["latency_ms"]) / 1000.0,
        "output_text": result["response"],
        "reasoning_text": result.get("reasoning", ""),
        "errors": result["errors"],
        "eval_passed": result["judge_equivalent"],
        "eval_score": result["score"],
        "eval_details": result["judge_response"],
    }

    with open(output_data_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_detailed_result_rows(detailed_rows: List[Dict[str, Any]], detailed_results_path: str) -> None:
    with open(detailed_results_path, "a", encoding="utf-8") as f:
        for row in detailed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class SyncMathPythonBackend:
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
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY.")

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
        return min(self.backoff_start_s * attempt, self.backoff_cap_s)

    def judge_equivalence(self, predicted: Optional[str], expected: Optional[str]) -> Dict[str, Any]:
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
            "messages": [{"role": "user", "content": prompt}],
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
                    "raw_response": f"Judge failed after {self.max_retries} attempts: {type(exc).__name__}: {exc}",
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
                    "raw_response": f"Judge failed after {self.max_retries} attempts: {type(exc).__name__}: {exc}",
                    "status_code": None,
                    "response_json": None,
                }

        return {
            "equivalent": False,
            "raw_response": (
                f"Judge failed after {self.max_retries} attempts: {type(last_error).__name__}: {last_error}"
                if last_error is not None
                else "Judge failed for unknown reason."
            ),
            "status_code": None,
            "response_json": None,
        }


def resolve_model_path(model_path: str, local_model_root: Optional[str] = None) -> str:
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


@dataclass
class RuntimeConfig:
    served_model_name: str
    model_path: str
    port: int
    host: str
    seed: int
    kv_cache_dtype: Optional[str]
    dtype: Optional[str]
    context_tokens: int
    batch_size: int  # benchmark metadata only; not passed as --max-total-tokens
    mem_fraction_static: float
    tensor_parallel_size: int
    data_parallel_size: int
    expert_parallel_size: int
    enable_expert_parallel: bool
    enable_dp_attention: bool
    tool_call_parser: Optional[str]
    reasoning_parser: Optional[str]
    chat_template: Optional[str]
    server_timeout: int
    preload_workers: int
    preload_model_weights: bool
    log_level: str
    chunked_prefill_size: Optional[int]
    cuda_graph_max_bs: Optional[int]
    max_running_requests: Optional[int]
    trust_remote_code: bool
    allow_auto_truncate: bool
    enable_metrics: bool
    extra_sglang_args: str


class SGLangInfraDeepSeek32:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.port = cfg.port
        client_host = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
        self.native_base_url = f"http://{client_host}:{cfg.port}"
        self.openai_base_url = f"{self.native_base_url}/v1"
        self.server_process: Optional[subprocess.Popen] = None
        self.log_file = None

    def start(self) -> None:
        if self.cfg.preload_model_weights:
            self._preload_model_weights()
        else:
            print(
                "Skipping model weight preload into OS page cache. "
                "This is safer for very large DeepSeek-V3.2 checkpoints.",
                flush=True,
            )
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
            with contextlib.suppress(Exception):
                self.log_file.close()

    def _preload_model_weights(self) -> None:
        if not os.path.isdir(self.cfg.model_path):
            raise FileNotFoundError(f"Resolved model path does not exist: {self.cfg.model_path}")

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
            f"Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) "
            f"in {elapsed:.2f} seconds.\n",
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
            "--served-model-name",
            self.cfg.served_model_name,
            "--host",
            self.cfg.host,
            "--port",
            str(self.cfg.port),
            "--log-level",
            self.cfg.log_level,
            "--context-length",
            str(self.cfg.context_tokens),
            "--mem-fraction-static",
            str(self.cfg.mem_fraction_static),
            "--tp-size",
            str(self.cfg.tensor_parallel_size),
            "--random-seed",
            str(self.cfg.seed),
        ]

        if self.cfg.data_parallel_size > 1:
            cmd += ["--dp-size", str(self.cfg.data_parallel_size)]

        if self.cfg.enable_dp_attention:
            cmd.append("--enable-dp-attention")

        if self.cfg.enable_expert_parallel:
            if self.cfg.expert_parallel_size > 0:
                cmd += ["--ep-size", str(self.cfg.expert_parallel_size)]
            else:
                cmd += ["--ep-size", str(self.cfg.tensor_parallel_size)]

        if self.cfg.trust_remote_code:
            cmd.append("--trust-remote-code")

        if self.cfg.dtype:
            cmd += ["--dtype", self.cfg.dtype]

        if self.cfg.kv_cache_dtype:
            cmd += ["--kv-cache-dtype", self.cfg.kv_cache_dtype]

        # Keep batch_size as benchmark metadata only. For SGLang serving concurrency,
        # use --max-running-requests instead; do not map batch_size to --max-total-tokens.

        if self.cfg.tool_call_parser:
            cmd += ["--tool-call-parser", self.cfg.tool_call_parser]

        if self.cfg.reasoning_parser:
            cmd += ["--reasoning-parser", self.cfg.reasoning_parser]

        if self.cfg.chat_template:
            cmd += ["--chat-template", self.cfg.chat_template]

        if self.cfg.chunked_prefill_size is not None:
            cmd += ["--chunked-prefill-size", str(self.cfg.chunked_prefill_size)]

        if self.cfg.cuda_graph_max_bs is not None:
            cmd += ["--cuda-graph-max-bs", str(self.cfg.cuda_graph_max_bs)]

        if self.cfg.max_running_requests is not None:
            cmd += ["--max-running-requests", str(self.cfg.max_running_requests)]

        if self.cfg.allow_auto_truncate:
            cmd.append("--allow-auto-truncate")

        if self.cfg.enable_metrics:
            cmd.append("--enable-metrics")

        if self.cfg.extra_sglang_args.strip():
            import shlex

            cmd.extend(shlex.split(self.cfg.extra_sglang_args))

        self.log_file = open("sglang_server_deepseek32.log", "w", encoding="utf-8")

        print("Launching SGLang DeepSeek-V3.2:")
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
                with open("sglang_server_deepseek32.log", "r", encoding="utf-8", errors="ignore") as log_file:
                    logs = log_file.read()
                raise RuntimeError(f"SGLang server died with code {return_code}. Full logs:\n{logs}\n")

            try:
                req = urllib.request.Request(models_url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
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


def _scan_for_answer(text: str) -> Optional[str]:
    boxed_content = extract_last_boxed_content(text)
    if boxed_content is not None:
        return boxed_content.strip()

    matches = re.findall(r"final\s+answer\s+is\s*(.+)", text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    bold_matches = re.findall(r"(?:\*\*|__)\s*(.+?)\s*(?:\*\*|__)", text)
    if bold_matches:
        return bold_matches[-1].strip()

    return None


def compute_score(
    solution_str: str,
    ground_truth: str,
    judge: OpenRouterEquivalenceJudge,
) -> tuple[float, Optional[str], Dict[str, Any]]:
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


def _task_message_to_openai(msg: Dict[str, Any]) -> Dict[str, Any]:
    role = msg["role"]
    content = msg["content"]
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"Unsupported benchmark message role: {role}")
    return {"role": role, "content": content}


def _safe_json_loads_arguments(arguments: Optional[str]) -> Dict[str, Any]:
    if arguments is None or arguments.strip() == "":
        return {}
    try:
        parsed = json.loads(arguments)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except Exception:
        return {"code": arguments}


def _get_usage_int(usage: Any, name: str, default: int = 0) -> int:
    value = getattr(usage, name, None)
    if value is None:
        return default
    return int(value)


def _get_cached_tokens_from_usage(usage: Any) -> int:
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    cached = getattr(details, "cached_tokens", None)
    return int(cached) if cached is not None else 0


def _normalise_tool_call_for_history(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": tool_call["id"],
        "type": "function",
        "function": {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"].get("arguments", "{}"),
        },
    }


def _build_tool_definitions(backend: SyncMathPythonBackend) -> List[Dict[str, Any]]:
    tools = backend.list_tools()
    if tools:
        return tools

    return [
        {
            "type": "function",
            "function": {
                "name": "python",
                "description": (
                    "Execute Python code for calculations, verification, examples, "
                    "and small brute-force checks. Always use print() to show results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute.",
                        }
                    },
                    "required": ["code"],
                },
            },
        }
    ]


def _build_extra_body(enable_thinking: bool, separate_reasoning: bool, stream_reasoning: bool) -> Dict[str, Any]:
    extra_body: Dict[str, Any] = {
        "chat_template_kwargs": {
            "thinking": enable_thinking,
        }
    }
    if separate_reasoning:
        extra_body["separate_reasoning"] = True
        extra_body["stream_reasoning"] = stream_reasoning
    return extra_body


def stream_deepseek32_sglang_chat_completion(
    *,
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    enable_thinking: bool,
    separate_reasoning: bool,
    stream_reasoning: bool,
) -> Dict[str, Any]:
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_body": _build_extra_body(
            enable_thinking=enable_thinking,
            separate_reasoning=separate_reasoning,
            stream_reasoning=stream_reasoning,
        ),
    }

    if tools:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = "auto"

    stream_start = time.time()
    first_token_time = None
    stream_end = stream_start

    content_chunks: List[str] = []
    reasoning_chunks: List[str] = []
    tool_call_parts: Dict[int, Dict[str, Any]] = {}

    final_usage = None
    finish_reason = None

    stream = client.chat.completions.create(**request_kwargs)

    try:
        for chunk in stream:
            stream_end = time.time()

            usage = getattr(chunk, "usage", None)
            if usage is not None:
                final_usage = usage

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            choice = choices[0]
            if getattr(choice, "finish_reason", None) is not None:
                finish_reason = choice.finish_reason

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            got_meaningful_delta = False

            content_delta = getattr(delta, "content", None)
            if content_delta:
                content_chunks.append(content_delta)
                got_meaningful_delta = True

            reasoning_delta = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
            )
            if reasoning_delta:
                reasoning_chunks.append(reasoning_delta)
                got_meaningful_delta = True

            delta_tool_calls = getattr(delta, "tool_calls", None) or []
            for tc in delta_tool_calls:
                idx = getattr(tc, "index", None)
                if idx is None:
                    idx = len(tool_call_parts)

                if idx not in tool_call_parts:
                    tool_call_parts[idx] = {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }

                tc_id = getattr(tc, "id", None)
                if tc_id:
                    tool_call_parts[idx]["id"] = tc_id

                tc_type = getattr(tc, "type", None)
                if tc_type:
                    tool_call_parts[idx]["type"] = tc_type

                fn = getattr(tc, "function", None)
                if fn is not None:
                    fn_name = getattr(fn, "name", None)
                    if fn_name:
                        tool_call_parts[idx]["function"]["name"] += fn_name

                    fn_args = getattr(fn, "arguments", None)
                    if fn_args:
                        tool_call_parts[idx]["function"]["arguments"] += fn_args

                got_meaningful_delta = True

            if got_meaningful_delta and first_token_time is None:
                first_token_time = time.time()

    finally:
        with contextlib.suppress(Exception):
            stream.close()

    if first_token_time is None:
        first_token_time = stream_end

    tool_calls: List[Dict[str, Any]] = []
    for idx in sorted(tool_call_parts):
        item = tool_call_parts[idx]
        function_name = item["function"]["name"]
        if not function_name:
            continue
        if not item["id"]:
            item["id"] = f"call_{idx}"
        if not item["function"].get("arguments"):
            item["function"]["arguments"] = "{}"
        tool_calls.append(_normalise_tool_call_for_history(item))

    prompt_tokens = _get_usage_int(final_usage, "prompt_tokens", 0)
    completion_tokens = _get_usage_int(final_usage, "completion_tokens", 0)
    cached_tokens = _get_cached_tokens_from_usage(final_usage)

    return {
        "content": "".join(content_chunks),
        "reasoning": "".join(reasoning_chunks),
        "tool_calls": tool_calls,
        "usage": final_usage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "prefill_time_s": max(0.0, first_token_time - stream_start),
        "decode_time_s": max(0.0, stream_end - first_token_time),
        "finish_reason": finish_reason,
    }


def run_deepseek32_sglang_attempt(
    *,
    task: Any,
    example_index: int,
    client: OpenAI,
    model: str,
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
    enable_thinking: bool,
    separate_reasoning: bool,
    stream_reasoning: bool,
) -> Dict[str, Any]:
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
    total_cached_tokens = 0

    tool_call_count = 0
    errors: List[str] = []
    response_text = ""
    reasoning_text = ""
    num_requests = 0
    detailed_rows: List[Dict[str, Any]] = []

    backend.setup(task.eval_config or {})

    try:
        tools = _build_tool_definitions(backend)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        messages.extend(_task_message_to_openai(m) for m in task.messages)

        for turn_idx in range(max_turns):
            num_requests += 1

            request_result = stream_deepseek32_sglang_chat_completion(
                client=client,
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                enable_thinking=enable_thinking,
                separate_reasoning=separate_reasoning,
                stream_reasoning=stream_reasoning,
            )

            content = request_result["content"]
            reasoning = request_result["reasoning"]
            tool_calls = request_result["tool_calls"]

            input_tokens_this_request = int(request_result["prompt_tokens"])
            output_tokens_this_request = int(request_result["completion_tokens"])
            cached_tokens_this_request = int(request_result["cached_tokens"])

            prefill_time_s_this_request = float(request_result["prefill_time_s"])
            decode_time_s_this_request = float(request_result["decode_time_s"])

            total_input_tokens += input_tokens_this_request
            total_output_tokens += output_tokens_this_request
            total_cached_tokens += cached_tokens_this_request
            total_prefill_time_s += prefill_time_s_this_request
            total_decode_time_s += decode_time_s_this_request

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

            has_tool_calls_this_request = bool(tool_calls)
            num_tool_calls_this_request = len(tool_calls)

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
                    "finish_reason": request_result.get("finish_reason"),
                }
            )

            if not content and not reasoning and not tool_calls:
                errors.append("Model returned no streamed content, reasoning, or tool calls.")
                break

            if reasoning:
                reasoning_text += reasoning

            if tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": content if content else None,
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_message)

                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments_str = tool_call["function"].get("arguments", "{}")
                    tool_args = _safe_json_loads_arguments(arguments_str)

                    try:
                        tool_output = backend.execute_tool(function_name, tool_args)
                    except Exception as exc:
                        tool_output = f"[ERROR] Tool execution failed: {type(exc).__name__}: {exc}"

                    tool_call_count += 1
                    print(
                        f"[run_imo_answerbench_deepseek32_sglang.py] "
                        f"tool called {tool_call_count} times: {function_name}",
                        flush=True,
                    )

                    if "[ERROR] Execution timed out" in tool_output:
                        errors.append("Python tool timeout")
                    elif (
                        tool_output.startswith("[ERROR]")
                        or "Traceback" in tool_output
                        or "Error:" in tool_output
                    ):
                        errors.append("Python tool error")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_output,
                        }
                    )

                continue

            response_text = content or ""
            if not response_text and reasoning:
                response_text = reasoning
            break

        expected = (task.eval_config or {}).get("expected")
        score, extracted_predicted, judge_result = compute_score(response_text, expected, judge)

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
            "reasoning": reasoning_text,
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
            "total_cached_tokens": total_cached_tokens,
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
    top_p: float,
    startup_timeout: float,
    exec_timeout: float,
    preload: str,
    auto_print_last_expr: bool,
    seed: int,
    judge: OpenRouterEquivalenceJudge,
    enable_thinking: bool,
    separate_reasoning: bool,
    stream_reasoning: bool,
) -> Dict[str, Any]:
    client = OpenAI(base_url=base_url, api_key="dummy", timeout=600)

    return await asyncio.to_thread(
        run_deepseek32_sglang_attempt,
        task=task,
        example_index=example_index,
        client=client,
        model=model,
        max_turns=max_turns,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        startup_timeout=startup_timeout,
        exec_timeout=exec_timeout,
        preload=preload,
        auto_print_last_expr=auto_print_last_expr,
        seed=seed,
        judge=judge,
        enable_thinking=enable_thinking,
        separate_reasoning=separate_reasoning,
        stream_reasoning=stream_reasoning,
    )


def print_task_result(index: int, total: int, result: Dict[str, Any]) -> None:
    status = "✅" if result["correct"] else "❌"
    print("\n" + "=" * 126)
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
    base_url = base_url.rstrip("/")
    print("\n" + "=" * 100, flush=True)
    print("SGLang endpoint probe", flush=True)
    print("=" * 100, flush=True)
    print(f"Base URL: {base_url}", flush=True)

    def _preview(text: str, limit: int = 1000) -> str:
        text = text.replace("\n", "\\n")
        return text[:limit] + "..." if len(text) > limit else text

    def _request(method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> None:
        url = f"{base_url}{path}"
        try:
            response = requests.request(method, url, json=json_body, timeout=10)
            content_type = response.headers.get("content-type", "")
            print(
                f"{method:7s} {path:30s} -> {response.status_code} {response.reason} "
                f"content-type={content_type}",
                flush=True,
            )
            body = response.text or ""
            if body:
                print(f"    body: {_preview(body)}", flush=True)
        except Exception as exc:
            print(f"{method:7s} {path:30s} -> {type(exc).__name__}: {exc}", flush=True)

    for path in ["/", "/health", "/health_generate", "/v1/models", "/get_model_info", "/docs", "/openapi.json"]:
        _request("GET", path)

    for path in ["/generate", "/v1/completions", "/v1/chat/completions", "/v1/responses"]:
        _request("OPTIONS", path)

    print("\nEmpty POST probes. 400/422 usually means route exists; 404 means absent.", flush=True)
    for path in ["/generate", "/v1/completions", "/v1/chat/completions", "/v1/responses"]:
        _request("POST", path, json_body={})

    print("=" * 100 + "\n", flush=True)


async def async_main(args: argparse.Namespace, output_paths: Dict[str, str]) -> List[Dict[str, Any]]:
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
            top_p=args.top_p,
            startup_timeout=args.startup_timeout,
            exec_timeout=args.exec_timeout,
            preload=args.preload,
            auto_print_last_expr=args.auto_print_last_expr,
            seed=args.seed + index,
            judge=judge,
            enable_thinking=args.enable_thinking,
            separate_reasoning=args.separate_reasoning,
            stream_reasoning=args.stream_reasoning,
        )

        results.append(result)
        append_detailed_result_rows(result.get("detailed_rows", []), output_paths["detailed_results_path"])
        append_output_data_row(result, index, output_paths["output_data_path"])
        print_task_result(index, len(tasks), result)
        print(f"[JUDGE RESPONSE]: {result['judge_response']}", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="deepseek-v32")
    parser.add_argument("--served-model-name", type=str, default="deepseek-v32")
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument(
        "--preload-model-weights",
        dest="preload_model_weights",
        action="store_true",
        default=False,
        help="Read all model files into OS page cache before starting SGLang.",
    )
    parser.add_argument("--no-preload-model-weights", dest="preload_model_weights", action="store_false")

    parser.add_argument("--kv-cache-dtype", type=str, default="fp8_e4m3")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--context-tokens", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--mem-fraction-static", type=float, default=None)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=0)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--enable-dp-attention", action="store_true", default=False)

    parser.add_argument("--tool-call-parser", type=str, default="deepseekv32")
    parser.add_argument("--reasoning-parser", type=str, default="deepseek-v3")
    parser.add_argument("--chat-template", type=str, default=None)

    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--separate-reasoning", action="store_true", default=True)
    parser.add_argument("--no-separate-reasoning", dest="separate_reasoning", action="store_false")
    parser.add_argument("--stream-reasoning", action="store_true", default=True)
    parser.add_argument("--no-stream-reasoning", dest="stream_reasoning", action="store_false")

    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--exec-timeout", type=float, default=5.0)
    parser.add_argument("--preload", type=str, default="minimal", choices=["none", "minimal", "full"])
    parser.add_argument("--auto-print-last-expr", action="store_true")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="warning")
    parser.add_argument("--server-timeout", type=int, default=3600)
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument("--judge-model", type=str, default="openrouter/elephant-alpha")

    parser.add_argument("--chunked-prefill-size", type=int, default=16384)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--allow-auto-truncate", action="store_true", default=True)
    parser.add_argument("--no-allow-auto-truncate", dest="allow_auto_truncate", action="store_false")
    parser.add_argument("--enable-metrics", action="store_true", default=False)
    parser.add_argument("--extra-sglang-args", type=str, default="")

    parser.add_argument("--probe-sglang-endpoints-and-exit", action="store_true")

    args = parser.parse_args()

    if args.mem_fraction_static is None:
        args.mem_fraction_static = args.gpu_memory_utilization

    args.model_path = resolve_model_path(args.model_path)
    output_paths = initialize_output_files(args)
    t0 = time.time()
    results: List[Dict[str, Any]] = []

    runtime_cfg = RuntimeConfig(
        served_model_name=args.served_model_name,
        model_path=args.model_path,
        port=args.port,
        host=args.host,
        seed=args.seed,
        kv_cache_dtype=args.kv_cache_dtype,
        dtype=args.dtype,
        context_tokens=args.context_tokens,
        batch_size=args.batch_size,
        mem_fraction_static=args.mem_fraction_static,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        enable_dp_attention=args.enable_dp_attention,
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        chat_template=args.chat_template,
        server_timeout=args.server_timeout,
        preload_workers=args.preload_workers,
        preload_model_weights=args.preload_model_weights,
        log_level=args.log_level,
        chunked_prefill_size=args.chunked_prefill_size,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        max_running_requests=args.max_running_requests,
        trust_remote_code=args.trust_remote_code,
        allow_auto_truncate=args.allow_auto_truncate,
        enable_metrics=args.enable_metrics,
        extra_sglang_args=args.extra_sglang_args,
    )

    infra = SGLangInfraDeepSeek32(runtime_cfg)
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
