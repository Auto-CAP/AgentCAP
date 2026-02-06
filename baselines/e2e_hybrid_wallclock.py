#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from agent_cap import Tracer, StepType
from agent_cap.core.types import Trace


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] * (c - k) + xs_sorted[c] * (k - f)


def summarize_numbers(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0.0, "mean": 0.0, "stdev": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": float(len(xs)),
        "mean": float(statistics.mean(xs)),
        "stdev": float(statistics.pstdev(xs) if len(xs) > 1 else 0.0),
        "p50": float(percentile(xs, 50)),
        "p95": float(percentile(xs, 95)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


@dataclass(frozen=True)
class HybridWorkloadConfig:
    planning_base_ms: float = 80.0
    retrieval_base_ms: float = 200.0
    embedding_base_ms: float = 70.0
    prefill_base_ms: float = 0.0
    tool_base_ms: float = 120.0
    code_base_ms: float = 160.0
    reasoning_base_ms: float = 180.0
    tool_calls: int = 1
    jitter: float = 0.10


def _sleep_ms(base_ms: float, jitter: float) -> float:
    j = max(0.0, jitter)
    factor = random.uniform(1.0 - j, 1.0 + j)
    duration_ms = max(0.0, base_ms * factor)
    time.sleep(duration_ms / 1000.0)
    return duration_ms


class Decoder:
    name: str

    def generate(
        self,
        prompt: str,
        *,
        image_path: Optional[Path],
        max_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError


class SleepDecoder(Decoder):
    def __init__(self, base_ms: float, jitter: float) -> None:
        self.name = "sleep"
        self._base_ms = base_ms
        self._jitter = jitter

    def generate(
        self,
        prompt: str,
        *,
        image_path: Optional[Path],
        max_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> Tuple[str, Dict[str, Any]]:
        slept = _sleep_ms(self._base_ms, self._jitter)
        return "OK(sleep)", {"slept_ms": slept, "max_tokens": max_tokens, "temperature": temperature}


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext in [".png"]:
        return "image/png"
    if ext in [".webp"]:
        return "image/webp"
    return "application/octet-stream"


def _file_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime = _guess_mime(path)
    return f"data:{mime};base64,{b64}"


class OpenAICompatChatDecoder(Decoder):
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.name = "openai_compat"
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def _build_messages(self, prompt: str, image_path: Optional[Path]) -> List[Dict[str, Any]]:
        if image_path is None:
            return [{"role": "user", "content": prompt}]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _file_to_data_url(image_path)}},
                ],
            }
        ]

    def generate(
        self,
        prompt: str,
        *,
        image_path: Optional[Path],
        max_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> Tuple[str, Dict[str, Any]]:
        url = f"{self._base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": self._build_messages(prompt, image_path),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        req = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        t0 = time.perf_counter()
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        t1 = time.perf_counter()

        data = json.loads(raw)
        text = ""
        finish_reason = None
        try:
            choice0 = data["choices"][0]
            finish_reason = choice0.get("finish_reason")
            if "message" in choice0 and "content" in choice0["message"]:
                text = choice0["message"]["content"] or ""
            elif "text" in choice0:
                text = choice0["text"] or ""
        except Exception:
            text = ""

        usage = data.get("usage") or {}
        meta: Dict[str, Any] = {
            "backend": "openai_compat_chat_completions",
            "model": self._model,
            "http_elapsed_ms": (t1 - t0) * 1000.0,
            "finish_reason": finish_reason,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        return text, meta


class PythonFunctionDecoder(Decoder):
    def __init__(self, func_spec: str) -> None:
        self.name = "python_func"
        self._func = _load_func(func_spec)
        self._func_spec = func_spec

    def generate(
        self,
        prompt: str,
        *,
        image_path: Optional[Path],
        max_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> Tuple[str, Dict[str, Any]]:
        out = self._func(
            prompt=prompt,
            image_path=str(image_path) if image_path is not None else None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if isinstance(out, str):
            return out, {"backend": "python_func", "func": self._func_spec}
        if isinstance(out, dict):
            text = str(out.get("text", ""))
            meta: Dict[str, Any] = {"backend": "python_func", "func": self._func_spec}
            if "meta" in out and isinstance(out["meta"], dict):
                meta.update(out["meta"])
            if "usage" in out and isinstance(out["usage"], dict):
                for k, v in out["usage"].items():
                    meta[k] = v
            return text, meta
        return str(out), {"backend": "python_func", "func": self._func_spec, "warning": "unexpected_return_type"}


def _load_func(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError(f"--decoder_func must be like 'module:function', got: {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Cannot load callable {fn_name!r} from module {mod_name!r}")
    return fn


def _to_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    try:
        return int(x)
    except Exception:
        return None


def run_one(
    config: HybridWorkloadConfig,
    *,
    seed: int,
    decoder: Decoder,
    prompt: str,
    image_path: Optional[Path],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> Trace:
    random.seed(seed)

    tracer = Tracer(name="hybrid-e2e-wallclock")
    tracer.start()

    # ---- per-trace accumulators (will be stored on agent_run metadata) ----
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_tool_calls = 0

    with tracer.step("agent_run", StepType.OTHER) as s:
        s.metadata["decoder"] = getattr(decoder, "name", "unknown")
        s.metadata["seed"] = seed
        s.metadata["jitter"] = config.jitter
        s.metadata["tool_calls"] = config.tool_calls
        s.metadata["prompt_chars"] = len(prompt)
        s.metadata["prompt_sha16"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        if image_path is not None:
            s.metadata["image_path"] = str(image_path)

        with tracer.step("planning", StepType.PLANNING) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.planning_base_ms, config.jitter)

        with tracer.step("retrieval", StepType.RETRIEVAL) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.retrieval_base_ms, config.jitter)

        with tracer.step("embedding", StepType.EMBEDDING) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.embedding_base_ms, config.jitter)
            step.metadata["embedding_dim"] = 768

        with tracer.step("llm_prefill", StepType.PREFILL) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.prefill_base_ms, config.jitter)
            step.metadata["max_tokens"] = max_tokens
            step.metadata["temperature"] = float(temperature)

        # ---- real decode happens here ----
        with tracer.step("llm_decode", StepType.DECODE) as step:
            text, meta = decoder.generate(
                prompt,
                image_path=image_path,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            step.metadata.update(meta)
            step.metadata["output_chars"] = len(text)

            # accumulate tokens if present
            pt = _to_int_or_none(step.metadata.get("prompt_tokens"))
            ct = _to_int_or_none(step.metadata.get("completion_tokens"))
            tt = _to_int_or_none(step.metadata.get("total_tokens"))
            if pt is not None:
                total_prompt_tokens += pt
            if ct is not None:
                total_completion_tokens += ct
            if tt is not None:
                total_tokens += tt

        # ---- tool calling steps (currently mocked) ----
        for i in range(config.tool_calls):
            with tracer.step(f"tool_call_{i+1}", StepType.TOOL_CALLING) as step:
                step.metadata["slept_ms"] = _sleep_ms(config.tool_base_ms, config.jitter)
                step.metadata["tool_name"] = "mock_tool"

                # tool call count for this step
                # NOTE: in real tool-calling runner, set this to len(model_tool_calls)
                step.metadata["tool_calls_n"] = 0
                total_tool_calls += int(step.metadata["tool_calls_n"])

        with tracer.step("code_execution", StepType.CODE_EXECUTION) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.code_base_ms, config.jitter)
            step.metadata["language"] = "python"

        with tracer.step("reasoning", StepType.REASONING) as step:
            step.metadata["slept_ms"] = _sleep_ms(config.reasoning_base_ms, config.jitter)

        # ---- store per-trace totals on top-level agent_run ----
        s.metadata["total_prompt_tokens"] = total_prompt_tokens if total_prompt_tokens > 0 else None
        s.metadata["total_completion_tokens"] = total_completion_tokens if total_completion_tokens > 0 else None
        s.metadata["total_tokens"] = total_tokens if total_tokens > 0 else None
        s.metadata["total_tool_calls"] = total_tool_calls

    return tracer.stop()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hybrid E2E wall-clock baseline (Agent skeleton + real decode).")

    ap.add_argument("--runs", type=int, default=10, help="Number of runs for statistics")
    ap.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed")

    ap.add_argument("--out_dir", type=str, default="baselines/hybrid_e2e_runs", help="Output directory for traces")
    ap.add_argument("--summary_path", type=str, default="baselines/hybrid_e2e_wallclock_summary.json", help="Summary output path")

    ap.add_argument("--jitter", type=float, default=0.10, help="Jitter ratio for sleep steps")
    ap.add_argument("--tool_calls", type=int, default=1, help="Number of mock tool calls")
    ap.add_argument("--planning_ms", type=float, default=80.0)
    ap.add_argument("--retrieval_ms", type=float, default=200.0)
    ap.add_argument("--embedding_ms", type=float, default=70.0)
    ap.add_argument("--prefill_ms", type=float, default=0.0, help="Prefill time (default 0 to avoid double-count)")
    ap.add_argument("--tool_ms", type=float, default=120.0)
    ap.add_argument("--code_ms", type=float, default=160.0)
    ap.add_argument("--reasoning_ms", type=float, default=180.0)

    ap.add_argument("--prompt", type=str, default=None, help="Prompt string")
    ap.add_argument("--prompt_file", type=str, default=None, help="Read prompt from file")
    ap.add_argument("--image", type=str, default=None, help="Optional image path for multimodal models")

    ap.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.0, help="Temperature (0 for reproducibility)")
    ap.add_argument("--timeout_s", type=float, default=120.0, help="HTTP/backend timeout in seconds")

    ap.add_argument("--decoder", choices=["sleep", "openai_compat", "python"], default="sleep")

    ap.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", type=str, default=None, help="Model name for openai-compatible API")

    ap.add_argument("--decoder_func", type=str, default=None, help="module:function for python decoder")

    return ap.parse_args()


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    raise SystemExit("You must provide --prompt or --prompt_file")


def _build_decoder(args: argparse.Namespace, config: HybridWorkloadConfig) -> Decoder:
    if args.decoder == "sleep":
        return SleepDecoder(base_ms=350.0, jitter=args.jitter)
    if args.decoder == "openai_compat":
        if not args.model:
            raise SystemExit("--model is required when --decoder openai_compat")
        return OpenAICompatChatDecoder(base_url=args.base_url, api_key=args.api_key, model=args.model)
    if args.decoder == "python":
        if not args.decoder_func:
            raise SystemExit("--decoder_func is required when --decoder python")
        return PythonFunctionDecoder(args.decoder_func)
    raise SystemExit(f"Unknown decoder: {args.decoder}")


def main() -> None:
    args = parse_args()
    prompt = _read_prompt(args)
    image_path = Path(args.image) if args.image else None

    config = HybridWorkloadConfig(
        planning_base_ms=args.planning_ms,
        retrieval_base_ms=args.retrieval_ms,
        embedding_base_ms=args.embedding_ms,
        prefill_base_ms=args.prefill_ms,
        tool_base_ms=args.tool_ms,
        code_base_ms=args.code_ms,
        reasoning_base_ms=args.reasoning_ms,
        tool_calls=args.tool_calls,
        jitter=args.jitter,
    )

    decoder = _build_decoder(args, config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # warmup (not recorded)
    for i in range(max(0, args.warmup)):
        _ = run_one(
            config,
            seed=args.seed + i,
            decoder=decoder,
            prompt=prompt,
            image_path=image_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )

    totals_ms: List[float] = []
    trace_paths: List[str] = []

    for i in range(max(1, args.runs)):
        seed_i = args.seed + 1000 + i
        trace = run_one(
            config,
            seed=seed_i,
            decoder=decoder,
            prompt=prompt,
            image_path=image_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        totals_ms.append(trace.total_duration_ms)

        path = out_dir / f"trace_{i+1:03d}.json"
        path.write_text(trace.to_json(indent=2), encoding="utf-8")
        trace_paths.append(str(path))

    summary = {
        "definition": "Hybrid E2E wall-clock latency per agent run = Trace.total_duration_ms",
        "workload": "agent skeleton + real decode backend",
        "runs": args.runs,
        "warmup": args.warmup,
        "decoder": {
            "type": args.decoder,
            "model": args.model,
            "base_url": args.base_url if args.decoder == "openai_compat" else None,
            "decoder_func": args.decoder_func if args.decoder == "python" else None,
            "max_tokens": args.max_tokens,
            "temperature": float(args.temperature),
        },
        "config": {
            "planning_base_ms": config.planning_base_ms,
            "retrieval_base_ms": config.retrieval_base_ms,
            "embedding_base_ms": config.embedding_base_ms,
            "prefill_base_ms": config.prefill_base_ms,
            "tool_base_ms": config.tool_base_ms,
            "code_base_ms": config.code_base_ms,
            "reasoning_base_ms": config.reasoning_base_ms,
            "tool_calls": config.tool_calls,
            "jitter": config.jitter,
        },
        "prompt": {
            "chars": len(prompt),
            "sha16": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "has_image": bool(image_path),
        },
        "env": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "trace_dir": str(out_dir),
        "trace_files": trace_paths[:5] + (["..."] if len(trace_paths) > 5 else []),
        "total_duration_ms": summarize_numbers(totals_ms),
    }

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] saved {len(trace_paths)} traces -> {out_dir}")
    print(f"[OK] saved summary -> {summary_path}")
    print(
        "[E2E] total_duration_ms  "
        f"p50={summary['total_duration_ms']['p50']:.2f}  "
        f"p95={summary['total_duration_ms']['p95']:.2f}  "
        f"mean={summary['total_duration_ms']['mean']:.2f}"
    )

if __name__ == "__main__":
    main()