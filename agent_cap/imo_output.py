"""Shared output, aggregation, and concurrency helpers for IMO producers."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple


def positive_int(value: str) -> int:
    """Argparse type for strictly positive concurrency limits."""
    parsed = int(value)
    if parsed < 1:
        raise ValueError("must be a positive integer")
    return parsed


class ActivityTracker:
    """Thread-safe current/peak activity counter usable as a context manager."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0
        self._maximum = 0

    @property
    def maximum(self) -> int:
        with self._lock:
            return self._maximum

    def reset(self) -> None:
        with self._lock:
            if self._current:
                raise RuntimeError("cannot reset an active concurrency tracker")
            self._maximum = 0

    @contextlib.contextmanager
    def active(self) -> Iterable[None]:
        with self._lock:
            self._current += 1
            self._maximum = max(self._maximum, self._current)
        try:
            yield
        finally:
            with self._lock:
                self._current -= 1


# Both DeepSeek-v3.2 entry points use this process-wide tracker. It spans the
# full streamed-response lifetime, not merely client.chat.completions.create().
LLM_REQUEST_CONCURRENCY = ActivityTracker()


class _TrackedStream:
    def __init__(self, create_stream: Callable[[], Any]) -> None:
        self._activity = LLM_REQUEST_CONCURRENCY.active()
        self._closed = False
        self._activity.__enter__()
        try:
            self._stream = create_stream()
        except BaseException:
            self._activity.__exit__(*__import__("sys").exc_info())
            self._closed = True
            raise

    def __iter__(self):
        return iter(self._stream)

    def close(self) -> None:
        if self._closed:
            return
        try:
            close = getattr(self._stream, "close", None)
            if close is not None:
                close()
        finally:
            self._closed = True
            self._activity.__exit__(None, None, None)


def tracked_stream(create_stream: Callable[[], Any]) -> _TrackedStream:
    """Create a stream whose entire creation/iteration/close lifetime is tracked."""
    return _TrackedStream(create_stream)


def build_failed_task_result(
    task: Any,
    error_message: str,
    latency_ms: float,
) -> Dict[str, Any]:
    """Return a publishable task row for an uncaught task-level exception."""
    expected = (getattr(task, "eval_config", None) or {}).get("expected")
    return {
        "task_id": task.id,
        "task_name": task.name,
        "category": task.category,
        "expected": expected,
        "predicted": None,
        "score": 0.0,
        "correct": False,
        "response": "",
        "reasoning": "",
        "tool_calls": 0,
        "num_requests": 0,
        "tool_latencies_ms": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": latency_ms,
        # No request completed, so these measurements are unavailable rather
        # than measured zero.
        "ttft_ms": None,
        "prefill_total_s": None,
        "tpot_ms_avg": None,
        "tpot_ms_p99": None,
        "errors": [error_message],
        "judge_equivalent": False,
        "judge_response": f"Evaluation skipped: {error_message}",
        "judge_status_code": None,
        "judge_attempts": 0,
        "detailed_rows": [],
        "total_cached_tokens": 0,
        "finish_reason": "unhandled_task_exception",
    }


async def run_tasks_concurrently(
    tasks: List[Any],
    concurrency: int,
    run_one: Callable[[int, Any], Awaitable[Dict[str, Any]]],
    persist_one: Callable[[int, Dict[str, Any]], None],
) -> Tuple[List[Dict[str, Any]], int]:
    """Run indexed tasks with a hard cap and serialized, uniquely indexed writes."""
    if concurrency < 1:
        raise ValueError("concurrency must be a positive integer")

    semaphore = asyncio.Semaphore(concurrency)
    task_activity = ActivityTracker()

    async def worker(index: int, task: Any) -> Tuple[int, Dict[str, Any]]:
        started = time.monotonic()
        async with semaphore:
            try:
                with task_activity.active():
                    result = await run_one(index, task)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                result = build_failed_task_result(
                    task,
                    message,
                    (time.monotonic() - started) * 1000.0,
                )
        return index, result

    futures = [
        asyncio.create_task(worker(index, task))
        for index, task in enumerate(tasks)
    ]
    ordered: List[Optional[Dict[str, Any]]] = [None] * len(tasks)

    # This loop is the sole writer even when tasks finish out of order.
    for future in asyncio.as_completed(futures):
        index, result = await future
        ordered[index] = result
        persist_one(index, result)

    return [result for result in ordered if result is not None], task_activity.maximum


def update_metadata_concurrency(
    metadata_path: str,
    *,
    concurrency: int,
    observed_max_concurrency: int,
    observed_max_simultaneous_llm_requests: int,
) -> None:
    """Persist requested and observed concurrency in the canonical metadata section."""
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    environment = metadata.setdefault("system_environment", {})
    environment["concurrency"] = concurrency
    environment["observed_max_concurrency"] = observed_max_concurrency
    environment["observed_max_simultaneous_llm_requests"] = (
        observed_max_simultaneous_llm_requests
    )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)


def _mean(values: List[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def _p99(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=100, method="inclusive")[98])


def _numeric_values(
    results: List[Dict[str, Any]], key: str, *, scale: float = 1.0
) -> List[float]:
    values: List[float] = []
    for result in results:
        value = _finite_nonnegative_number(result.get(key))
        if value is not None:
            values.append(value * scale)
    return values


def _exact_nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        if value == "0" or (value.isascii() and value.isdigit() and not value.startswith("0")):
            return int(value)
    return None


def _finite_nonnegative_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _result_count(result: Dict[str, Any], key: str) -> int:
    value = _exact_nonnegative_int(result.get(key, 0))
    if value is None:
        raise ValueError(f"{key} must be an exact nonnegative integer")
    return value


def reconstruct_request_timings(
    rows: List[Dict[str, Any]],
    *,
    expected_requests: int,
    output_tokens: int,
) -> Dict[str, Optional[float]]:
    """Rebuild task timing only from complete, explicitly retained fields."""
    if _exact_nonnegative_int(expected_requests) is None:
        raise ValueError("expected_requests must be an exact nonnegative integer")
    if _exact_nonnegative_int(output_tokens) is None:
        raise ValueError("output_tokens must be an exact nonnegative integer")

    rows_by_index: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        request_index = _exact_nonnegative_int(row.get("request_index"))
        if request_index is None:
            return {
                "ttft_ms": None,
                "prefill_total_s": None,
                "tpot_ms_avg": None,
            }
        if request_index in rows_by_index:
            return {
                "ttft_ms": None,
                "prefill_total_s": None,
                "tpot_ms_avg": None,
            }
        rows_by_index[request_index] = row

    expected_indexes = set(range(expected_requests))
    complete_population = set(rows_by_index) == expected_indexes
    first_row = rows_by_index.get(0)
    first_prefill = first_row.get("prefill_time_s") if first_row is not None else None
    first_prefill_s = _finite_nonnegative_number(first_prefill)
    ttft_ms = first_prefill_s * 1000.0 if first_prefill_s is not None else None

    def complete_values(key: str) -> Optional[List[float]]:
        if not complete_population or expected_requests == 0:
            return None
        values = [
            _finite_nonnegative_number(rows_by_index[index].get(key))
            for index in range(expected_requests)
        ]
        if any(value is None for value in values):
            return None
        return [value for value in values if value is not None]

    prefill_values = complete_values("prefill_time_s")
    decode_values = complete_values("decode_time_s")
    prefill_total_s = float(sum(prefill_values)) if prefill_values is not None else None
    tpot_ms_avg = (
        float(sum(decode_values)) * 1000.0 / output_tokens
        if decode_values is not None and output_tokens > 0
        else None
    )
    return {
        "ttft_ms": ttft_ms,
        "prefill_total_s": prefill_total_s,
        "tpot_ms_avg": tpot_ms_avg,
    }


def _max_input_tokens_by_task(
    path: str,
    *,
    request_counts: Dict[int, int],
    expected_input_total: int,
) -> Optional[Dict[int, int]]:
    """Return true task maxima only when request evidence fully reconciles."""
    maxima: Dict[int, int] = {}
    seen: set[Tuple[int, int]] = set()
    indexes_by_task: Dict[int, set[int]] = {}
    detailed_input_total = 0
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return None
                task_index = _exact_nonnegative_int(row.get("example_index"))
                request_index = _exact_nonnegative_int(row.get("request_index"))
                input_tokens = _exact_nonnegative_int(row.get("input_tokens"))
                if (
                    task_index is None
                    or request_index is None
                    or input_tokens is None
                    or task_index not in request_counts
                    or (task_index, request_index) in seen
                ):
                    return None
                seen.add((task_index, request_index))
                indexes_by_task.setdefault(task_index, set()).add(request_index)
                detailed_input_total += input_tokens
                maxima[task_index] = max(maxima.get(task_index, input_tokens), input_tokens)
    except OSError:
        return None
    if detailed_input_total != expected_input_total:
        return None
    for task_index, expected_requests in request_counts.items():
        if indexes_by_task.get(task_index, set()) != set(range(expected_requests)):
            return None
    return maxima


def write_metrics_file(
    results: List[Dict[str, Any]],
    wall_time_s: float,
    output_paths: Dict[str, str],
    args: Any,
    *,
    engine: str,
    engine_version: Optional[str],
) -> None:
    """Write the metrics schema shared by every standalone IMO producer."""
    total_examples = len(results)
    input_tokens = [_result_count(result, "input_tokens") for result in results]
    output_tokens = [_result_count(result, "output_tokens") for result in results]
    tool_calls = [_result_count(result, "tool_calls") for result in results]
    request_counts = [_result_count(result, "num_requests") for result in results]
    request_counts_by_task: Optional[Dict[int, int]] = {}
    result_indexes: List[int] = []
    for position, (result, request_count) in enumerate(zip(results, request_counts)):
        task_index = _exact_nonnegative_int(result.get("example_index", position))
        if task_index is None:
            request_counts_by_task = None
            break
        if task_index in request_counts_by_task:
            request_counts_by_task = None
            break
        request_counts_by_task[task_index] = request_count
        result_indexes.append(task_index)
    cached_tokens = [_result_count(result, "total_cached_tokens") for result in results]

    total_input = sum(input_tokens)
    total_output = sum(output_tokens)
    total_requests = sum(request_counts)
    total_cached = sum(cached_tokens)

    latencies_s = _numeric_values(results, "latency_ms", scale=0.001)
    ttft_s = _numeric_values(results, "ttft_ms", scale=0.001)
    tpot_s = _numeric_values(results, "tpot_ms_avg", scale=0.001)
    decode_time_s: List[float] = []
    decode_population_complete = True
    for result, result_output_tokens in zip(results, output_tokens):
        if result_output_tokens == 0:
            continue
        tpot_ms = _finite_nonnegative_number(result.get("tpot_ms_avg"))
        if tpot_ms is None:
            decode_population_complete = False
            continue
        decode_time_s.append(tpot_ms * 0.001 * result_output_tokens)
    total_decode_time = (
        float(sum(decode_time_s))
        if decode_population_complete and total_output > 0
        else None
    )

    maxima = (
        _max_input_tokens_by_task(
            output_paths.get("detailed_results_path", ""),
            request_counts=request_counts_by_task,
            expected_input_total=total_input,
        )
        if request_counts_by_task is not None
        else None
    )
    avg_task_request_max = (
        sum(float(maxima.get(index, 0)) for index in result_indexes) / total_examples
        if maxima is not None and total_examples > 0
        else None
    )

    version_key = f"{engine}_version"
    metrics = {
        "performance": {
            "e2e_s": float(wall_time_s),
            "avg_e2e_latency_s": _mean(latencies_s),
            "p50_e2e_latency_s": (
                float(statistics.median(latencies_s)) if latencies_s else None
            ),
            "p99_e2e_latency_s": _p99(latencies_s),
            "examples_per_second": (
                float(total_examples) / wall_time_s if wall_time_s > 0 else 0.0
            ),
            "ttft": _mean(ttft_s),
            "p99_ttft": _p99(ttft_s),
            "tpot": _mean(tpot_s),
            "p99_tpot": _p99(tpot_s),
            "decode_time_s": total_decode_time,
            "p99_decode_time_s": (
                _p99(decode_time_s) if decode_population_complete else None
            ),
            "output_throughput_tok_s": (
                float(total_output) / total_decode_time
                if total_decode_time is not None and total_decode_time > 0
                else None
            ),
        },
        "agentic": {
            "avg_total_input_tokens": _mean([float(value) for value in input_tokens]),
            "avg_total_output_tokens": _mean([float(value) for value in output_tokens]),
            "avg_tool_call_count": _mean([float(value) for value in tool_calls]),
            "avg_num_requests": _mean([float(value) for value in request_counts]),
            "avg_input_tokens_per_request": (
                float(total_input) / total_requests if total_requests > 0 else None
            ),
            "avg_output_tokens_per_request": (
                float(total_output) / total_requests if total_requests > 0 else None
            ),
            "avg_max_input_tokens_per_request": (
                avg_task_request_max
            ),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cached_tokens": total_cached,
            "avg_cache_hit_rate": (
                float(total_cached) / total_input if total_input > 0 else 0.0
            ),
            "total_requests": total_requests,
            "total_tool_calls": sum(tool_calls),
        },
        "quality": {
            "acc": (
                sum(float(result.get("score") or 0.0) for result in results) / total_examples
                if total_examples
                else 0.0
            ),
            "claim_coverage": "",
            "eval_judge": args.judge_model,
        },
        "hardware": {
            "gpu_type": os.getenv("GPU_TYPE", "unknown"),
            "num_gpus": int(os.getenv("NUM_GPUS", str(args.tensor_parallel_size))),
            version_key: engine_version,
            "avg_gpu_utilization_pct": "",
            "peak_gpu_memory_used_mb": "",
            "avg_cpu_utilization_pct": "",
        },
    }

    with Path(output_paths["metrics_path"]).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4)
    print(f"Wrote metrics file: {output_paths['metrics_path']}")


def _dry_run_paths(output_dir: str) -> Dict[str, str]:
    root = Path(output_dir)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"dry-run output directory already exists; refusing to overwrite: {root}"
        ) from exc
    return {
        "results_dir": str(root),
        "metadata_path": str(root / "metadata_imo-answerbench_dry-run.json"),
        "metrics_path": str(root / "metrics_imo-answerbench_dry-run.json"),
        "detailed_results_path": str(root / "detailed-results_imo-answerbench_dry-run.jsonl"),
        "output_data_path": str(root / "output-data_imo-answerbench_dry-run.jsonl"),
    }


def run_producer_dry_run(
    args: Any,
    *,
    engine: str,
    engine_version: Optional[str],
) -> Dict[str, Any]:
    """Exercise the producer data path without a model, server, judge, or GPU."""
    if not args.dry_run_output_dir:
        raise RuntimeError("--dry-run-output-dir is required for a producer dry run")
    paths = _dry_run_paths(args.dry_run_output_dir)
    Path(paths["detailed_results_path"]).touch()
    Path(paths["output_data_path"]).touch()

    version_key = f"{engine}_version"
    metadata = {
        "hardware": {"gpu_type": "synthetic", "num_gpus": 0},
        "model_config": {"model_name": "synthetic-dry-run", "precision": None},
        "system_environment": {
            "inference_engine": engine,
            version_key: engine_version,
            "dataset": "imo_answerbench",
            "num_examples": 8,
            "concurrency": args.concurrency,
            "dry_run": True,
            "publishable": False,
            "dry_run_scope": "producer-control-and-artifact-path-only",
        },
    }
    with Path(paths["metadata_path"]).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    tasks = [
        SimpleNamespace(
            id=f"dry-run-{index}",
            name=f"Synthetic dry-run task {index}",
            category="dry-run",
            eval_config={"expected": str(index)},
        )
        for index in range(8)
    ]

    async def run_one(index: int, task: Any) -> Dict[str, Any]:
        # Keep a tracked synthetic stream open while workers overlap. Descending
        # delays deliberately make completion (and therefore writes) out of order.
        stream = tracked_stream(lambda: iter(()))
        try:
            delay_rank = args.concurrency - (index % args.concurrency)
            await asyncio.sleep(0.02 * delay_rank)
        finally:
            stream.close()
        first_input = 10 + index
        second_input = 20 + index
        detailed_rows = [
            {
                "example_index": index,
                "request_index": 0,
                "input_tokens": first_input,
                "output_tokens": 2,
                "prefill_time_s": 0.01,
                "decode_time_s": 0.02,
            },
            {
                "example_index": index,
                "request_index": 1,
                "input_tokens": second_input,
                "output_tokens": 3,
                "prefill_time_s": 0.01,
                "decode_time_s": 0.03,
            },
        ]
        return {
            "task_id": task.id,
            "task_name": task.name,
            "category": task.category,
            "expected": str(index),
            "predicted": str(index),
            "score": 1.0,
            "correct": True,
            "response": str(index),
            "reasoning": "",
            "tool_calls": 1,
            "num_requests": 2,
            "tool_latencies_ms": [],
            "input_tokens": first_input + second_input,
            "output_tokens": 5,
            "latency_ms": 50.0,
            "ttft_ms": 10.0,
            "prefill_total_s": 0.02,
            "tpot_ms_avg": 10.0,
            "tpot_ms_p99": 10.0,
            "errors": [],
            "judge_equivalent": True,
            "judge_response": "synthetic dry-run",
            "judge_status_code": None,
            "judge_attempts": 0,
            "detailed_rows": detailed_rows,
            "total_cached_tokens": 0,
            "finish_reason": "stop",
        }

    def persist_one(index: int, result: Dict[str, Any]) -> None:
        with Path(paths["detailed_results_path"]).open("a", encoding="utf-8") as handle:
            for row in result["detailed_rows"]:
                handle.write(json.dumps(row) + "\n")
        output_row = {
            "index": index,
            "task_id": result["task_id"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "tool_call_count": result["tool_calls"],
            "num_requests": result["num_requests"],
            "e2e_latency_s": result["latency_ms"] / 1000.0,
            "output_text": result["response"],
            "errors": result["errors"],
            "eval_passed": result["judge_equivalent"],
            "eval_score": result["score"],
            "eval_details": result["judge_response"],
        }
        with Path(paths["output_data_path"]).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(output_row) + "\n")

    LLM_REQUEST_CONCURRENCY.reset()
    started = time.monotonic()
    results, observed_tasks = asyncio.run(
        run_tasks_concurrently(tasks, args.concurrency, run_one, persist_one)
    )
    wall_time_s = time.monotonic() - started
    observed_requests = LLM_REQUEST_CONCURRENCY.maximum
    update_metadata_concurrency(
        paths["metadata_path"],
        concurrency=args.concurrency,
        observed_max_concurrency=observed_tasks,
        observed_max_simultaneous_llm_requests=observed_requests,
    )
    write_metrics_file(
        results,
        wall_time_s,
        paths,
        args,
        engine=engine,
        engine_version=engine_version,
    )
    return {
        "paths": paths,
        "observed_max_concurrency": observed_tasks,
        "observed_max_simultaneous_llm_requests": observed_requests,
        "num_results": len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated, non-publishable IMO producer control-path dry run."
    )
    parser.add_argument("--engine", required=True, choices=["sglang", "vllm"])
    parser.add_argument("--engine-version", default=None)
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--dry-run-output-dir", required=True)
    parser.add_argument("--judge-model", default="synthetic-dry-run")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    report = run_producer_dry_run(
        args,
        engine=args.engine,
        engine_version=args.engine_version,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
