import argparse
import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_cap.imo_output import (
    ActivityTracker,
    LLM_REQUEST_CONCURRENCY,
    positive_int,
    reconstruct_request_timings,
    run_producer_dry_run,
    run_tasks_concurrently,
    tracked_stream,
    update_metadata_concurrency,
    write_metrics_file,
)


def _task(index):
    return SimpleNamespace(
        id=f"task-{index}",
        name=f"Task {index}",
        category="test",
        eval_config={"expected": str(index)},
    )


def _result(task, *, latency_ms=10.0):
    return {
        "task_id": task.id,
        "task_name": task.name,
        "category": task.category,
        "score": 1.0,
        "correct": True,
        "response": "ok",
        "tool_calls": 0,
        "num_requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": latency_ms,
        "ttft_ms": 1.0,
        "tpot_ms_avg": 2.0,
        "errors": [],
        "judge_equivalent": True,
        "judge_response": "ok",
        "detailed_rows": [],
    }


def test_positive_int_rejects_zero_and_negative():
    assert positive_int("3") == 3
    with pytest.raises(ValueError):
        positive_int("0")
    with pytest.raises(ValueError):
        positive_int("-1")


def test_all_imo_forks_use_shared_metrics_and_deepseek_forks_expose_concurrency():
    repo = Path(__file__).resolve().parents[1]
    scripts = [
        repo / "agent_cap" / "run_imo_answerbench_4.py",
        repo / "agent_cap" / "run_imo_answerbench_5.py",
        repo / "agent_cap" / "run_imo_answerbench_deepseek32_sglang.py",
        repo / "agent_cap" / "run_imo_answerbench_deepseek32_vllm.py",
    ]
    for script in scripts:
        source = script.read_text()
        assert "write_metrics_file as write_shared_metrics_file" in source
        assert "def write_metrics_file(" not in source
        assert "resolve_precision" in source
        assert "def infer_model_precision(" not in source

    for script in scripts[2:]:
        source = script.read_text()
        assert 'add_argument("--concurrency", type=positive_int, default=1)' in source
        assert "run_tasks_concurrently(" in source
        assert "tracked_stream(" in source
        assert "update_metadata_concurrency(" in source

    resume_source = scripts[1].read_text()
    assert "build_failed_task_result," in resume_source
    assert "def build_failed_task_result(" not in resume_source
    assert "reconstruct_request_timings(" in resume_source


def test_scheduler_caps_overlap_serializes_writes_and_captures_exceptions():
    tasks = [_task(index) for index in range(5)]
    active = ActivityTracker()
    writes = []

    async def run_one(index, task):
        with active.active():
            await asyncio.sleep(0.01 * (5 - index))
            if index == 2:
                raise RuntimeError("synthetic failure")
            return _result(task)

    def persist_one(index, result):
        writes.append((index, result["task_id"]))

    results, observed = asyncio.run(
        run_tasks_concurrently(tasks, 2, run_one, persist_one)
    )

    assert observed == 2
    assert active.maximum == 2
    assert [result["task_id"] for result in results] == [task.id for task in tasks]
    assert sorted(index for index, _ in writes) == list(range(5))
    assert len({index for index, _ in writes}) == 5
    failed = results[2]
    assert failed["finish_reason"] == "unhandled_task_exception"
    assert failed["num_requests"] == 0
    assert failed["ttft_ms"] is None
    assert failed["tpot_ms_avg"] is None
    assert "RuntimeError: synthetic failure" in failed["errors"][0]


class _BlockingStream:
    def __init__(self, ready, release):
        self.ready = ready
        self.release = release

    def __iter__(self):
        self.ready.set()
        self.release.wait(timeout=2)
        return iter(())

    def close(self):
        return None


def test_tracked_stream_counts_the_full_stream_lifetime():
    LLM_REQUEST_CONCURRENCY.reset()
    release = threading.Event()
    ready = [threading.Event(), threading.Event()]

    def consume(index):
        stream = tracked_stream(lambda: _BlockingStream(ready[index], release))
        try:
            list(stream)
        finally:
            stream.close()

    threads = [threading.Thread(target=consume, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    assert all(event.wait(timeout=2) for event in ready)
    release.set()
    for thread in threads:
        thread.join(timeout=2)
    assert LLM_REQUEST_CONCURRENCY.maximum == 2


def _metrics_result(*, input_tokens, output_tokens, num_requests, score=1.0):
    return {
        "latency_ms": 100.0,
        "ttft_ms": 10.0,
        "tpot_ms_avg": 2.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_calls": 0,
        "num_requests": num_requests,
        "score": score,
        "total_cached_tokens": 0,
    }


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_metrics(tmp_path, results, rows):
    details = tmp_path / "detailed.jsonl"
    metrics = tmp_path / "metrics.json"
    _write_rows(details, rows)
    args = SimpleNamespace(
        judge_model="judge",
        tensor_parallel_size=1,
    )
    write_metrics_file(
        results,
        1.0,
        {"detailed_results_path": str(details), "metrics_path": str(metrics)},
        args,
        engine="vllm",
        engine_version="test",
    )
    return json.loads(metrics.read_text())


def test_metrics_use_global_request_denominator_and_reconciled_task_maxima(tmp_path):
    results = [
        _metrics_result(input_tokens=40, output_tokens=8, num_requests=2),
        _metrics_result(input_tokens=20, output_tokens=4, num_requests=1),
        _metrics_result(input_tokens=0, output_tokens=0, num_requests=0, score=0.0),
    ]
    rows = [
        {"example_index": 0, "request_index": 0, "input_tokens": 10},
        {"example_index": 0, "request_index": 1, "input_tokens": 30},
        {"example_index": 1, "request_index": 0, "input_tokens": 20},
    ]
    metrics = _write_metrics(tmp_path, results, rows)
    agentic = metrics["agentic"]
    assert agentic["avg_input_tokens_per_request"] == 20.0
    assert agentic["avg_output_tokens_per_request"] == 4.0
    # D106: (max(10,30) + max(20) + zero-request example 0) / 3.
    assert agentic["avg_max_input_tokens_per_request"] == pytest.approx(50 / 3)


def test_metrics_use_explicit_task_indexes_after_non_prefix_resume(tmp_path):
    results = [
        {
            **_metrics_result(input_tokens=63, output_tokens=6, num_requests=3),
            "example_index": 2,
        },
        {
            **_metrics_result(input_tokens=10, output_tokens=2, num_requests=1),
            "example_index": 0,
        },
        {
            **_metrics_result(input_tokens=52, output_tokens=4, num_requests=2),
            "example_index": 1,
        },
    ]
    rows = [
        {"example_index": 0, "request_index": 0, "input_tokens": 10},
        {"example_index": 1, "request_index": 0, "input_tokens": 20},
        {"example_index": 1, "request_index": 1, "input_tokens": 32},
        {"example_index": 2, "request_index": 0, "input_tokens": 21},
        {"example_index": 2, "request_index": 1, "input_tokens": 21},
        {"example_index": 2, "request_index": 2, "input_tokens": 21},
    ]
    metrics = _write_metrics(tmp_path, results, rows)
    assert metrics["agentic"]["avg_max_input_tokens_per_request"] == 21.0


def test_resume_timing_reconstruction_preserves_missing_as_null():
    complete = reconstruct_request_timings(
        [
            {"request_index": 0, "prefill_time_s": 0.2, "decode_time_s": 0.3},
            {"request_index": 1, "prefill_time_s": 0.4, "decode_time_s": 0.5},
        ],
        expected_requests=2,
        output_tokens=4,
    )
    assert complete == {
        "ttft_ms": 200.0,
        "prefill_total_s": pytest.approx(0.6),
        "tpot_ms_avg": 200.0,
    }

    missing = reconstruct_request_timings(
        [{"request_index": 0}],
        expected_requests=1,
        output_tokens=0,
    )
    assert missing == {
        "ttft_ms": None,
        "prefill_total_s": None,
        "tpot_ms_avg": None,
    }


def test_resume_timing_fields_fail_independently():
    missing_prefill = reconstruct_request_timings(
        [
            {"request_index": 0, "decode_time_s": 0.1},
            {"request_index": 1, "prefill_time_s": 0.2, "decode_time_s": 0.3},
        ],
        expected_requests=2,
        output_tokens=4,
    )
    assert missing_prefill == {
        "ttft_ms": None,
        "prefill_total_s": None,
        "tpot_ms_avg": pytest.approx(100.0),
    }

    missing_decode = reconstruct_request_timings(
        [
            {"request_index": 0, "prefill_time_s": 0.1, "decode_time_s": 0.1},
            {"request_index": 1, "prefill_time_s": 0.2},
        ],
        expected_requests=2,
        output_tokens=4,
    )
    assert missing_decode == {
        "ttft_ms": 100.0,
        "prefill_total_s": pytest.approx(0.3),
        "tpot_ms_avg": None,
    }


@pytest.mark.parametrize("invalid_timing", ["NaN", float("nan"), -0.1, True])
def test_resume_timing_rejects_nonfinite_negative_and_boolean_values(invalid_timing):
    timings = reconstruct_request_timings(
        [
            {
                "request_index": 0,
                "prefill_time_s": invalid_timing,
                "decode_time_s": invalid_timing,
            }
        ],
        expected_requests=1,
        output_tokens=1,
    )
    assert timings == {
        "ttft_ms": None,
        "prefill_total_s": None,
        "tpot_ms_avg": None,
    }


def test_strict_max_rejects_fractional_indexes_and_token_counts(tmp_path):
    result = {
        **_metrics_result(input_tokens=10, output_tokens=1, num_requests=1),
        "example_index": 0.5,
    }
    rows = [{"example_index": 0.5, "request_index": 0.5, "input_tokens": 10.9}]
    metrics = _write_metrics(tmp_path, [result], rows)
    assert metrics["agentic"]["avg_max_input_tokens_per_request"] is None

    timings = reconstruct_request_timings(
        [{"request_index": 0.5, "prefill_time_s": 0.1, "decode_time_s": 0.2}],
        expected_requests=1,
        output_tokens=1,
    )
    assert timings == {
        "ttft_ms": None,
        "prefill_total_s": None,
        "tpot_ms_avg": None,
    }


def test_decode_throughput_is_null_when_any_output_timing_is_missing(tmp_path):
    results = [
        _metrics_result(input_tokens=0, output_tokens=100, num_requests=0),
        {
            **_metrics_result(input_tokens=0, output_tokens=100, num_requests=0),
            "tpot_ms_avg": None,
        },
    ]
    metrics = _write_metrics(tmp_path, results, [])
    performance = metrics["performance"]
    assert performance["decode_time_s"] is None
    assert performance["p99_decode_time_s"] is None
    assert performance["output_throughput_tok_s"] is None


@pytest.mark.parametrize(
    "rows",
    [
        # Truncated request sequence.
        [{"example_index": 0, "request_index": 0, "input_tokens": 40}],
        # Duplicate request identity.
        [
            {"example_index": 0, "request_index": 0, "input_tokens": 20},
            {"example_index": 0, "request_index": 0, "input_tokens": 20},
        ],
        # Reconciles neither request population nor aggregate input total.
        [
            {"example_index": 0, "request_index": 0, "input_tokens": 10},
            {"example_index": 0, "request_index": 1, "input_tokens": 20},
        ],
    ],
)
def test_metrics_null_max_when_detailed_evidence_is_not_complete(tmp_path, rows):
    results = [_metrics_result(input_tokens=40, output_tokens=8, num_requests=2)]
    metrics = _write_metrics(tmp_path, results, rows)
    assert metrics["agentic"]["avg_max_input_tokens_per_request"] is None


def test_metadata_concurrency_fields_are_finalized(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"system_environment": {"dataset": "imo_answerbench"}}))
    update_metadata_concurrency(
        str(path),
        concurrency=4,
        observed_max_concurrency=4,
        observed_max_simultaneous_llm_requests=3,
    )
    environment = json.loads(path.read_text())["system_environment"]
    assert environment["concurrency"] == 4
    assert environment["observed_max_concurrency"] == 4
    assert environment["observed_max_simultaneous_llm_requests"] == 3


def test_isolated_dry_run_writes_nonpublishable_reconciled_artifacts(tmp_path):
    output_dir = tmp_path / "dry-run"
    args = argparse.Namespace(
        dry_run_output_dir=str(output_dir),
        concurrency=4,
        judge_model="synthetic",
        tensor_parallel_size=1,
    )
    report = run_producer_dry_run(args, engine="sglang", engine_version="test")
    assert report["observed_max_concurrency"] == 4
    assert report["observed_max_simultaneous_llm_requests"] == 4
    assert report["num_results"] == 8

    metadata = json.loads(Path(report["paths"]["metadata_path"]).read_text())
    environment = metadata["system_environment"]
    assert environment["dry_run"] is True
    assert environment["publishable"] is False
    assert environment["observed_max_concurrency"] == 4

    output_rows = [
        json.loads(line)
        for line in Path(report["paths"]["output_data_path"]).read_text().splitlines()
    ]
    assert len(output_rows) == 8
    assert sorted(row["index"] for row in output_rows) == list(range(8))
    assert len({row["task_id"] for row in output_rows}) == 8

    metrics = json.loads(Path(report["paths"]["metrics_path"]).read_text())
    assert metrics["agentic"]["total_requests"] == 16
    assert metrics["agentic"]["avg_max_input_tokens_per_request"] == 23.5

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        run_producer_dry_run(args, engine="sglang", engine_version="test")
