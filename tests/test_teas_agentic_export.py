import json

import pytest

from agent_cap.agents.metrics import aggregate_agent_metrics
from agent_cap.agents.teas_output import (
    TeasOutputError,
    export_teas_leaf,
    write_teas_outputs,
)
from scripts.export_teas_agentic_run import _canonical_rows


def _row(task_id: str, *, placeholder_passed: bool = False):
    return {
        "task_id": task_id,
        "strategy": "sweagent",
        "e2e_latency_s": 2.0,
        "eval_passed": placeholder_passed,
        "eval_score": float(placeholder_passed),
        "eval_details": {},
        "output_text": "patch",
        "errors": [],
        "tool_calls": 1,
        "total_usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "completion_tokens": 3,
            "reasoning_tokens": 1,
            "cached_tokens": 2,
            "requests": 1,
        },
        "turn_stats": [
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "completion_tokens": 3,
                "reasoning_tokens": 1,
                "cached_tokens": 2,
                "ttft_s": 0.1,
                "decode_s": 0.2,
                "num_tool_calls": 1,
            }
        ],
    }


def _write_report(run_dir, run_id, task_id, *, resolved):
    report_dir = run_dir / "logs" / "run_evaluation" / run_id / "agentcap-unified" / task_id
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps({task_id: {"resolved": resolved}}),
        encoding="utf-8",
    )


def _complete_env(monkeypatch):
    values = {
        "TEAS_ENGINE": "vllm",
        "TEAS_ENGINE_VERSION": "0.26.0",
        "TEAS_GPU_TYPE": "NVIDIA B300",
        "TEAS_NUM_GPUS": "1",
        "TEAS_CPU_TYPE": "Example CPU",
        "TEAS_NUM_CPUS": "32",
        "TEAS_TP": "1",
        "TEAS_MAX_MODEL_LEN": "131072",
        "TEAS_MODEL_NAME": "unsloth/gpt-oss-120b",
        "TEAS_PRECISION": "mxfp4",
        "TEAS_BASE_URL": "http://127.0.0.1:8000/v1",
        "TEAS_BACKEND": "swebench-modal",
        "TEAS_CONCURRENCY": "4",
        "TEAS_OBSERVED_MAX_CONCURRENCY": "4",
        "TEAS_OBSERVED_MAX_LLM_CONCURRENCY": "4",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_swebench_official_reports_replace_generation_zero(monkeypatch, tmp_path):
    _complete_env(monkeypatch)
    rows = [_row("task-a"), _row("task-b")]
    _write_report(tmp_path, "official", "task-a", resolved=True)
    _write_report(tmp_path, "official", "task-b", resolved=False)

    metrics_path = write_teas_outputs(
        tmp_path,
        rows,
        "swe-bench-lite",
        120.0,
        timestamp="20260731_000000",
    )
    quality = json.loads(metrics_path.read_text())["quality"]

    assert quality == {"acc": 0.5, "total_examples": 2, "passed": 1}


def test_swebench_missing_official_quality_refuses_placeholder_zero(monkeypatch, tmp_path):
    _complete_env(monkeypatch)

    with pytest.raises(TeasOutputError, match="placeholder zeros"):
        write_teas_outputs(
            tmp_path,
            [_row("task-a")],
            "swe-bench-lite",
            120.0,
            timestamp="20260731_000000",
        )

    assert not list(tmp_path.glob("metrics_swe-bench-lite_*.json"))


def test_swebench_legitimate_official_zero_is_allowed(monkeypatch, tmp_path):
    _complete_env(monkeypatch)
    _write_report(tmp_path, "official", "task-a", resolved=False)

    metrics_path = write_teas_outputs(
        tmp_path,
        [_row("task-a", placeholder_passed=True)],
        "swe-bench-lite",
        120.0,
        timestamp="20260731_000000",
    )

    assert json.loads(metrics_path.read_text())["quality"] == {
        "acc": 0.0,
        "total_examples": 1,
        "passed": 0,
    }


def test_conflicting_retry_reports_are_rejected(monkeypatch, tmp_path):
    _complete_env(monkeypatch)
    _write_report(tmp_path, "official-a", "task-a", resolved=True)
    _write_report(tmp_path, "official-b", "task-a", resolved=False)

    with pytest.raises(TeasOutputError, match="conflicting SWE-bench reports"):
        write_teas_outputs(
            tmp_path,
            [_row("task-a")],
            "swe-bench-lite",
            120.0,
            timestamp="20260731_000000",
        )


def test_partial_reports_do_not_silently_turn_pending_rows_into_failures(monkeypatch, tmp_path):
    _complete_env(monkeypatch)
    _write_report(tmp_path, "official", "task-a", resolved=True)

    with pytest.raises(TeasOutputError, match="evaluation is incomplete"):
        write_teas_outputs(
            tmp_path,
            [_row("task-a"), _row("task-b")],
            "swe-bench-lite",
            120.0,
            timestamp="20260731_000000",
        )


def test_logged_harness_failure_is_counted_as_official_failure(monkeypatch, tmp_path):
    _complete_env(monkeypatch)
    _write_report(tmp_path, "official", "task-a", resolved=True)
    (tmp_path / "swebench_eval.log").write_text(
        "swebench.harness.utils.EvaluationError: task-b: Error creating sandbox\n",
        encoding="utf-8",
    )

    metrics_path = write_teas_outputs(
        tmp_path,
        [_row("task-a"), _row("task-b")],
        "swe-bench-lite",
        120.0,
        timestamp="20260731_000000",
    )

    assert json.loads(metrics_path.read_text())["quality"] == {
        "acc": 0.5,
        "total_examples": 2,
        "passed": 1,
    }


def test_unverified_generic_swebench_metrics_mark_quality_pending():
    metrics = aggregate_agent_metrics(
        [_row("task-a")],
        wall_time_s=1.0,
        evaluator_name="swebench",
        hardware_info={},
    )

    assert metrics["quality"] == {
        "acc": None,
        "task_coverage": None,
        "evaluator": "swebench",
    }


def test_mcp_metrics_include_task_level_e2e_latencies(monkeypatch, tmp_path):
    _complete_env(monkeypatch)
    rows = [_row("task-a"), _row("task-b")]
    rows[0]["e2e_latency_s"] = 2.0
    rows[1]["e2e_latency_s"] = 4.0
    for row in rows:
        row["eval_details"] = {
            "evaluator": "example-judge",
            "coverage_score": row["eval_score"],
        }

    metrics_path = write_teas_outputs(
        tmp_path,
        rows,
        "mcp-atlas",
        12.0,
        timestamp="20260731_000000",
    )

    assert json.loads(metrics_path.read_text())["performance"] == {
        "total_wall_time_min": 0.2,
        "avg_e2e_latency_s": 3.0,
        "p50_e2e_latency_s": 3.0,
        "p99_e2e_latency_s": 3.98,
        "ttft": 0.1,
        "p99_ttft": 0.1,
        "tpot": 0.05,
        "p99_tpot": 0.05,
    }


def test_mcp_quality_uses_mean_coverage_score_not_pass_rate(
    monkeypatch,
    tmp_path,
):
    _complete_env(monkeypatch)
    rows = [_row("task-a", placeholder_passed=True), _row("task-b")]
    rows[0]["eval_score"] = 1.0
    rows[1]["eval_score"] = 0.4
    for row in rows:
        row["eval_details"] = {
            "evaluator": "example-judge",
            "coverage_score": row["eval_score"],
        }

    metrics_path = write_teas_outputs(
        tmp_path,
        rows,
        "mcp-atlas",
        12.0,
        timestamp="20260731_000000",
    )

    assert json.loads(metrics_path.read_text())["quality"] == {
        "acc": 0.7,
        "total_examples": 2,
        "passed": 1,
    }


def test_canonical_output_data_restores_order_after_concurrent_completion(
    tmp_path,
):
    rows = [_row("task-b"), _row("task-a")]
    order_path = tmp_path / "output-data_swe-bench-lite_20260731_000000.jsonl"
    order_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "task-a"}),
                json.dumps({"task_id": "task-b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ordered = _canonical_rows(
        tmp_path,
        rows,
        "swe-bench-lite",
        "20260731_000000",
    )

    assert [row["task_id"] for row in ordered] == ["task-a", "task-b"]


def test_export_creates_exactly_the_current_four_file_leaf(monkeypatch, tmp_path):
    _complete_env(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    rows = [_row("task-a")]
    _write_report(source, "official", "task-a", resolved=True)
    run_script = source / "run.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\n"
        "agent-cap --engine vllm --model unsloth/gpt-oss-120b "
        "--dataset swe-bench-lite --concurrency 4\n",
        encoding="utf-8",
    )
    destination = tmp_path / "leaf"

    export_teas_leaf(
        source,
        destination,
        rows,
        "swe-bench-lite",
        120.0,
        run_script=run_script,
        timestamp="20260731_000000",
    )

    assert sorted(path.name for path in destination.iterdir()) == [
        "detailed-results_swe-bench-lite_20260731_000000.jsonl",
        "metadata_swe-bench-lite_20260731_000000.json",
        "metrics_swe-bench-lite_20260731_000000.json",
        "run.sh",
    ]
    quality = json.loads((destination / "metrics_swe-bench-lite_20260731_000000.json").read_text())[
        "quality"
    ]
    assert list(quality) == ["acc", "total_examples", "passed"]
