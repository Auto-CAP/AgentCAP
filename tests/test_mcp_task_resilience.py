import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_cap.agents.cli import (
    DEFAULT_MCP_TASK_TIMEOUT_S,
    RUNTIME_OBSERVATION_FILENAME,
    _compact_resume_results,
    _load_resume,
    _partial_failure_result,
    _require_complete_results,
    _resolve_task_timeout,
    _run_with_timeout,
    _serialize_result,
    _uses_mcp_backend,
    _write_runtime_observation,
)
from agent_cap.agents.metrics import aggregate_agent_metrics
from agent_cap.agents.types import Task, TurnRecord, Usage


def _args(**overrides):
    values = {"task_timeout": None, "tool_backend": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_mcp_tasks_have_a_default_outer_deadline():
    assert _resolve_task_timeout(_args(tool_backend="mcp"), {}) == DEFAULT_MCP_TASK_TIMEOUT_S
    assert _resolve_task_timeout(_args(), {}) is None
    assert _resolve_task_timeout(_args(), {"tool_backend": "mcp-atlas", "task_timeout": 9}) == 9


def test_hung_task_is_cancelled_at_its_deadline():
    async def hang():
        await asyncio.Event().wait()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run_with_timeout(hang(), 0.01))


def test_timeout_row_preserves_elapsed_partial_usage_and_turns():
    usage = Usage(input_tokens=17, output_tokens=5, requests=1)
    turn = TurnRecord(
        role="agent",
        model="model",
        messages_in=[],
        assistant={"role": "assistant", "content": "partial answer"},
        usage=usage,
        latency_s=0.25,
        ttft_s=0.05,
        decode_s=0.20,
    )
    agent = SimpleNamespace(
        state=SimpleNamespace(usage=usage, turns=[turn]),
        final_text=lambda: "partial answer",
    )
    result = _partial_failure_result(
        task=Task(task_id="timed-out", user_prompt="prompt"),
        strategy_name="single",
        agents={"agent": agent},
        elapsed_s=3.5,
        exc=asyncio.TimeoutError(),
        timeout_s=3.0,
    )

    row = _serialize_result(result, verbose=0)

    assert row["e2e_latency_s"] == 3.5
    assert row["num_turns"] == 1
    assert row["total_usage"] == {
        "input_tokens": 17,
        "output_tokens": 5,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "requests": 1,
    }
    assert row["output_text"] == "partial answer"
    assert row["errors"] == ["TimeoutError: task exceeded 3s deadline"]
    assert row["turn_stats"][0]["input_tokens"] == 17
    metrics = aggregate_agent_metrics(
        [row], wall_time_s=3.5, hardware_info={}
    )
    assert metrics["performance"]["avg_e2e_latency_s"] == 3.5
    assert metrics["agentic"]["total_input_tokens"] == 17
    assert metrics["agentic"]["total_output_tokens"] == 5
    assert metrics["agentic"]["total_requests"] == 1


def test_resume_error_policy_is_backend_specific(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [
        {"task_id": "complete", "errors": [], "output_text": "done"},
        {"task_id": "retry", "errors": [], "output_text": "old"},
        {"task_id": "retry", "errors": ["TimeoutError: deadline"], "output_text": ""},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    ordered = [Task(task_id=name, user_prompt=name) for name in ("complete", "retry", "missing")]
    swe_done = _load_resume(
        path, retry_errors=_uses_mcp_backend(_args(tool_backend="local"), {})
    )
    mcp_done = _load_resume(
        path, retry_errors=_uses_mcp_backend(_args(tool_backend="mcp"), {})
    )

    assert list(swe_done) == ["complete", "retry"]
    assert swe_done["retry"]["errors"]
    assert [task.task_id for task in ordered if task.task_id not in swe_done] == ["missing"]
    assert list(mcp_done) == ["complete"]
    assert [task.task_id for task in ordered if task.task_id not in mcp_done] == ["retry", "missing"]


def test_resume_compaction_is_atomic_and_task_ordered(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"task_id":"stale"}\n', encoding="utf-8")
    tasks = [Task(task_id=name, user_prompt=name) for name in ("b", "a", "missing")]
    done = {"a": {"task_id": "a"}, "b": {"task_id": "b"}}

    with patch("agent_cap.agents.cli.os.fsync", wraps=os.fsync) as fsync:
        _compact_resume_results(path, tasks, done)
    assert fsync.called
    assert [json.loads(line)["task_id"] for line in path.read_text().splitlines()] == ["b", "a"]

    before = path.read_text()
    with patch.object(Path, "replace", side_effect=OSError("interrupted")):
        with pytest.raises(OSError, match="interrupted"):
            _compact_resume_results(path, tasks[:1], done)
    assert path.read_text() == before
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_missing_result_slots_fail_closed():
    tasks = [Task(task_id="a", user_prompt="a"), Task(task_id="b", user_prompt="b")]
    with pytest.raises(RuntimeError, match="without result rows for: b"):
        _require_complete_results(tasks, [{"task_id": "a"}, None])


def test_runtime_observation_survives_interruption_and_resume(tmp_path):
    path = _write_runtime_observation(
        tmp_path,
        requested_task_concurrency=4,
        observed_max_task_concurrency=0,
        resume=False,
    )
    assert path.name == RUNTIME_OBSERVATION_FILENAME

    _write_runtime_observation(
        tmp_path,
        requested_task_concurrency=4,
        observed_max_task_concurrency=4,
        resume=True,
    )
    _write_runtime_observation(
        tmp_path,
        requested_task_concurrency=4,
        observed_max_task_concurrency=0,
        resume=True,
    )

    assert json.loads(path.read_text()) == {
        "schema_version": 1,
        "publishable": False,
        "requested_task_concurrency": 4,
        "observed_max_task_concurrency": 4,
    }
