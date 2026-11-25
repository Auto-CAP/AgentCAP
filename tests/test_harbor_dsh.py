"""Tests for the DeepSeek Harness Harbor agent (agent_cap.harbor.dsh)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

if importlib.util.find_spec("harbor") is None:
    pytest.skip("harbor not installed", allow_module_level=True)

from agent_cap.harbor import cli, jobspec  # noqa: E402
from agent_cap.harbor.dsh import (  # noqa: E402
    KEY_ENV_IN_CONTAINER,
    TEAM_BUNDLE,
    DSHAgent,
    build_profile,
    final_text_from_stdout,
    read_sessions,
    sessions_to_row,
)


def _ids(patches):
    return [p["id"] for p in patches]


def test_profile_single_and_team():
    single = build_profile(model="m", base_url="http://h/v1", protocol="openai-completions",
                           teammates=0, sessions_root="/logs/agent/dsh-home/sessions")
    assert TEAM_BUNDLE not in single["package.json"]["dsh"]["profile"]["bundles"]
    assert "agent-team" not in _ids(single["cordis.patch.yml"])
    team = build_profile(model="m", base_url="http://h/v1", protocol="openai-completions",
                         teammates=3, sessions_root="/s", reasoning_effort="high")
    patches = {p["id"]: p for p in team["cordis.patch.yml"]}
    assert TEAM_BUNDLE in team["package.json"]["dsh"]["profile"]["bundles"]
    assert patches["agent-team"]["config"]["maxMembers"] == 3
    assert patches["subagent"]["config"]["maxActiveSubagents"] == 3
    assert patches["agent-default-model"]["config"] == {
        "provider": "agentcard-endpoint", "model": "m", "reasoningEffort": "high"}
    route = patches["llm-pi-ai"]["config"]["providers"]["agentcard-endpoint"]
    assert route["apiKeyEnv"] == KEY_ENV_IN_CONTAINER and "apiKey" not in json.dumps(route).replace("apiKeyEnv", "")
    assert patches["llm-deepseek"]["disabled"] is True
    assert patches["tool-web"]["disabled"] is True


def test_profile_native_deepseek_route():
    files = build_profile(model="deepseek-chat", base_url="https://api.deepseek.com", protocol="deepseek-messages",
                          teammates=0, sessions_root="/s")
    patches = {p["id"]: p for p in files["cordis.patch.yml"]}
    assert patches["agent-default-model"]["config"]["provider"] == "deepseek-official"
    assert patches["llm-deepseek"]["config"]["baseURL"] == "https://api.deepseek.com"
    assert "llm-pi-ai" not in patches


def test_profile_rejects_bad_settings():
    with pytest.raises(ValueError):
        build_profile(model="m", base_url="u", protocol="grpc", teammates=0, sessions_root="/s")
    with pytest.raises(ValueError):
        build_profile(model="m", base_url="u", protocol="openai-completions", teammates=0,
                      sessions_root="/s", context_window=1000, max_tokens=2000)


def _write_session(path: Path, header: dict, events: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in [header, *events]) + "\n")


def _request(t0: int, inp: int, out: int, n_tools: int = 0):
    blocks = [{"type": "tool-call"}] * n_tools or [{"type": "text"}]
    return [
        {"type": "request/header", "time": t0, "data": {"header": {}}},
        {"type": "assistant/message", "time": t0 + 300, "data": {
            "usage": {"inputTokens": inp, "outputTokens": out, "totalTokens": inp + out},
            "message": {"source": {"replayState": {"blocks": blocks}}},
            "stream": [{"time": t0 + 100}, {"time": t0 + 250}],
        }},
    ]


def test_sessions_to_row_splits_usage_by_teammate(tmp_path):
    lead = "session-0000-lead"
    member = "aaaa-bbbb-1111"
    _write_session(
        tmp_path / "ws" / lead / "session.v4.jsonl",
        {"type": "session", "version": 4, "id": lead, "parentSession": None},
        [
            *_request(1000, 100, 20, n_tools=1),
            {"type": "tool/call", "data": {"name": "spawn_teammate"}},
            {"type": "team/member", "data": {"member": {"id": member, "name": "worker1"}}},
            *_request(2000, 200, 10),
            {"type": "turn/end", "data": {"reason": {"kind": "completed"}}},
        ],
    )
    _write_session(
        tmp_path / "ws" / f"session-{member}" / "session.v4.jsonl",
        {"type": "session", "version": 4, "id": f"session-{member}", "parentSession": lead},
        [*_request(1500, 50, 5), {"type": "turn/end", "data": {"reason": {"kind": "completed"}}}],
    )
    row = sessions_to_row(read_sessions(tmp_path), task_id="t", output_text="done", e2e_latency_s=3.0,
                          errors=[], meta={"teammates": 1, "strategy": "dsh-agent-team"})
    assert row["per_role_usage"]["lead"]["requests"] == 2
    assert row["per_role_usage"]["worker1"]["input_tokens"] == 50
    assert row["total_usage"]["input_tokens"] == 350 and row["total_usage"]["requests"] == 3
    assert [s["role"] for s in row["turn_stats"]] == ["lead", "worker1", "lead"]  # by request start time
    first = row["turn_stats"][0]
    assert first["ttft_s"] == 0.1 and first["decode_s"] == 0.15 and first["num_tool_calls"] == 1
    assert row["agentcard"]["team"] == {"teammates_expected": 1, "teammates_spawned": 1,
                                        "teammate_sessions": 1, "teammates_completed": 1}
    assert row["tool_calls"] == 1


def test_final_text_from_stdout():
    text = "\n".join([json.dumps({"type": "status"}), "not json", json.dumps({"type": "final", "text": "ok"})])
    assert final_text_from_stdout(text) == "ok"


def test_agent_requires_endpoint_and_valid_settings(tmp_path):
    with pytest.raises(ValueError):
        DSHAgent(logs_dir=tmp_path, model="m")
    with pytest.raises(ValueError):
        DSHAgent(logs_dir=tmp_path, model="m", endpoint_url="u", protocol="nope")
    agent = DSHAgent(logs_dir=tmp_path, model="m", endpoint_url="u", teammates=2)
    assert agent.name() == "deepseek-harness" and agent.version() == "0.2.0-rc.2"
    assert "exactly 2 teammates" in agent._prompt("Build it.")
    assert DSHAgent(logs_dir=tmp_path, model="m", endpoint_url="u")._prompt("Build it.") == "Build it."


def test_cli_writes_valid_dsh_job(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")
    out = tmp_path / "job.yaml"
    rc = cli.main([
        "job", "--harness", "dsh", "--benchmark", "nl2repo", "--dataset", str(tmp_path / "ds"),
        "--dsh-model", "deepseek-chat", "--dsh-base-url", "https://api.deepseek.com/v1",
        "--dsh-api-key-env", "DEEPSEEK_API_KEY", "--dsh-teammates", "2", "--out", str(out),
    ])
    assert rc == 0
    text = out.read_text()
    assert "secret-value" not in text
    import yaml

    agent = yaml.safe_load(text)["agents"][0]
    assert agent["import_path"] == jobspec.DSH_IMPORT_PATH
    assert agent["kwargs"]["endpoint_url"] == "https://api.deepseek.com/v1"
    assert agent["kwargs"]["teammates"] == 2 and agent["kwargs"]["api_key_env"] == "DEEPSEEK_API_KEY"
