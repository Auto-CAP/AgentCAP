import asyncio
from types import SimpleNamespace
from typing import cast

import aiohttp

from agent_cap.runner import team_runner as module
from agent_cap.runner.team_runner import (
    ModelEndpoint,
    PlanExecuteStrategy,
    PlanVerifyExecuteStrategy,
    RoleMetrics,
    TeamRunner,
)
from agent_cap.runner.unified_runner import UnifiedTask
from scripts.aggregate_rebuttal_three_role import exact_mcnemar_p
from scripts.regrade_rebuttal_three_role_pairwise import parse_json as parse_judge_json


def _metric(role: str, name: str) -> RoleMetrics:
    return RoleMetrics(
        model_name=name,
        role=role,
        input_tokens=0,
        output_tokens=0,
        cached_tokens=0,
        prefill_time_s=0.0,
        decode_time_s=0.0,
        num_requests=0,
    )


def test_plan_verify_execute_registered_and_requires_three_roles():
    strategy = TeamRunner._resolve_strategy("plan-verify-execute")
    assert isinstance(strategy, PlanVerifyExecuteStrategy)
    assert strategy.required_roles() == ["planner", "verifier", "executor"]


def test_exact_mcnemar_two_sided_values():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(1, 1) == 1.0
    assert exact_mcnemar_p(3, 0) == 0.25
    assert exact_mcnemar_p(6, 0) == 0.03125


def test_pairwise_judge_parser_is_strict():
    parsed = parse_judge_json(
        '<think>hidden</think> {"A_correct": true, "B_correct": false, "reason": "A matches."}'
    )
    assert parsed == {
        "A_correct": True,
        "B_correct": False,
        "reason": "A matches.",
    }


def test_verifier_json_parser_strips_reasoning_and_fences():
    text = (
        '<think>private</think>\n```json\n'
        '{"status":"APPROVED","feedback":""}\n```'
    )
    parsed = PlanVerifyExecuteStrategy._parse_verifier_json(text)
    assert parsed == {"status": "APPROVED", "feedback": ""}
    assert PlanVerifyExecuteStrategy._parse_verifier_json("not json") is None


def test_base_postprocess_hook_is_identity():
    strategy = PlanExecuteStrategy()
    plan, index = asyncio.run(
        strategy._postprocess_plan(
            session=cast(aiohttp.ClientSession, None),
            models={},
            task=cast(UnifiedTask, SimpleNamespace()),
            user_prompt="task",
            plan_text="draft",
            tools=[],
            role_metrics={},
            per_request_details=[],
            errors=[],
            all_input_tokens=[],
            request_index=1,
            task_dir=None,
        )
    )
    assert (plan, index) == ("draft", 1)


def test_verifier_hook_records_role_separated_usage(monkeypatch, tmp_path):
    captured = {}

    async def fake_chat(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            response_json={
                "choices": [
                    {"message": {"content": '{"status":"APPROVED","feedback":""}'}}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cached_tokens": 5,
                },
            },
            input_tokens=100,
            output_tokens=20,
            cached_tokens=5,
            ttft_seconds=0.1,
            decode_seconds=0.2,
            is_streaming=True,
        )

    monkeypatch.setattr(module, "_chat_with_fallback", fake_chat)
    strategy = PlanVerifyExecuteStrategy()
    models = {
        "planner": ModelEndpoint(name="p", base_url="http://p"),
        "verifier": ModelEndpoint(name="v", base_url="http://v"),
        "executor": ModelEndpoint(name="e", base_url="http://e"),
    }
    metrics = {role: _metric(role, endpoint.name) for role, endpoint in models.items()}
    details = []
    errors = []
    inputs = []
    plan, index = asyncio.run(
        strategy._postprocess_plan(
            session=cast(aiohttp.ClientSession, None),
            models=models,
            task=cast(UnifiedTask, SimpleNamespace(task_id="t1")),
            user_prompt="Find revenue.",
            plan_text="1. Guess.",
            tools=[{"function": {"name": "financebench_retrieve"}}],
            role_metrics=metrics,
            per_request_details=details,
            errors=errors,
            all_input_tokens=inputs,
            request_index=1,
            task_dir=tmp_path,
        )
    )

    assert plan == "1. Guess."
    assert index == 2
    assert errors == []
    assert metrics["verifier"].num_requests == 1
    assert metrics["verifier"].input_tokens == 100
    assert metrics["verifier"].output_tokens == 20
    assert details[0]["role"] == "verifier"
    assert inputs == [100]
    assert captured["tools"] is None
    verifier_user = captured["messages"][1]["content"]
    assert '"status":"APPROVED"' in verifier_user
    assert '"status":"NEEDS_REVISION"' in verifier_user
    assert "Evaluate this proposed plan" in verifier_user
    assert (tmp_path / "verify_request_round_1.json").exists()
    assert (tmp_path / "verify_response_round_1.json").exists()
    assert (tmp_path / "verified_plan.txt").read_text() == plan


def test_needs_revision_calls_planner_and_records_both_roles(monkeypatch, tmp_path):
    calls = []

    async def fake_chat(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "v":
            content = '{"status":"NEEDS_REVISION","feedback":"Use the available retrieval tool."}'
            prompt_tokens, completion_tokens = 100, 20
        else:
            content = "1. Call financebench_retrieve.\n2. Calculate the answer."
            prompt_tokens, completion_tokens = 120, 30
        return SimpleNamespace(
            response_json={
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": 0,
                },
            },
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cached_tokens=0,
            ttft_seconds=0.1,
            decode_seconds=0.2,
            is_streaming=True,
        )

    monkeypatch.setattr(module, "_chat_with_fallback", fake_chat)
    strategy = PlanVerifyExecuteStrategy()
    models = {
        "planner": ModelEndpoint(name="p", base_url="http://p"),
        "verifier": ModelEndpoint(name="v", base_url="http://v"),
        "executor": ModelEndpoint(name="e", base_url="http://e"),
    }
    metrics = {role: _metric(role, endpoint.name) for role, endpoint in models.items()}
    details = []
    errors = []
    inputs = []
    plan, index = asyncio.run(
        strategy._postprocess_plan(
            session=cast(aiohttp.ClientSession, None),
            models=models,
            task=cast(UnifiedTask, SimpleNamespace(task_id="t1")),
            user_prompt="Find revenue.",
            plan_text="1. Guess.",
            tools=[{"function": {"name": "financebench_retrieve"}}],
            role_metrics=metrics,
            per_request_details=details,
            errors=errors,
            all_input_tokens=inputs,
            request_index=1,
            task_dir=tmp_path,
        )
    )

    assert calls == ["v", "p", "v", "p"]
    assert plan.startswith("1. Call financebench_retrieve")
    assert index == 5
    assert errors == []
    assert metrics["verifier"].num_requests == 2
    assert metrics["planner"].num_requests == 2
    assert [row["role"] for row in details] == [
        "verifier",
        "planner",
        "verifier",
        "planner",
    ]
    assert (tmp_path / "revision_request_round_1.json").exists()
    assert (tmp_path / "revision_response_round_1.json").exists()
    assert (tmp_path / "revision_request_round_2.json").exists()
    assert (tmp_path / "revision_response_round_2.json").exists()
