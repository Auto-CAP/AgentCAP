#!/usr/bin/env python3
"""Run predeclared paired FinanceBench two-role and three-role rebuttal arms.

The task manifest and two-role prompts are identical to the response-period
prompt-sensitivity experiment. Each invocation runs one arm and writes a full
TeamRunner artifact tree plus a credential-free protocol specification.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_cap.runner.team_runner import (  # noqa: E402
    ModelEndpoint,
    PlanExecuteStrategy,
    PlanVerifyExecuteStrategy,
    TeamConfig,
    TeamRunner,
)
from agent_cap.runner.unified_runner import _load_dataset_tasks  # noqa: E402
from scripts.rebuttal_prompt_ablation import (  # noqa: E402
    select_stratified,
    stable_hash,
    task_stratum,
)

ARMS: dict[str, dict[str, Any]] = {
    "pve-mini-gpt54-gpt54": {
        "strategy": "plan-verify-execute",
        "planner": "gpt-5.4-mini",
        "verifier": "gpt-5.4",
        "executor": "gpt-5.4",
    },
    "pve-gpt54-mini-mini": {
        "strategy": "plan-verify-execute",
        "planner": "gpt-5.4",
        "verifier": "gpt-5.4-mini",
        "executor": "gpt-5.4-mini",
    },
    "pe-gpt54-claude46": {
        "strategy": "plan-execute",
        "planner": "gpt-5.4",
        "executor": "claude-opus-4-6",
    },
    "pve-gpt54-claude46-claude46": {
        "strategy": "plan-verify-execute",
        "planner": "gpt-5.4",
        "verifier": "claude-opus-4-6",
        "executor": "claude-opus-4-6",
    },
    "pe-claude46-gpt54": {
        "strategy": "plan-execute",
        "planner": "claude-opus-4-6",
        "executor": "gpt-5.4",
    },
    "pve-claude46-claude46-gpt54": {
        "strategy": "plan-verify-execute",
        "planner": "claude-opus-4-6",
        "verifier": "claude-opus-4-6",
        "executor": "gpt-5.4",
    },
}


def _load_env() -> None:
    env_path = Path("/root/.hermes/.env")
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _endpoint(
    name: str,
    base_url: str,
    api_key: str,
    *,
    max_tokens: int = 8192,
) -> ModelEndpoint:
    return ModelEndpoint(
        name=name,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
        max_tokens=max_tokens,
        use_streaming=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--base-url", default="http://127.0.0.1:8317/v1")
    parser.add_argument("--api-key-env", default="CLIPROXYAPI_API_KEY")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    all_tasks = _load_dataset_tasks("financebench", 0)
    tasks = select_stratified(all_tasks, "financebench", args.num_tasks, args.seed)
    if len(tasks) != args.num_tasks:
        raise SystemExit(f"selected {len(tasks)} tasks, expected {args.num_tasks}")
    task_ids = [str(task.task_id) for task in tasks]
    manifest: dict[str, Any] = {
        "dataset": "financebench",
        "dataset_tasks_available": len(all_tasks),
        "num_tasks": len(tasks),
        "seed": args.seed,
        "selection": "deterministic round-robin over dataset-specific strata",
        "task_ids": task_ids,
        "task_manifest_sha256": stable_hash("\n".join(task_ids)),
        "strata_counts": {},
    }
    for task in tasks:
        key = task_stratum("financebench", task)
        manifest["strata_counts"][key] = manifest["strata_counts"].get(key, 0) + 1

    _load_env()
    api_key = os.environ.get(args.api_key_env, "")
    if not args.validate_only and not api_key:
        raise SystemExit(f"missing API key in {args.api_key_env}")

    arm = ARMS[args.arm]
    role_max_tokens = {
        role: 1024 if role == "verifier" else 8192
        for role in arm
        if role in {"planner", "verifier", "executor"}
    }
    models = {
        role: _endpoint(
            model,
            args.base_url,
            api_key,
            max_tokens=role_max_tokens[role],
        )
        for role, model in arm.items()
        if role in {"planner", "verifier", "executor"}
    }
    config = TeamConfig(
        strategy=arm["strategy"],
        models=models,
        dataset="financebench",
        backend="financebench",
        max_turns=10,
        output_root=args.out,
    )
    runner = TeamRunner(config)
    runner._validate_roles()

    args.out.mkdir(parents=True, exist_ok=True)
    spec = {
        **manifest,
        "arm": args.arm,
        "strategy": arm["strategy"],
        "roles": {role: endpoint.name for role, endpoint in models.items()},
        "base_url": args.base_url,
        "temperature": 0.0,
        "max_tokens": 8192,
        "role_max_tokens": role_max_tokens,
        "max_turns": 10,
        "planner_prompt_sha256": stable_hash(PlanExecuteStrategy.PLAN_SYSTEM_PROMPT),
        "verifier_prompt_sha256": (
            stable_hash(PlanVerifyExecuteStrategy.VERIFY_SYSTEM_PROMPT)
            if "verifier" in models
            else None
        ),
        "executor_prompt_sha256": stable_hash(PlanExecuteStrategy.EXEC_SYSTEM_PROMPT),
        "protocol_sha256": _sha256_json(
            {
                "task_ids": manifest["task_ids"],
                "strategy": arm["strategy"],
                "roles": {role: endpoint.name for role, endpoint in models.items()},
                "planner_prompt": PlanExecuteStrategy.PLAN_SYSTEM_PROMPT,
                "verifier_prompt": (
                    PlanVerifyExecuteStrategy.VERIFY_SYSTEM_PROMPT
                    if "verifier" in models
                    else None
                ),
                "executor_prompt": PlanExecuteStrategy.EXEC_SYSTEM_PROMPT,
                "temperature": 0.0,
                "role_max_tokens": role_max_tokens,
                "max_turns": 10,
            }
        ),
        "credentials_serialized": False,
        "failures_scored_as_zero": True,
    }
    (args.out / "three_role_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.validate_only:
        print(json.dumps({"status": "validated", **spec}, ensure_ascii=False, indent=2))
        return

    result = asyncio.run(runner.run(tasks))
    task_ids = [row.task_id for row in result.task_results]
    summary = {
        "status": "completed",
        "arm": args.arm,
        "output_dir": str(result.output_dir),
        "task_manifest_sha256": manifest["task_manifest_sha256"],
        "requested_tasks": len(tasks),
        "returned_tasks": len(result.task_results),
        "task_ids_match": task_ids == manifest["task_ids"],
        "errors": sum(1 for row in result.task_results if row.errors),
        "passed": sum(1 for row in result.task_results if row.eval_passed is True),
        "scored": sum(1 for row in result.task_results if row.eval_passed is not None),
        "requests": sum(row.num_requests for row in result.task_results),
        "input_tokens": sum(row.total_input_tokens for row in result.task_results),
        "output_tokens": sum(row.total_output_tokens for row in result.task_results),
    }
    (args.out / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
