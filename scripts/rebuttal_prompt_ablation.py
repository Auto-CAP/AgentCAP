#!/usr/bin/env python3
"""Run a real, paired planner/executor prompt-sensitivity experiment.

The two prompt arms preserve the same task requirements while paraphrasing the
planner and executor system messages. All other settings are held fixed. Each
run writes a task manifest, prompt hashes, full TeamRunner artifacts, and a
machine-readable summary. No credentials are serialized.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_cap.runner.team_runner import (
    ModelEndpoint,
    PlanExecuteStrategy,
    TeamConfig,
    TeamRunner,
)
from agent_cap.runner.unified_runner import _load_dataset_tasks

ORIGINAL_PLAN = PlanExecuteStrategy.PLAN_SYSTEM_PROMPT
ORIGINAL_EXEC = PlanExecuteStrategy.EXEC_SYSTEM_PROMPT
PARAPHRASE_PLAN = """You are the planner for a tool-using executor. Produce a numbered, decision-complete procedure for the user's task. Name the relevant tools and concrete arguments, explain how each result determines the next step, include recovery branches for likely failures, and finish with the required output format. Do not execute tools or answer the task yourself. Do not ask the user for information that can be obtained with the available tools."""
PARAPHRASE_EXEC = """You are the executor in a planner-executor team. Complete the user's task by carrying out the supplied plan with the available tools. Call tools rather than describing hypothetical calls; inspect returned values, adapt when observations differ from the plan, recover from tool errors when possible, and return the requested final answer. Do not ask the user for information that an available tool can obtain."""

READ_ONLY_MCP_SERVERS = {
    "arxiv", "brave-search", "calculator", "clinicaltrialsgov-mcp-server",
    "context7", "ddg-search", "fetch", "met-museum", "open-library",
    "osm-mcp-server", "pubmed", "weather", "whois", "wikipedia",
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tool_server(tool_name: str) -> str:
    for server in sorted(READ_ONLY_MCP_SERVERS, key=len, reverse=True):
        if tool_name.startswith(server):
            return server
    return tool_name.split("_", 1)[0]


def is_mcp_read_only(task: Any) -> bool:
    return {tool_server(str(x)) for x in (task.enabled_tools or [])}.issubset(
        READ_ONLY_MCP_SERVERS
    )


def task_stratum(dataset: str, task: Any) -> str:
    cfg = task.eval_config or {}
    if dataset == "imo-answerbench":
        return str(cfg.get("category") or "math")
    if dataset == "financebench":
        return str(cfg.get("question_type") or "unknown")
    tools = sorted(str(x) for x in (task.enabled_tools or []))
    servers = sorted({tool_server(x) for x in tools})
    return "+".join(servers) or "no-tools"


def select_stratified(tasks: Sequence[Any], dataset: str, n: int, seed: int) -> list[Any]:
    if n <= 0 or n >= len(tasks):
        return list(tasks)
    groups: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        groups[task_stratum(dataset, task)].append(task)
    queues: dict[str, deque[Any]] = {}
    for key, values in groups.items():
        values = sorted(values, key=lambda t: stable_hash(f"{seed}:{t.task_id}"))
        queues[key] = deque(values)
    selected: list[Any] = []
    ordered_groups = sorted(queues)
    while len(selected) < n:
        progressed = False
        for key in ordered_groups:
            if queues[key]:
                selected.append(queues[key].popleft())
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            break
    return selected


def endpoint(name: str, base_url: str, api_key: str) -> ModelEndpoint:
    return ModelEndpoint(
        name=name,
        base_url=base_url,
        api_key=api_key,
        is_local=False,
        max_tokens=8192,
        temperature=0.0,
        use_streaming=False,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        choices=["financebench", "imo-answerbench", "mcp-atlas"],
        required=True,
    )
    ap.add_argument("--variant", choices=["original", "paraphrase"], required=True)
    ap.add_argument("--planner", required=True)
    ap.add_argument("--executor", required=True)
    ap.add_argument("--num-tasks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--base-url", default="http://127.0.0.1:8317/v1")
    ap.add_argument("--api-key-env", default="CLIPROXYAPI_API_KEY")
    ap.add_argument("--mcp-server-url", default="http://127.0.0.1:1984")
    ap.add_argument("--mcp-read-only", action="store_true",
                    help="restrict MCP-Atlas to tasks whose tools cannot mutate shared state")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--validate-only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.variant == "original":
        PlanExecuteStrategy.PLAN_SYSTEM_PROMPT = ORIGINAL_PLAN
        PlanExecuteStrategy.EXEC_SYSTEM_PROMPT = ORIGINAL_EXEC
    else:
        PlanExecuteStrategy.PLAN_SYSTEM_PROMPT = PARAPHRASE_PLAN
        PlanExecuteStrategy.EXEC_SYSTEM_PROMPT = PARAPHRASE_EXEC

    source_tasks = _load_dataset_tasks(args.dataset, 0)
    all_tasks = source_tasks
    if args.mcp_read_only:
        if args.dataset != "mcp-atlas":
            raise SystemExit("--mcp-read-only is valid only for mcp-atlas")
        all_tasks = [task for task in source_tasks if is_mcp_read_only(task)]
    tasks = select_stratified(all_tasks, args.dataset, args.num_tasks, args.seed)
    if len(tasks) != min(args.num_tasks, len(all_tasks)):
        raise SystemExit(f"selected {len(tasks)} tasks, expected {min(args.num_tasks, len(all_tasks))}")

    task_ids = [str(t.task_id) for t in tasks]
    manifest = {
        "dataset": args.dataset,
        "source_dataset_tasks_available": len(source_tasks),
        "dataset_tasks_available": len(all_tasks),
        "num_tasks": len(tasks),
        "seed": args.seed,
        "selection": "deterministic round-robin over dataset-specific strata",
        "mcp_read_only_subset": bool(args.mcp_read_only),
        "task_ids": task_ids,
        "task_manifest_sha256": stable_hash("\n".join(task_ids)),
        "strata_counts": {},
    }
    for task in tasks:
        key = task_stratum(args.dataset, task)
        manifest["strata_counts"][key] = manifest["strata_counts"].get(key, 0) + 1

    run_name = f"{args.dataset}_{args.variant}_{args.planner.replace('/', '_')}_to_{args.executor.replace('/', '_')}"
    out = args.output_root / run_name
    out.mkdir(parents=True, exist_ok=True)
    spec = {
        **manifest,
        "variant": args.variant,
        "planner": args.planner,
        "executor": args.executor,
        "base_url": args.base_url,
        "temperature": 0.0,
        "max_tokens": 8192,
        "max_turns": 20 if args.dataset == "mcp-atlas" else 10,
        "planner_prompt_sha256": stable_hash(PlanExecuteStrategy.PLAN_SYSTEM_PROMPT),
        "executor_prompt_sha256": stable_hash(PlanExecuteStrategy.EXEC_SYSTEM_PROMPT),
        "planner_prompt": PlanExecuteStrategy.PLAN_SYSTEM_PROMPT,
        "executor_prompt": PlanExecuteStrategy.EXEC_SYSTEM_PROMPT,
        "credentials_serialized": False,
    }
    (out / "ablation_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    if args.validate_only:
        print(json.dumps({"status": "validated", "output": str(out), **manifest}, indent=2))
        return

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    enabled_tools: list[str] = []
    if args.dataset == "mcp-atlas":
        enabled_tools = sorted({str(x) for t in tasks for x in (t.enabled_tools or [])})
    cfg = TeamConfig(
        strategy="plan-execute",
        models={
            "planner": endpoint(args.planner, args.base_url, api_key),
            "executor": endpoint(args.executor, args.base_url, api_key),
        },
        dataset=args.dataset,
        backend={
            "mcp-atlas": "mcp",
            "imo-answerbench": "math-python",
            "financebench": "financebench",
        }[args.dataset],
        mcp_server_url=args.mcp_server_url,
        max_turns=spec["max_turns"],
        output_root=out,
        enabled_tools=enabled_tools,
    )
    result = asyncio.run(TeamRunner(cfg).run(tasks))
    summary = {
        "status": "completed",
        "run_name": run_name,
        "output_dir": str(result.output_dir),
        "metrics": result.metrics,
        "spec": spec,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
