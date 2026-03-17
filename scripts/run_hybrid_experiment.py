#!/usr/bin/env python3
"""Two-Phase Plan-Execute experiment driver for AgentCAP / MCP-Atlas."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
import yaml
from datasets import load_dataset

from agent_cap.cost.hybrid import (
    APICostConfig,
    LocalCostConfig,
    compute_api_cost,
    compute_local_cost,
)

LOGGER = logging.getLogger("hybrid_experiment")

SYSTEM_PROMPT = (
    "Role: You are a factual, tool-aware assistant connected to a variety of tools. "
    "Use the available tools to answer the user query. Do not ask the user for "
    "clarification; fully complete the task using the information provided in the prompt."
)

PLAN_SYSTEM_PROMPT = (
    "You are an expert planning assistant. Given a task, you must create a clear, "
    "specific, step-by-step plan that another AI agent can follow to complete the task. "
    "The executor agent has access to tools but may have limited reasoning ability, "
    "so your plan must be detailed and unambiguous.\n\n"
    "Output your plan as a numbered list of concrete steps. Each step should specify:\n"
    "- What tool to call (if applicable)\n"
    "- What arguments to use\n"
    "- What to do with the result\n\n"
    "Do NOT execute the task yourself. Only produce the plan."
)

EXEC_WITH_PLAN_PROMPT = (
    "You have been given a task and a step-by-step plan created by a planning agent. "
    "Follow the plan carefully and execute each step using the available tools. "
    "If a step fails, try to recover and continue with the remaining steps.\n\n"
    "TASK: {task}\n\n"
    "PLAN:\n{plan}\n\n"
    "Execute the plan now."
)

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for a single model endpoint."""

    id: str
    base_url: str
    api_key: str = "dummy"
    is_local: bool = False
    throughput_tok_per_sec: float = 50.0
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelConfig":
        api_key = str(raw.get("api_key", "dummy"))
        if api_key.startswith("${") and api_key.endswith("}"):
            api_key = os.environ.get(api_key[2:-1], "")
        return cls(
            id=str(raw["id"]),
            base_url=str(raw["base_url"]),
            api_key=api_key,
            is_local=bool(raw.get("is_local", False)),
            throughput_tok_per_sec=float(raw.get("throughput_tok_per_sec", 50.0)),
            input_price_per_1m=float(raw.get("input_price_per_1m", 0.0)),
            output_price_per_1m=float(raw.get("output_price_per_1m", 0.0)),
        )

    def cost_config(self) -> APICostConfig | LocalCostConfig:
        if self.is_local:
            return LocalCostConfig(
                model_id=self.id,
                throughput_tok_per_sec=self.throughput_tok_per_sec,
            )
        return APICostConfig(
            model_id=self.id,
            input_price_per_1m=self.input_price_per_1m,
            output_price_per_1m=self.output_price_per_1m,
        )


@dataclass
class HybridConfig:
    """Full experiment configuration."""

    name: str
    description: str
    experiment_type: str
    planner: Optional[ModelConfig]
    executor: ModelConfig
    mcp_server_url: str = "http://localhost:1984"
    max_turns: int = 20
    max_tokens: int = 8192
    plan_max_tokens: int = 4096

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HybridConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must be a top-level mapping")

        exp_type = str(raw.get("experiment_type", "plan-execute"))
        planner = None
        if exp_type == "plan-execute" and "planner" in raw:
            planner = ModelConfig.from_dict(raw["planner"])

        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            experiment_type=exp_type,
            planner=planner,
            executor=ModelConfig.from_dict(raw["executor"]),
            mcp_server_url=str(raw.get("mcp_server_url", "http://localhost:1984")),
            max_turns=int(raw.get("max_turns", 20)),
            max_tokens=int(raw.get("max_tokens", 8192)),
            plan_max_tokens=int(raw.get("plan_max_tokens", 4096)),
        )


# ---------------------------------------------------------------------------
# Phase results
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Result from a single phase (plan or execute)."""

    response: str
    messages: List[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    tool_call_count: int
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers (reused from mcpatlas_combo.py)
# ---------------------------------------------------------------------------


def strip_think_tags(text: str) -> str:
    if not text:
        return ""
    return THINK_PATTERN.sub("", text).strip()


def flatten_tool_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("content"), list):
            return flatten_tool_payload(payload["content"])
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(payload, list):
        parts: List[str] = []
        for item in payload:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return str(payload)


def count_tool_calls(messages: Sequence[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            total += len(msg.get("tool_calls") or [])
    return total


def extract_final_assistant_text(messages: Sequence[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg.get("content", ""))
    return ""


def parse_enabled_tools(raw: Any) -> List[str]:
    if raw is None:
        return []
    value = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list) or not value:
        return []
    if isinstance(value[0], str):
        return [str(item) for item in value if isinstance(item, str)]
    names: List[str] = []
    for item in value:
        if isinstance(item, dict) and "name" in item:
            names.append(str(item["name"]))
    return names


def run_id(experiment_name: str, task_id: str) -> str:
    key = f"{experiment_name}|{task_id}"
    return f"hybrid-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# MCP-Atlas tool interaction
# ---------------------------------------------------------------------------


async def list_openai_tools(
    session: aiohttp.ClientSession,
    mcp_server_url: str,
    enabled_tools: Sequence[str],
) -> List[Dict[str, Any]]:
    async with session.post(f"{mcp_server_url.rstrip('/')}/list-tools") as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"list-tools failed ({resp.status}): {body}")
        payload = await resp.json()
    enabled = set(enabled_tools)
    transformed: List[Dict[str, Any]] = []
    for tool in payload:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name", ""))
        if not name:
            continue
        if enabled and name not in enabled:
            continue
        transformed.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description", "")),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return transformed


async def chat_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    max_tokens: int,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    headers = {}
    if api_key and api_key != "dummy":
        headers["Authorization"] = f"Bearer {api_key}"

    is_openai = "api.openai.com" in base_url
    token_key = "max_completion_tokens" if is_openai else "max_tokens"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        token_key: max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    async with session.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"chat failed ({resp.status}): {body}")
        return await resp.json()


async def mcp_call_tool(
    session: aiohttp.ClientSession,
    mcp_server_url: str,
    tool_name: str,
    tool_args: Any,
) -> Any:
    async with session.post(
        f"{mcp_server_url.rstrip('/')}/call-tool",
        json={"tool_name": tool_name, "tool_args": tool_args},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"tool call failed ({resp.status}): {body}")
        return await resp.json()


# ---------------------------------------------------------------------------
# Phase 1: Planning
# ---------------------------------------------------------------------------


async def run_plan_phase(
    session: aiohttp.ClientSession,
    planner: ModelConfig,
    task_prompt: str,
    enabled_tools: Sequence[str],
    mcp_server_url: str,
    plan_max_tokens: int,
) -> PhaseResult:
    """Generate a step-by-step plan using the planner model (no tool calls)."""
    tools = await list_openai_tools(session, mcp_server_url, enabled_tools)
    tool_descriptions = "\n".join(
        f"- {t['function']['name']}: {t['function'].get('description', '')}"
        for t in tools
    )

    user_content = (
        f"TASK: {task_prompt}\n\n"
        f"AVAILABLE TOOLS:\n{tool_descriptions}\n\n"
        "Create a detailed step-by-step plan to complete this task. "
        "Be specific about which tools to use and with what arguments."
    )

    messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    start = time.perf_counter()
    result = await chat_completion(
        session,
        planner.base_url,
        planner.api_key,
        planner.id,
        messages,
        None,
        plan_max_tokens,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - start

    usage = result.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))

    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError("planner returned empty choices")
    assistant = choices[0].get("message") or {}
    plan_text = strip_think_tags(str(assistant.get("content", "")))

    messages.append(assistant)

    return PhaseResult(
        response=plan_text,
        messages=messages,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
        tool_call_count=0,
    )


# ---------------------------------------------------------------------------
# Phase 2: Execution (with or without plan)
# ---------------------------------------------------------------------------


async def run_exec_phase(
    session: aiohttp.ClientSession,
    executor: ModelConfig,
    task_prompt: str,
    plan_text: Optional[str],
    enabled_tools: Sequence[str],
    mcp_server_url: str,
    max_turns: int,
    max_tokens: int,
) -> PhaseResult:
    """Execute a task, optionally guided by a plan."""
    tools = await list_openai_tools(session, mcp_server_url, enabled_tools)

    if plan_text:
        user_content = EXEC_WITH_PLAN_PROMPT.format(task=task_prompt, plan=plan_text)
    else:
        user_content = task_prompt

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    errors: List[str] = []
    input_tokens = 0
    output_tokens = 0
    start = time.perf_counter()

    for turn in range(max_turns):
        try:
            result = await chat_completion(
                session,
                executor.base_url,
                executor.api_key,
                executor.id,
                messages,
                tools,
                max_tokens,
                temperature=0.0,
            )
        except Exception as exc:
            errors.append(f"turn {turn}: {exc}")
            break

        usage = result.get("usage") or {}
        input_tokens += int(usage.get("prompt_tokens", 0))
        output_tokens += int(usage.get("completion_tokens", 0))

        choices = result.get("choices") or []
        if not choices:
            errors.append(f"turn {turn}: empty choices")
            break

        assistant = choices[0].get("message") or {}
        messages.append(assistant)

        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            break

        for tc in tool_calls:
            function = tc.get("function") or {}
            name = str(function.get("name", ""))
            raw_args = function.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            try:
                tool_payload = await mcp_call_tool(session, mcp_server_url, name, args)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                tool_payload = [{"type": "text", "text": f"ERROR: {exc}"}]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": flatten_tool_payload(tool_payload),
                }
            )

    elapsed = time.perf_counter() - start
    final_response = strip_think_tags(extract_final_assistant_text(messages))

    return PhaseResult(
        response=final_response,
        messages=messages,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
        tool_call_count=count_tool_calls(messages),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


def compute_cost(model_cfg: ModelConfig, in_tokens: int, out_tokens: int) -> float:
    """Compute cost in USD for a phase."""
    cfg = model_cfg.cost_config()
    if isinstance(cfg, APICostConfig):
        return compute_api_cost(cfg, in_tokens, out_tokens)
    return compute_local_cost(cfg, in_tokens, out_tokens)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS hybrid_runs (
            id TEXT PRIMARY KEY,
            experiment_name TEXT NOT NULL,
            experiment_type TEXT,
            task_id TEXT,
            task_prompt TEXT,

            -- Plan phase
            plan_model_id TEXT,
            plan_text TEXT,
            plan_input_tokens INTEGER DEFAULT 0,
            plan_output_tokens INTEGER DEFAULT 0,
            plan_cost_usd REAL DEFAULT 0.0,
            plan_latency_s REAL DEFAULT 0.0,

            -- Exec phase
            exec_model_id TEXT,
            exec_response TEXT,
            exec_input_tokens INTEGER DEFAULT 0,
            exec_output_tokens INTEGER DEFAULT 0,
            exec_cost_usd REAL DEFAULT 0.0,
            exec_latency_s REAL DEFAULT 0.0,
            exec_tool_calls INTEGER DEFAULT 0,
            exec_errors TEXT,

            -- Totals
            total_cost_usd REAL DEFAULT 0.0,
            total_latency_s REAL DEFAULT 0.0,
            total_input_tokens INTEGER DEFAULT 0,
            total_output_tokens INTEGER DEFAULT 0,

            -- Scoring (filled later by scoring script)
            task_success BOOLEAN,
            quality_score REAL,
            coverage_score REAL,

            -- Metadata
            trajectory_log TEXT,
            started_at TEXT,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_hybrid_exp ON hybrid_runs(experiment_name);
        CREATE INDEX IF NOT EXISTS idx_hybrid_task ON hybrid_runs(experiment_name, task_id);
    """)
    conn.commit()
    return conn


def save_hybrid_run(conn: sqlite3.Connection, record: Dict[str, Any]) -> None:
    cols = list(record.keys())
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO hybrid_runs ({', '.join(cols)}) VALUES ({placeholders})"
    conn.execute(sql, [record[col] for col in cols])
    conn.commit()


def existing_task_ids(conn: sqlite3.Connection, experiment_name: str) -> set:
    rows = conn.execute(
        "SELECT task_id FROM hybrid_runs WHERE experiment_name = ?",
        (experiment_name,),
    ).fetchall()
    return {str(row["task_id"]) for row in rows}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tasks(num_tasks: int) -> List[Dict[str, Any]]:
    ds = load_dataset("ScaleAI/MCP-Atlas", split="train")
    n = min(max(num_tasks, 0), len(ds))
    if n == 0:
        return []
    sel = ds.select(range(n))
    return [sel[i] for i in range(n)]


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


async def run_experiment(args: argparse.Namespace) -> None:
    config = HybridConfig.from_yaml(args.config)
    tasks = load_tasks(args.num_tasks)

    if args.dry_run:
        print(f"Experiment: {config.name}")
        print(f"Type: {config.experiment_type}")
        if config.planner:
            print(f"Planner: {config.planner.id} @ {config.planner.base_url}")
        print(f"Executor: {config.executor.id} @ {config.executor.base_url}")
        print(f"Tasks: {len(tasks)}")
        print(f"MCP Server: {config.mcp_server_url}")
        return

    conn = init_db(Path(args.db))
    done = existing_task_ids(conn, config.name)
    LOGGER.info(
        "Experiment %s: %d tasks, %d already done",
        config.name,
        len(tasks),
        len(done),
    )

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=900)
    ) as session:
        for idx, row in enumerate(tasks, start=1):
            task_id = str(row.get("TASK", f"mcpatlas-{idx}"))
            prompt = str(row.get("PROMPT", ""))
            enabled_tools = parse_enabled_tools(row.get("ENABLED_TOOLS"))

            if task_id in done:
                LOGGER.info(
                    "[%d/%d] %s — skipped (already done)", idx, len(tasks), task_id
                )
                continue

            started_at = now_iso()
            LOGGER.info("[%d/%d] %s — starting...", idx, len(tasks), task_id)

            # --- Phase 1: Plan ---
            plan_result: Optional[PhaseResult] = None
            plan_text: Optional[str] = None

            if config.experiment_type == "plan-execute" and config.planner:
                try:
                    plan_result = await run_plan_phase(
                        session,
                        config.planner,
                        prompt,
                        enabled_tools,
                        config.mcp_server_url,
                        config.plan_max_tokens,
                    )
                    plan_text = plan_result.response
                    LOGGER.info(
                        "  Plan: %d in + %d out tokens, %.1fs",
                        plan_result.input_tokens,
                        plan_result.output_tokens,
                        plan_result.elapsed_seconds,
                    )
                except Exception as exc:
                    LOGGER.error("  Plan phase failed: %s", exc)
                    plan_result = PhaseResult(
                        response=f"ERROR: {exc}",
                        messages=[],
                        input_tokens=0,
                        output_tokens=0,
                        elapsed_seconds=0,
                        tool_call_count=0,
                        errors=[str(exc)],
                    )

            # --- Phase 2: Execute ---
            try:
                exec_result = await run_exec_phase(
                    session,
                    config.executor,
                    prompt,
                    plan_text,
                    enabled_tools,
                    config.mcp_server_url,
                    config.max_turns,
                    config.max_tokens,
                )
                LOGGER.info(
                    "  Exec: %d in + %d out tokens, %d tool calls, %.1fs",
                    exec_result.input_tokens,
                    exec_result.output_tokens,
                    exec_result.tool_call_count,
                    exec_result.elapsed_seconds,
                )
            except Exception as exc:
                LOGGER.error("  Exec phase failed: %s", exc)
                exec_result = PhaseResult(
                    response=f"ERROR: {exc}",
                    messages=[],
                    input_tokens=0,
                    output_tokens=0,
                    elapsed_seconds=0,
                    tool_call_count=0,
                    errors=[str(exc)],
                )

            # --- Cost ---
            plan_cost = 0.0
            if plan_result and config.planner:
                plan_cost = compute_cost(
                    config.planner,
                    plan_result.input_tokens,
                    plan_result.output_tokens,
                )
            exec_cost = compute_cost(
                config.executor,
                exec_result.input_tokens,
                exec_result.output_tokens,
            )
            total_cost = plan_cost + exec_cost

            # --- Save ---
            completed_at = now_iso()
            p_in = plan_result.input_tokens if plan_result else 0
            p_out = plan_result.output_tokens if plan_result else 0
            p_lat = plan_result.elapsed_seconds if plan_result else 0.0

            record = {
                "id": run_id(config.name, task_id),
                "experiment_name": config.name,
                "experiment_type": config.experiment_type,
                "task_id": task_id,
                "task_prompt": prompt[:2000],
                "plan_model_id": config.planner.id if config.planner else "",
                "plan_text": (plan_text or "")[:5000],
                "plan_input_tokens": p_in,
                "plan_output_tokens": p_out,
                "plan_cost_usd": plan_cost,
                "plan_latency_s": p_lat,
                "exec_model_id": config.executor.id,
                "exec_response": exec_result.response[:5000],
                "exec_input_tokens": exec_result.input_tokens,
                "exec_output_tokens": exec_result.output_tokens,
                "exec_cost_usd": exec_cost,
                "exec_latency_s": exec_result.elapsed_seconds,
                "exec_tool_calls": exec_result.tool_call_count,
                "exec_errors": json.dumps(exec_result.errors, ensure_ascii=False),
                "total_cost_usd": total_cost,
                "total_latency_s": p_lat + exec_result.elapsed_seconds,
                "total_input_tokens": p_in + exec_result.input_tokens,
                "total_output_tokens": p_out + exec_result.output_tokens,
                "task_success": None,
                "quality_score": None,
                "coverage_score": None,
                "trajectory_log": json.dumps(exec_result.messages, ensure_ascii=False)[
                    :50000
                ],
                "started_at": started_at,
                "completed_at": completed_at,
            }
            save_hybrid_run(conn, record)
            done.add(task_id)

            LOGGER.info(
                "  Done: cost=$%.4f (plan=$%.4f + exec=$%.4f), tools=%d, time=%.1fs",
                total_cost,
                plan_cost,
                exec_cost,
                exec_result.tool_call_count,
                p_lat + exec_result.elapsed_seconds,
            )

    conn.close()
    LOGGER.info("Experiment %s complete.", config.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid plan-execute experiments on MCP-Atlas"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file for the experiment",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of MCP-Atlas tasks to run (default: 50)",
    )
    parser.add_argument(
        "--db",
        default="results/hybrid_experiments.db",
        help="SQLite database path (default: results/hybrid_experiments.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
