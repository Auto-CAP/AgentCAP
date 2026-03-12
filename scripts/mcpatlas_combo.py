#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import aiohttp
import yaml
from datasets import load_dataset

LOGGER = logging.getLogger("mcpatlas_combo")

SYSTEM_PROMPT = (
    "Role: You are a factual, tool-aware assistant connected to a variety of tools. "
    "Use the available tools to answer the user query. Do not ask the user for "
    "clarification; fully complete the task using the information provided in the prompt."
)

SUPPORTED_STRATEGIES = {
    "cascade",
    "vote",
    "best-of-n-small",
    "best-of-n-large",
    "adaptive-cascade",
}

CSV_COLUMNS = [
    "TASK",
    "PROMPT",
    "TRAJECTORY",
    "GTFA_CLAIMS",
    "ENABLED_TOOLS",
    "script_model_response",
    "raw_conversation_history",
    "trajectory",
    "errors",
    "trajectory_time",
    "num_retry",
]

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass
class ModelSpec:
    id: str
    arch: str
    params_b: float
    active_b: float
    tp: int
    port: int

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelSpec":
        return cls(
            id=str(raw["id"]),
            arch=str(raw.get("arch", "dense")),
            params_b=float(raw.get("params_b", 0)),
            active_b=float(raw.get("active_b", 0)),
            tp=int(raw.get("tp", 1)),
            port=int(raw.get("port", 30000)),
        )


@dataclass
class ComboConfig:
    name: str
    description: str
    small_model: ModelSpec
    large_model: ModelSpec
    strategies: List[str]
    mcp_server_url: str
    max_turns: int
    max_tokens: int
    gpu_type: str
    serving_engine: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ComboConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Config YAML must be a top-level mapping")
        if "small_model" not in raw or "large_model" not in raw:
            raise ValueError("Config requires small_model and large_model")
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            small_model=ModelSpec.from_dict(raw["small_model"]),
            large_model=ModelSpec.from_dict(raw["large_model"]),
            strategies=[str(s) for s in (raw.get("strategies", []) or [])],
            mcp_server_url=str(raw.get("mcp_server_url", "http://localhost:1984")),
            max_turns=int(raw.get("max_turns", 20)),
            max_tokens=int(raw.get("max_tokens", 8192)),
            gpu_type=str(raw.get("gpu_type", "")),
            serving_engine=str(raw.get("serving_engine", "sglang")),
        )


@dataclass
class AgentResult:
    response: str
    trajectory: List[Dict[str, Any]]
    tool_call_count: int
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int
    errors: List[str] = field(default_factory=list)


@dataclass
class StrategyResult:
    winner: AgentResult
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int
    temperature: float
    detail: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of tasks from HF dataset (default: 50)",
    )
    parser.add_argument(
        "--db",
        default="results/mcpatlas_combo.db",
        help="SQLite database path (default: results/mcpatlas_combo.db)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/mcpatlas/",
        help="Directory for per-strategy CSVs (default: results/mcpatlas/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show plan without running"
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_think_tags(text: str) -> str:
    if not text:
        return ""
    return THINK_PATTERN.sub("", text).strip()


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


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
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            total += len(message.get("tool_calls") or [])
    return total


def extract_final_assistant_text(messages: Sequence[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message.get("content", ""))
    return ""


def to_csv_trajectory(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            args_raw = function.get("arguments", "{}")
            try:
                parsed = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                parsed = {}
            out.append(
                {
                    "tool_name": str(function.get("name", "")),
                    "parameters": parsed if isinstance(parsed, dict) else {},
                    "response": None,
                    "error": None,
                }
            )
    return out


def run_id(experiment_name: str, strategy: str, task_id: str) -> str:
    key = f"{experiment_name}|{strategy}|{task_id}"
    return f"mcpatlas-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex}"


def strategy_model_meta(
    config: ComboConfig, strategy: str
) -> Tuple[str, float, str, int]:
    if strategy == "best-of-n-small":
        return (
            config.small_model.id,
            config.small_model.params_b,
            config.small_model.arch,
            config.small_model.tp,
        )
    if strategy == "best-of-n-large":
        return (
            config.large_model.id,
            config.large_model.params_b,
            config.large_model.arch,
            config.large_model.tp,
        )
    return (
        f"{config.small_model.id}+{config.large_model.id}",
        config.small_model.params_b + config.large_model.params_b,
        f"{config.small_model.arch}+{config.large_model.arch}",
        config.small_model.tp + config.large_model.tp,
    )


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            experiment_name TEXT NOT NULL,
            model_id TEXT, model_params_b REAL, model_arch TEXT,
            serving_engine TEXT, quantization TEXT,
            tensor_parallel INTEGER, gpu_type TEXT,
            skill_subset TEXT, num_retries INTEGER,
            temperature REAL, agent_mode TEXT,
            task_id TEXT, task_name TEXT, repetition INTEGER,
            task_success BOOLEAN, quality_score REAL,
            input_tokens INTEGER, output_tokens INTEGER,
            gpu_seconds REAL, peak_vram_mb REAL,
            latency_e2e_ms REAL,
            avg_gpu_util_pct REAL, avg_power_w REAL,
            output_text TEXT, trajectory_log TEXT,
            combination_strategy TEXT, combination_detail TEXT,
            tool_call_count INTEGER,
            started_at TEXT, completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_name);
        CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id, quantization);
        CREATE INDEX IF NOT EXISTS idx_runs_combo_resume
            ON runs(experiment_name, combination_strategy, task_id);
        """
    )
    conn.commit()
    return conn


def save_run(conn: sqlite3.Connection, record: Dict[str, Any]) -> None:
    cols = list(record.keys())
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO runs ({', '.join(cols)}) VALUES ({placeholders})"
    conn.execute(sql, [record[col] for col in cols])
    conn.commit()


def existing_task_ids(
    conn: sqlite3.Connection, experiment_name: str, strategy: str
) -> set[str]:
    rows = conn.execute(
        """
        SELECT task_id FROM runs
        WHERE experiment_name = ? AND combination_strategy = ?
        """,
        (experiment_name, strategy),
    ).fetchall()
    return {str(row["task_id"]) for row in rows}


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        if not header_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


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
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with session.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
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


async def run_agent(
    session: aiohttp.ClientSession,
    llm_base_url: str,
    model_name: str,
    task_prompt: str,
    enabled_tools: Sequence[str],
    mcp_server_url: str,
    max_turns: int,
    max_tokens: int,
    temperature: float,
) -> AgentResult:
    tools = await list_openai_tools(session, mcp_server_url, enabled_tools)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    errors: List[str] = []
    input_tokens = 0
    output_tokens = 0
    start = time.perf_counter()
    for _ in range(max_turns):
        result = await chat_completion(
            session,
            llm_base_url,
            model_name,
            messages,
            tools,
            temperature,
            max_tokens,
        )
        usage = result.get("usage") or {}
        input_tokens += int(usage.get("prompt_tokens", 0))
        output_tokens += int(usage.get("completion_tokens", 0))
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError("model returned empty choices")
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
    return AgentResult(
        response=final_response,
        trajectory=messages,
        tool_call_count=count_tool_calls(messages),
        elapsed_seconds=elapsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        errors=errors,
    )


async def self_assess(
    session: aiohttp.ClientSession,
    llm_base_url: str,
    model_name: str,
    task_prompt: str,
    answer: str,
) -> int:
    prompt = (
        "Rate your confidence 0-10 that you fully answered: "
        f"{task_prompt}\n"
        f"Your answer: {answer}"
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
    }
    async with session.post(
        f"{llm_base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 200:
            return 0
        result = await resp.json()
    text = strip_think_tags(
        str((result.get("choices") or [{}])[0].get("message", {}).get("content", ""))
    )
    for found in re.findall(r"\b\d+\b", text):
        value = int(found)
        if 0 <= value <= 10:
            return value
    return 0


def choose_best(items: Sequence[AgentResult], prefer_last_tie: bool) -> AgentResult:
    best = items[0]
    for item in items[1:]:
        key_item = (item.tool_call_count, len(item.response))
        key_best = (best.tool_call_count, len(best.response))
        if key_item > key_best:
            best = item
        elif key_item == key_best and prefer_last_tie:
            best = item
    return best


async def run_strategy(
    session: aiohttp.ClientSession,
    config: ComboConfig,
    strategy: str,
    task_prompt: str,
    enabled_tools: Sequence[str],
) -> StrategyResult:
    small_url = f"http://localhost:{config.small_model.port}"
    large_url = f"http://localhost:{config.large_model.port}"
    attempts: List[AgentResult] = []
    detail: Dict[str, Any] = {"strategy": strategy}
    if strategy == "cascade":
        small = await run_agent(
            session,
            small_url,
            config.small_model.id,
            task_prompt,
            enabled_tools,
            config.mcp_server_url,
            config.max_turns,
            config.max_tokens,
            0.0,
        )
        attempts.append(small)
        escalate = (
            small.tool_call_count == 0
            or len(small.response) < 50
            or small.response.lstrip().upper().startswith("ERROR")
        )
        detail["escalated"] = escalate
        if escalate:
            large = await run_agent(
                session,
                large_url,
                config.large_model.id,
                task_prompt,
                enabled_tools,
                config.mcp_server_url,
                config.max_turns,
                config.max_tokens,
                0.0,
            )
            attempts.append(large)
            winner = large
            detail["winner"] = "large"
        else:
            winner = small
            detail["winner"] = "small"
        temp = 0.0
    elif strategy == "vote":
        small = await run_agent(
            session,
            small_url,
            config.small_model.id,
            task_prompt,
            enabled_tools,
            config.mcp_server_url,
            config.max_turns,
            config.max_tokens,
            0.0,
        )
        large = await run_agent(
            session,
            large_url,
            config.large_model.id,
            task_prompt,
            enabled_tools,
            config.mcp_server_url,
            config.max_turns,
            config.max_tokens,
            0.0,
        )
        attempts.extend([small, large])
        winner = choose_best([small, large], prefer_last_tie=True)
        detail["winner"] = "large" if winner is large else "small"
        temp = 0.0
    elif strategy == "best-of-n-small":
        for _ in range(3):
            attempts.append(
                await run_agent(
                    session,
                    small_url,
                    config.small_model.id,
                    task_prompt,
                    enabled_tools,
                    config.mcp_server_url,
                    config.max_turns,
                    config.max_tokens,
                    0.7,
                )
            )
        winner = choose_best(attempts, prefer_last_tie=False)
        detail["winner"] = "small"
        temp = 0.7
    elif strategy == "best-of-n-large":
        for _ in range(3):
            attempts.append(
                await run_agent(
                    session,
                    large_url,
                    config.large_model.id,
                    task_prompt,
                    enabled_tools,
                    config.mcp_server_url,
                    config.max_turns,
                    config.max_tokens,
                    0.7,
                )
            )
        winner = choose_best(attempts, prefer_last_tie=False)
        detail["winner"] = "large"
        temp = 0.7
    elif strategy == "adaptive-cascade":
        small = await run_agent(
            session,
            small_url,
            config.small_model.id,
            task_prompt,
            enabled_tools,
            config.mcp_server_url,
            config.max_turns,
            config.max_tokens,
            0.0,
        )
        attempts.append(small)
        confidence = await self_assess(
            session,
            small_url,
            config.small_model.id,
            task_prompt,
            small.response,
        )
        detail["small_confidence"] = confidence
        if confidence < 7:
            large = await run_agent(
                session,
                large_url,
                config.large_model.id,
                task_prompt,
                enabled_tools,
                config.mcp_server_url,
                config.max_turns,
                config.max_tokens,
                0.0,
            )
            attempts.append(large)
            winner = large
            detail["winner"] = "large"
        else:
            winner = small
            detail["winner"] = "small"
        temp = 0.0
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    detail["candidates"] = [
        {
            "tool_call_count": item.tool_call_count,
            "response_length": len(item.response),
            "elapsed_seconds": item.elapsed_seconds,
        }
        for item in attempts
    ]
    return StrategyResult(
        winner=winner,
        elapsed_seconds=sum(item.elapsed_seconds for item in attempts),
        input_tokens=sum(item.input_tokens for item in attempts),
        output_tokens=sum(item.output_tokens for item in attempts),
        temperature=temp,
        detail=detail,
    )


def load_rows(num_tasks: int) -> List[Dict[str, Any]]:
    ds = load_dataset("ScaleAI/MCP-Atlas", split="train")
    n = min(max(num_tasks, 0), len(ds))
    if n == 0:
        return []
    sel = ds.select(range(n))
    return [sel[i] for i in range(n)]


def validate_config(config: ComboConfig) -> None:
    if not config.strategies:
        raise ValueError("Config must have strategies")
    bad = [item for item in config.strategies if item not in SUPPORTED_STRATEGIES]
    if bad:
        raise ValueError(f"Unknown strategies: {bad}")


def print_dry_run(config: ComboConfig, task_count: int) -> None:
    print(f"Experiment: {config.name}")
    print(f"Description: {config.description}")
    print(
        f"Small: {config.small_model.id} @ {config.small_model.port} | "
        f"Large: {config.large_model.id} @ {config.large_model.port}"
    )
    print(f"Strategies: {', '.join(config.strategies)}")
    print(f"Dataset rows: {task_count}")
    print(f"Planned runs: {len(config.strategies) * task_count}")


async def run_experiment(args: argparse.Namespace) -> None:
    config = ComboConfig.from_yaml(args.config)
    validate_config(config)
    rows = load_rows(args.num_tasks)
    if args.dry_run:
        print_dry_run(config, len(rows))
        return
    conn = init_db(Path(args.db))
    output_dir = Path(args.output_dir)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=720)
    ) as session:
        for strategy in config.strategies:
            done = existing_task_ids(conn, config.name, strategy)
            csv_file = output_dir / f"{strategy}.csv"
            print(f"\n--- {strategy} ({len(done)} already done) ---")
            for idx, row in enumerate(rows, start=1):
                task_id = str(row.get("TASK", f"mcpatlas-{idx}"))
                prompt = str(row.get("PROMPT", ""))
                if task_id in done:
                    continue
                started_at = now_iso()
                start_perf = time.perf_counter()
                try:
                    strategy_result = await run_strategy(
                        session,
                        config,
                        strategy,
                        prompt,
                        parse_enabled_tools(row.get("ENABLED_TOOLS")),
                    )
                    winner = strategy_result.winner
                    errors = list(winner.errors)
                except Exception as exc:
                    LOGGER.exception("Task failed: %s %s", strategy, task_id)
                    elapsed = time.perf_counter() - start_perf
                    winner = AgentResult(
                        response=f"ERROR: {exc}",
                        trajectory=[],
                        tool_call_count=0,
                        elapsed_seconds=elapsed,
                        input_tokens=0,
                        output_tokens=0,
                        errors=[str(exc)],
                    )
                    strategy_result = StrategyResult(
                        winner=winner,
                        elapsed_seconds=elapsed,
                        input_tokens=0,
                        output_tokens=0,
                        temperature=0.0,
                        detail={"strategy": strategy, "winner": "error"},
                    )
                    errors = [str(exc)]
                completed_at = now_iso()
                csv_row = {
                    "TASK": task_id,
                    "PROMPT": prompt,
                    "TRAJECTORY": to_text(row.get("TRAJECTORY", "")),
                    "GTFA_CLAIMS": to_text(row.get("GTFA_CLAIMS", "")),
                    "ENABLED_TOOLS": to_text(row.get("ENABLED_TOOLS", "[]")),
                    "script_model_response": winner.response,
                    "raw_conversation_history": json.dumps(
                        winner.trajectory, ensure_ascii=False
                    ),
                    "trajectory": json.dumps(
                        to_csv_trajectory(winner.trajectory), ensure_ascii=False
                    ),
                    "errors": json.dumps(errors, ensure_ascii=False),
                    "trajectory_time": strategy_result.elapsed_seconds,
                    "num_retry": 1,
                }
                append_csv(csv_file, csv_row)
                model_id, params_b, arch, tp = strategy_model_meta(config, strategy)
                db_row = {
                    "id": run_id(config.name, strategy, task_id),
                    "experiment_name": config.name,
                    "model_id": model_id,
                    "model_params_b": params_b,
                    "model_arch": arch,
                    "serving_engine": config.serving_engine,
                    "quantization": "fp16",
                    "tensor_parallel": tp,
                    "gpu_type": config.gpu_type,
                    "skill_subset": "all",
                    "num_retries": 0,
                    "temperature": strategy_result.temperature,
                    "agent_mode": "combo",
                    "task_id": task_id,
                    "task_name": task_id,
                    "repetition": 0,
                    "task_success": None,
                    "quality_score": None,
                    "input_tokens": strategy_result.input_tokens,
                    "output_tokens": strategy_result.output_tokens,
                    "gpu_seconds": strategy_result.elapsed_seconds,
                    "peak_vram_mb": 0.0,
                    "latency_e2e_ms": strategy_result.elapsed_seconds * 1000.0,
                    "avg_gpu_util_pct": 0.0,
                    "avg_power_w": 0.0,
                    "output_text": winner.response,
                    "trajectory_log": json.dumps(winner.trajectory, ensure_ascii=False),
                    "combination_strategy": strategy,
                    "combination_detail": json.dumps(
                        strategy_result.detail, ensure_ascii=False
                    ),
                    "tool_call_count": winner.tool_call_count,
                    "started_at": started_at,
                    "completed_at": completed_at,
                }
                save_run(conn, db_row)
                done.add(task_id)
                print(
                    f"[{idx}/{len(rows)}] {task_id} -> "
                    f"tool_calls={winner.tool_call_count}, time={strategy_result.elapsed_seconds:.1f}s"
                )
    conn.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
