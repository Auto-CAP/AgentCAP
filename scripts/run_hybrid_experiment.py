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
    compute_local_cost_runtime,
)

_PYTHON_BACKEND = None


def _get_python_backend():
    global _PYTHON_BACKEND
    if _PYTHON_BACKEND is None:
        from agent_cap.backends.math_python_backend import (
            MathPythonBackend,
            PYTHON_TOOL_DEFINITIONS,
        )
        _PYTHON_BACKEND = MathPythonBackend()
        _PYTHON_BACKEND.setup({})
    return _PYTHON_BACKEND

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

IMO_SYSTEM_PROMPT_2 = (
    "You are an expert mathematical problem solver with expertise at the IMO level, "
    "with access to a persistent Python (with sympy) execution tool.\n\n"
    "Rules:\n"
    "- Do NOT rely on pure chain-of-thought arithmetic for numeric, symbolic, or combinatorial steps. "
    "Whenever a step can be computed exactly with Python or sympy, call the `python` tool.\n"
    "- When exploring, avoid pure brute force over large unstructured spaces; instead, use python to "
    "verify identities, check small cases, confirm conjectures, enumerate well-chosen parametric families, "
    "and simplify or factor symbolic expressions via sympy.\n"
    "- Always import what you need fresh each call (the kernel is persistent, but state may be overwritten). "
    "Always `print(...)` the key intermediate and final values so the tool output is useful.\n"
    "- Incorporate each tool output back into your reasoning before the next step.\n"
    "- Justify non-trivial steps in text; do not skip proofs.\n"
    "- If you discover the answer depends on a parameter (e.g., m, n, k), confirm the functional form "
    "by checking multiple values, not just one.\n"
    "- Put your final answer inside \\boxed{...}."
)

IMO_SYSTEM_PROMPT = """You are an elite mathematical problem solver with expertise at the International Mathematical Olympiad (IMO) level.

# Output Format
- Provide a brief summary of the solution.
- Then state the final mathematical answer clearly.
- Put the final answer inside \\boxed{...}.
- The final answer may be an integer, fraction, expression, tuple, sequence, set, or other mathematical object, depending on the problem.
- Do not put anything except the final answer inside the final \\boxed{...}.
"""

IMO_PLAN_SYSTEM_PROMPT_2 = (
    "You are an elite mathematical problem-solving planner for IMO-level problems. "
    "The executor has access to a single `python` tool that runs Python/sympy in a "
    "persistent Jupyter kernel. Given a problem, output a short numbered plan the "
    "executor can follow. For each step, state (a) whether to call python, (b) what "
    "exact quantity/identity/expression to compute or verify, and (c) how the result "
    "feeds the next step.\n\n"
    "Strong preference rules for the plan:\n"
    "- Insist that the executor use python for numeric/symbolic/combinatorial checks "
    "rather than pure mental arithmetic.\n"
    "- If the answer likely depends on a parameter (m, n, k, ...), explicitly require "
    "verifying the pattern on at least 3 different parameter values before claiming a "
    "closed form.\n"
    "- Do NOT recommend naive brute force over huge unstructured spaces; prefer "
    "parametric families, symbolic simplification, or well-motivated small cases.\n"
    "- Keep the plan crisp: 4--8 steps. Do NOT solve the problem yourself; only plan."
)

IMO_PLAN_SYSTEM_PROMPT = (
    "You are an elite mathematical problem-solving planner for IMO-level problems. "
    "The first step is to draft a plan for the solution, outlining the main ideas only."
    "Keep the plan crisp: 4--8 steps. Do NOT solve the problem yourself; only plan."
)

IMO_EXEC_PROMPT = (
    "Problem:\n{task}\n\n"
    "You are provided with a solution plan."
    "The plan may be wrong or contain errors."
    "Plan:\n{plan}\n\n"
    "Solve the problem and output the solution in the correct format, together with a solution summary."
    "Put the final answer inside \\boxed{{...}}."
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
            input_price_per_1m=float(raw.get("input_price_per_1m", 0.0)),
            output_price_per_1m=float(raw.get("output_price_per_1m", 0.0)),
        )

    def cost_config(self) -> APICostConfig | LocalCostConfig:
        if self.is_local:
            return LocalCostConfig(model_id=self.id)
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
    total_prefill_seconds: float = 0.0
    total_decode_seconds: float = 0.0


@dataclass
class ChatCompletionTimedResult:
    response_json: Dict[str, Any]
    ttft_seconds: float
    decode_seconds: float


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


_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}


def _coerce_value(value: Any, schema: Optional[Dict[str, Any]]) -> Any:
    """Best-effort coerce LLM-emitted value to schema-declared JSON type."""
    if not isinstance(schema, dict):
        return value
    types = schema.get("type")
    if isinstance(types, list):
        type_set = [t for t in types if isinstance(t, str)]
    elif isinstance(types, str):
        type_set = [types]
    else:
        type_set = []

    if isinstance(value, str):
        if "integer" in type_set:
            try:
                return int(value)
            except ValueError:
                pass
        if "number" in type_set:
            try:
                return float(value)
            except ValueError:
                pass
        if "boolean" in type_set:
            lv = value.strip().lower()
            if lv in {"true", "1", "yes"}:
                return True
            if lv in {"false", "0", "no"}:
                return False
        if "array" in type_set and value.strip().startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        if "object" in type_set and value.strip().startswith("{"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
    if isinstance(value, dict) and "object" in type_set:
        props = schema.get("properties") or {}
        return {k: _coerce_value(v, props.get(k)) for k, v in value.items()}
    if isinstance(value, list) and "array" in type_set:
        items_schema = schema.get("items") if isinstance(schema.get("items"), dict) else None
        return [_coerce_value(v, items_schema) for v in value]
    return value


_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _coerce_time_field(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    kl = key.lower()
    is_time_field = (
        kl in {"timemin", "time_min", "timemax", "time_max", "start", "end",
               "starttime", "start_time", "endtime", "end_time", "after", "before"}
    )
    if is_time_field and _DATE_ONLY_RE.match(value.strip()):
        suffix = "T23:59:59Z" if "max" in kl or kl in {"end", "endtime", "end_time", "before"} else "T00:00:00Z"
        return value.strip() + suffix
    return value


def coerce_tool_args(tool_name: str, args: Any) -> Any:
    schema = _SCHEMA_CACHE.get(tool_name)
    if not isinstance(args, dict):
        return args
    if not schema:
        return {k: _coerce_time_field(k, v) for k, v in args.items()}
    props = schema.get("properties") or {}
    return {
        k: _coerce_time_field(k, _coerce_value(v, props.get(k)))
        for k, v in args.items()
    }


async def list_openai_tools(
    session: aiohttp.ClientSession,
    mcp_server_url: str,
    enabled_tools: Sequence[str],
) -> List[Dict[str, Any]]:
    if enabled_tools and "__PYTHON_TOOL__" in enabled_tools:
        from agent_cap.backends.math_python_backend import PYTHON_TOOL_DEFINITIONS
        for tool in PYTHON_TOOL_DEFINITIONS:
            schema = tool.get("function", {}).get("parameters") or {}
            _SCHEMA_CACHE[tool["function"]["name"]] = schema
        return list(PYTHON_TOOL_DEFINITIONS)
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
        schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        _SCHEMA_CACHE[name] = schema
        transformed.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description", "")),
                    "parameters": schema,
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
    needs_temp_1 = "kimi" in model.lower() or "moonshot" in base_url
    token_key = "max_completion_tokens" if is_openai else "max_tokens"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 1.0 if needs_temp_1 else temperature,
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


async def chat_completion_streaming(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    max_tokens: int,
    temperature: float = 0.0,
) -> ChatCompletionTimedResult:
    headers = {}
    if api_key and api_key != "dummy":
        headers["Authorization"] = f"Bearer {api_key}"

    is_openai = "api.openai.com" in base_url
    needs_temp_1 = "kimi" in model.lower() or "moonshot" in base_url
    token_key = "max_completion_tokens" if is_openai else "max_tokens"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 1.0 if needs_temp_1 else temperature,
        token_key: max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools

    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    t_last_token = t_start

    collected_content = ""
    collected_tool_calls: Dict[int, Dict[str, Any]] = {}
    finish_reason = None
    usage: Dict[str, Any] = {}

    async with session.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"chat failed ({resp.status}): {body}")

        async for raw_line in resp.content:
            text = raw_line.decode("utf-8").strip()
            if not text or not text.startswith("data:"):
                continue
            data_str = text[5:].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            now = time.perf_counter()
            choices = chunk.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}

                content_delta = delta.get("content")
                if content_delta:
                    if t_first_token is None:
                        t_first_token = now
                    t_last_token = now
                    collected_content += content_delta

                tool_call_deltas = delta.get("tool_calls") or []
                if tool_call_deltas:
                    if t_first_token is None:
                        t_first_token = now
                    t_last_token = now
                    for tc_delta in tool_call_deltas:
                        idx = int(tc_delta.get("index", 0))
                        if idx not in collected_tool_calls:
                            collected_tool_calls[idx] = {
                                "id": tc_delta.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.get("id"):
                            collected_tool_calls[idx]["id"] = tc_delta["id"]
                        fn = tc_delta.get("function") or {}
                        if fn.get("name"):
                            collected_tool_calls[idx]["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            collected_tool_calls[idx]["function"]["arguments"] += fn[
                                "arguments"
                            ]

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

            if chunk.get("usage"):
                usage = chunk["usage"]

    if t_first_token is None:
        t_first_token = t_start

    ttft = t_first_token - t_start
    decode_time = t_last_token - t_first_token

    assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": collected_content or None,
    }
    if collected_tool_calls:
        assistant_message["tool_calls"] = [
            collected_tool_calls[i] for i in sorted(collected_tool_calls.keys())
        ]

    response_json = {
        "choices": [{"message": assistant_message, "finish_reason": finish_reason}],
        "usage": usage,
    }
    return ChatCompletionTimedResult(
        response_json=response_json,
        ttft_seconds=ttft,
        decode_seconds=decode_time,
    )


MAX_TOOL_RESULT_CHARS = 40000


def _truncate_tool_result(payload: Any) -> Any:
    if isinstance(payload, list):
        total = 0
        out = []
        for item in payload:
            if isinstance(item, dict) and "text" in item:
                text = str(item.get("text", ""))
                if total + len(text) > MAX_TOOL_RESULT_CHARS:
                    allowed = max(0, MAX_TOOL_RESULT_CHARS - total)
                    item = dict(item)
                    item["text"] = (
                        text[:allowed]
                        + f"\n\n[...TRUNCATED: tool result was {len(text)} chars, kept first {allowed}]"
                    )
                    out.append(item)
                    break
                total += len(text)
            out.append(item)
        return out
    return payload


async def mcp_call_tool(
    session: aiohttp.ClientSession,
    mcp_server_url: str,
    tool_name: str,
    tool_args: Any,
) -> Any:
    if tool_name == "python":
        backend = _get_python_backend()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: backend.execute(tool_name, "tc", tool_args if isinstance(tool_args, dict) else {})
        )
        text = result.output if not result.success else (result.output or "(no output)")
        return _truncate_tool_result([{"type": "text", "text": text}])
    async with session.post(
        f"{mcp_server_url.rstrip('/')}/call-tool",
        json={"tool_name": tool_name, "tool_args": tool_args},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"tool call failed ({resp.status}): {body}")
        return _truncate_tool_result(await resp.json())


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

    is_imo = "__PYTHON_TOOL__" in set(enabled_tools)
    plan_system = IMO_PLAN_SYSTEM_PROMPT if is_imo else PLAN_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": plan_system},
        {"role": "user", "content": user_content},
    ]

    start = time.perf_counter()
    if planner.is_local:
        timed = await chat_completion_streaming(
            session,
            planner.base_url,
            planner.api_key,
            planner.id,
            messages,
            None,
            plan_max_tokens,
            temperature=0.0,
        )
        result = timed.response_json
        prefill_time = timed.ttft_seconds
        decode_time = timed.decode_seconds
    else:
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
        prefill_time = 0.0
        decode_time = 0.0
    elapsed = time.perf_counter() - start

    usage = result.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))

    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError("planner returned empty choices")
    assistant = choices[0].get("message") or {}
    plan_text = strip_think_tags(str(assistant.get("content", "")))

    # print(f'[ASSITANT PLAN LINE 719] {assistant}', flush = True)

    messages.append(assistant)

    return PhaseResult(
        response=plan_text,
        messages=messages,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
        tool_call_count=0,
        total_prefill_seconds=prefill_time,
        total_decode_seconds=decode_time,
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
    is_imo = "__PYTHON_TOOL__" in set(enabled_tools)

    if plan_text:
        if is_imo:
            user_content = IMO_EXEC_PROMPT.format(task=task_prompt, plan=plan_text)
        else:
            user_content = EXEC_WITH_PLAN_PROMPT.format(task=task_prompt, plan=plan_text)
    else:
        user_content = task_prompt

    exec_system = IMO_SYSTEM_PROMPT if is_imo else SYSTEM_PROMPT
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": exec_system},
        {"role": "user", "content": user_content},
    ]

    errors: List[str] = []
    input_tokens = 0
    output_tokens = 0
    total_prefill_time = 0.0
    total_decode_time = 0.0
    start = time.perf_counter()

    for turn in range(max_turns):
        try:
            if executor.is_local:
                timed = await chat_completion_streaming(
                    session,
                    executor.base_url,
                    executor.api_key,
                    executor.id,
                    messages,
                    tools,
                    max_tokens,
                    temperature=0.0,
                )
                result = timed.response_json
                total_prefill_time += timed.ttft_seconds
                total_decode_time += timed.decode_seconds
            else:
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
        # print(f'[ASSITANT THINKING LINE 817] {assistant}', flush = True)

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
                if name == "python" and isinstance(raw_args, str) and raw_args.strip():
                    args = {"code": raw_args}
                else:
                    args = {}
            if not isinstance(args, dict):
                args = {"code": str(args)} if name == "python" else {}
            args = coerce_tool_args(name, args)
            LOGGER.info(
                "  Tool call turn=%d name=%s args=%s",
                turn,
                name,
                json.dumps(args, ensure_ascii=False),
            )


            try:
                tool_payload = await mcp_call_tool(session, mcp_server_url, name, args)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                tool_payload = [{"type": "text", "text": f"ERROR: {exc}"}]

            tool_output= flatten_tool_payload(tool_payload)
            LOGGER.info(
                "  Tool output turn=%d name=%s output=%s",
                turn,
                name,
                tool_output,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": flatten_tool_payload(tool_output),
                }
            )

    needs_finalization = False
    if messages:
        last = messages[-1]
        if last.get("role") == "tool":
            needs_finalization = True
        elif last.get("role") == "assistant" and last.get("tool_calls"):
            needs_finalization = True

    if needs_finalization:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Now provide the final answer to the original task using the "
                    "tool results above. Do not call additional tools."
                ),
            }
        )
        try:
            if executor.is_local:
                timed = await chat_completion_streaming(
                    session,
                    executor.base_url,
                    executor.api_key,
                    executor.id,
                    messages,
                    None,
                    max_tokens,
                    temperature=0.0,
                )
                final_result = timed.response_json
                total_prefill_time += timed.ttft_seconds
                total_decode_time += timed.decode_seconds
            else:
                final_result = await chat_completion(
                    session,
                    executor.base_url,
                    executor.api_key,
                    executor.id,
                    messages,
                    None,
                    max_tokens,
                    temperature=0.0,
                )
            usage = final_result.get("usage") or {}
            input_tokens += int(usage.get("prompt_tokens", 0))
            output_tokens += int(usage.get("completion_tokens", 0))
            choices = final_result.get("choices") or []
            if choices:
                assistant = choices[0].get("message") or {}
                messages.append(assistant)
        except Exception as exc:
            errors.append(f"finalize: {exc}")

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
        total_prefill_seconds=total_prefill_time,
        total_decode_seconds=total_decode_time,
    )


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


def compute_cost(
    model_cfg: ModelConfig,
    in_tokens: int,
    out_tokens: int,
    total_prefill_seconds: float = 0.0,
    total_decode_seconds: float = 0.0,
) -> float:
    """Compute cost in USD for a phase."""
    cfg = model_cfg.cost_config()
    if isinstance(cfg, APICostConfig):
        return compute_api_cost(cfg, in_tokens, out_tokens)
    return compute_local_cost_runtime(
        cfg,
        in_tokens,
        out_tokens,
        total_prefill_seconds,
        total_decode_seconds,
    )


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


def load_tasks(num_tasks: int, dataset: str = "mcp-atlas") -> List[Dict[str, Any]]:
    if dataset == "mcp-atlas":
        ds = load_dataset("ScaleAI/MCP-Atlas", split="train")
        n = min(max(num_tasks, 0), len(ds))
        sel = ds.select(range(n)) if n > 0 else []
        return [dict(sel[i]) for i in range(n)]
    if dataset == "imo-answerbench":
        ds = load_dataset("Hwilner/imo-answerbench", split="train")
        by_cat: Dict[str, List[Any]] = {}
        for ex in ds:
            by_cat.setdefault(ex.get("Category", "Other"), []).append(ex)
        categories = sorted(by_cat.keys())
        per_cat = max(1, num_tasks // max(1, len(categories)))
        out = []
        for cat in categories:
            for ex in by_cat[cat][:per_cat]:
                prompt = (
                    ex["Problem"]
                    + "\n\nSolve this step by step. "
                    + r"Place your final answer inside \boxed{}."
                )
                out.append({
                    "TASK": ex.get("Problem ID", f"imo-{cat}-{len(out)}"),
                    "PROMPT": prompt,
                    "ENABLED_TOOLS": "[]",
                    "GTFA_CLAIMS": [],
                    "expected_answer": str(ex.get("Short Answer", "")).strip(),
                    "Category": cat,
                })
        return out[:num_tasks]
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


async def run_experiment(args: argparse.Namespace) -> None:
    config = HybridConfig.from_yaml(args.config)
    tasks = load_tasks(args.num_tasks, dataset=args.dataset)

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
            if args.dataset == "imo-answerbench":
                enabled_tools = ["__PYTHON_TOOL__"]

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
                    plan_result.total_prefill_seconds,
                    plan_result.total_decode_seconds,
                )
            exec_cost = compute_cost(
                config.executor,
                exec_result.input_tokens,
                exec_result.output_tokens,
                exec_result.total_prefill_seconds,
                exec_result.total_decode_seconds,
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
                "exec_response": exec_result.response[:50000],
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
        "--dataset",
        default="mcp-atlas",
        choices=["mcp-atlas", "imo-answerbench"],
        help="Which benchmark to run (default: mcp-atlas)",
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
