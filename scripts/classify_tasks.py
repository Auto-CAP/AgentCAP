#!/usr/bin/env python3
"""Classify MCP-Atlas tasks as plan-heavy or execute-heavy.

Uses an LLM (via OpenRouter) to classify each task based on prompt text
and enabled tools. Outputs a JSONL file with annotations.

Usage:
    python scripts/classify_tasks.py --api-key sk-xxx
    python scripts/classify_tasks.py --api-key sk-xxx --model google/gemini-2.5-flash --output results/task_classifications.jsonl
    python scripts/classify_tasks.py --api-key sk-xxx --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiohttp
from datasets import load_dataset


CLASSIFICATION_PROMPT = """You are classifying agentic AI benchmark tasks.

For each task, decide: is the main difficulty in PLANNING or EXECUTING?

**plan-heavy**: The agent needs to think hard before it can act. The task requires complex reasoning, problem decomposition, mathematical derivation, or figuring out a non-obvious strategy. Even if execution is trivial, the planning is hard. Examples: mathematical problems, multi-step logic where you must combine information from unrelated domains, tasks where you must derive intermediate results before knowing what to do next, conditional decision-making.

**execute-heavy**: The agent knows what to do (or can figure it out quickly), but carrying it out is the hard part. Execution involves substantial tool interactions, modifying large amounts of code, processing many records, navigating complex APIs, or coordinating multiple long tool-call sequences. The plan may be simple but the execution workload is heavy. Examples: fixing a bug that spans many files, retrieving and aggregating data from multiple sources, long multi-turn tool-call chains, large code modifications.

You MUST pick exactly one. If the task has elements of both, decide which difficulty DOMINATES.

Respond in JSON only: {{"classification": "plan-heavy|execute-heavy", "reasoning": "one sentence"}}

Task: {prompt}

Tools: {tools}"""


VALID_LABELS = {"plan-heavy", "execute-heavy"}


def parse_tools(raw_tools: Any) -> list[str]:
    if isinstance(raw_tools, list):
        return [str(t) for t in raw_tools]

    if isinstance(raw_tools, str):
        stripped = raw_tools.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except json.JSONDecodeError:
            pass
        return [part.strip() for part in stripped.split(",") if part.strip()]

    return [str(raw_tools)] if raw_tools is not None else []


def parse_model_json(content: str) -> dict[str, str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    classification = str(parsed.get("classification", "")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()
    if classification not in VALID_LABELS:
        raise ValueError(f"Invalid classification: {classification}")

    return {"classification": classification, "reasoning": reasoning}


async def classify_one(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    task_prompt: str,
    tools: list[str],
    retries: int = 3,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": CLASSIFICATION_PROMPT.format(
                    prompt=task_prompt,
                    tools=", ".join(tools) if tools else "(none)",
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(retries):
        try:
            async with session.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                if resp.status == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                if resp.status >= 500:
                    wait = 2 ** (attempt + 1)
                    print(f"  Server error {resp.status}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                if resp.status != 200:
                    body = await resp.text()
                    print(f"  Error {resp.status}: {body[:200]}")
                    await asyncio.sleep(1)
                    continue

                result = await resp.json()
                content = result["choices"][0]["message"]["content"]
                return parse_model_json(content)

        except (
            json.JSONDecodeError,
            ValueError,
            KeyError,
            aiohttp.ClientError,
            asyncio.TimeoutError,
        ) as exc:
            print(f"  Attempt {attempt + 1} failed: {exc}")
            await asyncio.sleep(1)

    return {"classification": "error", "reasoning": f"Failed after {retries} retries"}


def load_existing_results(output_path: Path) -> tuple[set[int], dict[str, int]]:
    done_indices: set[int] = set()
    counts = {"plan-heavy": 0, "execute-heavy": 0, "balanced": 0, "error": 0}

    if not output_path.exists():
        return done_indices, counts

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            index = row.get("index")
            if isinstance(index, int):
                done_indices.add(index)

            label = str(row.get("classification", "error"))
            counts[label] = counts.get(label, 0) + 1

    return done_indices, counts


async def classify_batch(
    tasks: Sequence[Mapping[str, Any]],
    base_url: str,
    api_key: str,
    model: str,
    output_path: str,
    concurrency: int = 5,
) -> dict[str, int]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    done_indices, counts = load_existing_results(path)
    pending = [(idx, task) for idx, task in enumerate(tasks) if idx not in done_indices]

    if done_indices:
        print(
            f"Resuming from existing output: {len(done_indices)} task(s) already done."
        )

    write_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    completed = 0

    async with aiohttp.ClientSession() as session:
        with path.open("a", encoding="utf-8") as out:

            async def worker(index: int, task: Mapping[str, Any]) -> None:
                nonlocal completed

                task_id = task.get("TASK", f"task_{index}")
                prompt = str(task.get("PROMPT", ""))
                tools = parse_tools(task.get("ENABLED_TOOLS", []))

                async with semaphore:
                    result = await classify_one(
                        session=session,
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        task_prompt=prompt,
                        tools=tools,
                    )

                classification = result.get("classification", "error")
                reasoning = result.get("reasoning", "")

                line = {
                    "index": index,
                    "task_id": task_id,
                    "prompt": prompt[:200],
                    "classification": classification,
                    "reasoning": reasoning,
                }

                async with write_lock:
                    out.write(json.dumps(line, ensure_ascii=False) + "\n")
                    out.flush()
                    counts[classification] = counts.get(classification, 0) + 1

                async with progress_lock:
                    completed += 1
                    print(
                        f"[{len(done_indices) + completed}/{len(tasks)}] "
                        f"{classification:14s} | {prompt[:80]}..."
                    )

            await asyncio.gather(*(worker(idx, task) for idx, task in pending))

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify MCP-Atlas tasks as plan-heavy, execute-heavy, or balanced."
    )
    parser.add_argument("--api-key", type=str, required=True, help="OpenRouter API key")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.5-flash",
        help="Model to use for classification (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="API base URL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/task_classifications.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of tasks (0 = all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent API calls",
    )
    args = parser.parse_args()

    print("Loading MCP-Atlas dataset...")
    dataset = load_dataset("ScaleAI/mcp-atlas", split="train")
    tasks: list[Mapping[str, Any]] = []
    for item in dataset:
        if isinstance(item, Mapping):
            tasks.append(item)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"Classifying {len(tasks)} task(s) with {args.model}...")
    counts = asyncio.run(
        classify_batch(
            tasks=tasks,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            output_path=args.output,
            concurrency=args.concurrency,
        )
    )

    print("\n=== Classification Results ===")
    total = sum(counts.values())
    for label, count in sorted(counts.items()):
        pct = (count / total * 100.0) if total else 0.0
        print(f"  {label:14s}: {count:4d} ({pct:.1f}%)")
    print(f"  {'total':14s}: {total:4d}")
    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
