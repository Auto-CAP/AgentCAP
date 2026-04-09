#!/usr/bin/env python3
"""Classify MCP-Atlas tasks using rule-based heuristics.

No LLM or API key required. Classification based on structural
signals in task prompts.

Usage:
    python scripts/classify_tasks_heuristic.py
    python scripts/classify_tasks_heuristic.py --output results/task_classifications.jsonl
    python scripts/classify_tasks_heuristic.py --limit 10
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset


PLAN_SIGNALS = [
    (r"in the (same )?(year|month|day|date) (that|when|of)", "temporal_chain"),
    (r"on the (exact )?date (that|when|of)", "temporal_chain"),
    (r"the same (month|year|day) and (year|month|day)", "temporal_chain"),
    (r"(before|after) the", "temporal_comparison"),
    (r"\bif (so|not|there)\b", "conditional"),
    (r"\bis it (higher|lower|more|less|bigger|greater)\b", "conditional"),
    (r"\bif the\b.*\bthen\b", "conditional"),
    (r"\bwhether\b", "conditional"),
    (r"percentage (change|difference)", "computation"),
    (r"year difference", "computation"),
    (r"(straight.?line )?distance (in )?(km|kilometers|miles)", "computation"),
    (r"profit or loss", "computation"),
    (r"how (much|many) .* (more|less) than", "computation"),
    (
        r"(find|identify|tell me) .{10,60} (then|and then|after that|use that|using that)",
        "multi_hop",
    ),
    (r"first .{5,40} then .{5,40}", "multi_hop"),
]

EXEC_SIGNALS = [
    (r"check (the|our|my) (cloud )?(database|data|records|files)", "data_lookup"),
    (r"from (the|our|my) (cloud )?(database|nosql|records)", "data_lookup"),
    (r"according to (the|our|my) (cloud )?(database|data)", "data_lookup"),
    (r"using (the|our|my) (cloud )?(database|nosql)", "data_lookup"),
    (r"\b(average|total|count|sum|how many)\b.*\b(in|for|of|during)\b", "aggregation"),
    (r"\blist (all|the|every)\b", "aggregation"),
    (
        r"translat(e|ion) .{0,30} (to|into) (spanish|french|italian|german|portuguese|bulgarian|japanese|chinese)",
        "translation",
    ),
    (r"(in|to) (spanish|french|italian|german)", "translation"),
    (r"check if .{0,30} (unstaged|pushed|committed)", "file_ops"),
    (
        r"(list|show|get) (the )?files (that |which )?(were |are )?(changed|modified)",
        "file_ops",
    ),
]

DOMAIN_KEYWORDS = {
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "coinbase", "huobi"],
    "sports": [
        "fifa",
        "world cup",
        "nba",
        "nfl",
        "super bowl",
        "champions league",
        "wimbledon",
        "tour de france",
        "dota",
        "wrestling",
    ],
    "literature": ["author", "book", "novel", "published", "wrote", "writer", "poem"],
    "medical": [
        "clinical trial",
        "pubmed",
        "oncology",
        "malaria",
        "vaccine",
        "medical",
        "disease",
    ],
    "geography": [
        "national park",
        "coordinates",
        "elevation",
        "distance",
        "walking time",
        "driving",
    ],
    "entertainment": ["movie", "film", "oscar", "actor", "director", "imdb", "series"],
    "finance": [
        "stock",
        "closing price",
        "opening price",
        "market cap",
        "apple stock",
        "nvidia",
    ],
    "art": ["museum", "painting", "artwork", "artist", "collection", "metropolitan"],
}


def count_domains(prompt_lower: str) -> set[str]:
    domains_found: set[str] = set()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in prompt_lower:
                domains_found.add(domain)
                break
    return domains_found


def classify_task(prompt: str, tools: list[str]) -> tuple[str, str]:
    del tools

    prompt_lower = prompt.lower()

    plan_score = 0
    exec_score = 0
    plan_reasons: list[str] = []
    exec_reasons: list[str] = []

    for pattern, signal_type in PLAN_SIGNALS:
        matches = re.findall(pattern, prompt_lower)
        if matches:
            plan_score += len(matches)
            plan_reasons.append(f"{signal_type}({len(matches)})")

    for pattern, signal_type in EXEC_SIGNALS:
        matches = re.findall(pattern, prompt_lower)
        if matches:
            exec_score += len(matches)
            exec_reasons.append(f"{signal_type}({len(matches)})")

    domains = count_domains(prompt_lower)
    if len(domains) >= 3:
        plan_score += 2
        plan_reasons.append(f"cross_domain({','.join(sorted(domains))})")
    elif len(domains) >= 2:
        plan_score += 1
        plan_reasons.append(f"multi_domain({','.join(sorted(domains))})")

    if plan_score >= 3 and plan_score > exec_score * 1.5:
        classification = "plan-heavy"
    elif exec_score >= 2 and exec_score > plan_score * 1.5:
        classification = "execute-heavy"
    elif plan_score > exec_score:
        classification = "balanced"
    elif exec_score > plan_score:
        classification = "balanced"
    else:
        classification = "balanced"

    reasoning_parts = []
    if plan_reasons:
        reasoning_parts.append(f"plan_signals=[{','.join(plan_reasons)}]")
    if exec_reasons:
        reasoning_parts.append(f"exec_signals=[{','.join(exec_reasons)}]")
    reasoning = f"plan={plan_score},exec={exec_score}; {'; '.join(reasoning_parts)}"

    return classification, reasoning


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify MCP-Atlas tasks using rule-based heuristics (no LLM needed)."
    )
    parser.add_argument(
        "--output", type=str, default="results/task_classifications_heuristic.jsonl"
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    print("Loading MCP-Atlas dataset...")
    ds = load_dataset("ScaleAI/mcp-atlas", split="train")
    tasks = list(ds)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"plan-heavy": 0, "execute-heavy": 0, "balanced": 0}

    with output_path.open("w", encoding="utf-8") as f:
        for i, task_raw in enumerate(tasks):
            task = cast(dict[str, Any], task_raw)

            prompt_value = task.get("PROMPT", "")
            prompt = (
                prompt_value if isinstance(prompt_value, str) else str(prompt_value)
            )

            tools = task.get("ENABLED_TOOLS", [])
            if isinstance(tools, str):
                try:
                    tools = json.loads(tools)
                except json.JSONDecodeError:
                    tools = []
            if isinstance(tools, list):
                tool_list = [str(item) for item in tools]
            else:
                tool_list = []

            classification, reasoning = classify_task(prompt, tool_list)
            counts[classification] += 1

            task_id_value = task.get("TASK", "")
            task_id = (
                task_id_value if isinstance(task_id_value, str) else str(task_id_value)
            )

            line = {
                "index": i,
                "task_id": task_id,
                "prompt": prompt[:200],
                "classification": classification,
                "reasoning": reasoning,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("\n=== Classification Results (Heuristic) ===")
    total = sum(counts.values())
    for label, count in sorted(counts.items()):
        print(f"  {label:14s}: {count:4d} ({count / total * 100:.1f}%)")
    print(f"  {'total':14s}: {total:4d}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
