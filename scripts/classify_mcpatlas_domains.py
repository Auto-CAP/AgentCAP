#!/usr/bin/env python3
"""Rules-based domain classification for MCP-Atlas tasks.

Loads the MCP-Atlas train split, classifies each task into one or more domains
using tool-server and prompt-keyword signals, and writes JSONL output to:

    results/mcpatlas_domain_classifications.jsonl

Usage:
    python scripts/classify_mcpatlas_domains.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from datasets import load_dataset


EXPECTED_TASKS = 500
OUTPUT_REL_PATH = Path("results/mcpatlas_domain_classifications.jsonl")


DOMAINS: dict[str, dict[str, list[str]]] = {
    "software_engineering": {
        "tool_signals": [
            "github",
            "git",
            "mcp-code-executor",
            "cli-mcp-server",
            "e2b-server",
            "mcp-server-code-runner",
            "context7",
        ],
        "keyword_signals": [
            "repo",
            "repository",
            "commit",
            "code",
            "programming",
            "developer",
            "api",
            "github",
            "pull request",
            "branch",
            "debug",
        ],
    },
    "finance": {
        "tool_signals": ["twelvedata", "alchemy"],
        "keyword_signals": [
            "stock",
            "market",
            "price",
            "trading",
            "portfolio",
            "invest",
            "crypto",
            "bitcoin",
            "ethereum",
            "share price",
            "ticker",
            "revenue",
            "profit",
            "financial",
            "earnings",
            "dividend",
        ],
    },
    "biomedical": {
        "tool_signals": ["pubmed", "clinicaltrialsgov-mcp-server"],
        "keyword_signals": [
            "clinical trial",
            "pubmed",
            "medical",
            "disease",
            "drug",
            "patient",
            "treatment",
            "health",
            "cancer",
            "gene",
            "protein",
            "pharmaceutical",
            "diagnosis",
        ],
    },
    "geography_travel": {
        "tool_signals": [
            "osm-mcp-server",
            "google-maps",
            "national-parks",
            "weather",
            "weather-data",
        ],
        "keyword_signals": [
            "distance",
            "location",
            "map",
            "coordinate",
            "travel",
            "visit",
            "park",
            "restaurant",
            "hotel",
            "walking",
            "driving",
            "near",
            "weather",
            "temperature",
            "latitude",
            "longitude",
        ],
    },
    "arts_culture": {
        "tool_signals": ["met-museum", "rijksmuseum-server"],
        "keyword_signals": [
            "museum",
            "painting",
            "artist",
            "artwork",
            "sculpture",
            "exhibition",
            "gallery",
            "art department",
        ],
    },
    "knowledge_research": {
        "tool_signals": [
            "wikipedia",
            "ddg-search",
            "fetch",
            "arxiv",
            "open-library",
            "whois",
            "oxylabs",
            "exa",
        ],
        "keyword_signals": [
            "wikipedia",
            "search",
            "article",
            "book",
            "author",
            "published",
            "research",
            "paper",
            "history",
        ],
    },
    "data_analysis": {
        "tool_signals": [
            "filesystem",
            "desktop-commander",
            "mongodb",
            "airtable",
            "calculator",
        ],
        "keyword_signals": [
            "file",
            "csv",
            "database",
            "average",
            "total",
            "percentage",
            "calculate",
            "data",
            "spreadsheet",
            "statistics",
            "log",
            "records",
        ],
    },
    "productivity": {
        "tool_signals": ["notion", "slack", "google-workspace", "memory"],
        "keyword_signals": [
            "notion",
            "slack",
            "message",
            "task",
            "project",
            "team",
            "workspace",
            "calendar",
            "email",
        ],
    },
    "sports": {
        "tool_signals": ["balldontlie"],
        "keyword_signals": [
            "nba",
            "basketball",
            "football",
            "soccer",
            "player",
            "team",
            "championship",
            "league",
            "score",
            "game",
            "match",
            "season",
            "world cup",
            "champions league",
            "olympics",
        ],
    },
    "language_translation": {
        "tool_signals": ["lara-translate"],
        "keyword_signals": [
            "translate",
            "translation",
            "language",
            "french",
            "spanish",
            "german",
            "chinese",
            "japanese",
        ],
    },
}


def parse_tools(raw_tools: Any) -> list[str]:
    """Parse ENABLED_TOOLS into a list of tool names."""
    if isinstance(raw_tools, list):
        return [str(tool) for tool in raw_tools]

    if isinstance(raw_tools, str):
        stripped = raw_tools.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            return [str(tool) for tool in parsed]

        return [part.strip() for part in stripped.split(",") if part.strip()]

    return [str(raw_tools)] if raw_tools is not None else []


def extract_tool_servers(enabled_tools: list[str]) -> list[str]:
    """Extract unique tool server prefixes (substring before first underscore)."""
    servers: list[str] = []
    seen: set[str] = set()
    for tool_name in enabled_tools:
        prefix = str(tool_name).split("_", 1)[0].strip().lower()
        if prefix and prefix not in seen:
            seen.add(prefix)
            servers.append(prefix)
    return servers


def score_domains(prompt: str, tool_servers: list[str]) -> dict[str, int]:
    """Compute domain scores using fixed signal rules."""
    prompt_lower = prompt.lower()
    server_set = set(tool_servers)
    scores: dict[str, int] = {}

    for domain, cfg in DOMAINS.items():
        score = 0

        for tool_signal in cfg["tool_signals"]:
            if tool_signal.lower() in server_set:
                score += 2

        for keyword_signal in cfg["keyword_signals"]:
            if keyword_signal.lower() in prompt_lower:
                score += 1

        scores[domain] = score

    return scores


def classify_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a single MCP-Atlas task by domain."""
    task_id = str(task.get("TASK", ""))
    prompt = str(task.get("PROMPT", ""))
    enabled_tools = parse_tools(task.get("ENABLED_TOOLS", []))
    tool_servers = extract_tool_servers(enabled_tools)

    scores = score_domains(prompt=prompt, tool_servers=tool_servers)
    qualifying_domains = [domain for domain, score in scores.items() if score >= 2]

    if qualifying_domains:
        max_score = max(scores[domain] for domain in qualifying_domains)
        primary_domain = next(
            domain for domain in DOMAINS.keys() if scores[domain] == max_score
        )
        domains = qualifying_domains
        domain_scores = {domain: score for domain, score in scores.items() if score > 0}
    else:
        primary_domain = "general"
        domains = ["general"]
        domain_scores = {"general": 0}

    return {
        "task_id": task_id,
        "primary_domain": primary_domain,
        "domains": domains,
        "domain_scores": domain_scores,
        "tool_servers": tool_servers,
        "prompt_preview": prompt[:150],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def print_summary(records: list[dict[str, Any]]) -> None:
    total = len(records)
    primary_counts = Counter(record["primary_domain"] for record in records)
    coverage_counts = Counter()
    for record in records:
        for domain in record["domains"]:
            coverage_counts[domain] += 1

    multi_domain_count = sum(1 for record in records if len(record["domains"]) > 1)
    average_domains = (
        sum(len(record["domains"]) for record in records) / total if total else 0.0
    )

    print("=== MCP-Atlas Domain Classification ===")
    print(f"Total tasks: {total}\n")

    print("Primary Domain Distribution:")
    for domain, count in sorted(primary_counts.items(), key=lambda item: (-item[1], item[0])):
        pct = (count / total * 100.0) if total else 0.0
        print(f"  {domain:24s} {count:4d} ({pct:.1f}%)")

    print("\nDomain Coverage (tasks touching each domain):")
    for domain, count in sorted(
        coverage_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        pct = (count / total * 100.0) if total else 0.0
        print(f"  {domain:24s} {count:4d} ({pct:.1f}%)")

    multi_pct = (multi_domain_count / total * 100.0) if total else 0.0
    print(f"\nMulti-domain tasks: {multi_domain_count}/{total} ({multi_pct:.1f}%)")
    print(f"Average domains per task: {average_domains:.1f}")


def verify_integrity(
    source_tasks: list[Mapping[str, Any]], classified_records: list[dict[str, Any]]
) -> None:
    """Verification checks required by task instructions."""
    expected_task_ids = [str(task.get("TASK", "")) for task in source_tasks]
    seen_task_ids = [str(record.get("task_id", "")) for record in classified_records]

    if len(classified_records) != len(source_tasks):
        raise ValueError(
            "Output row count mismatch: "
            f"expected {len(source_tasks)}, got {len(classified_records)}"
        )

    if len(classified_records) != EXPECTED_TASKS:
        raise ValueError(
            f"Expected {EXPECTED_TASKS} tasks, got {len(classified_records)}"
        )

    expected_counter = Counter(expected_task_ids)
    seen_counter = Counter(seen_task_ids)
    if expected_counter != seen_counter:
        missing = list((expected_counter - seen_counter).elements())
        extras = list((seen_counter - expected_counter).elements())
        raise ValueError(
            "task_id integrity check failed. "
            f"missing={len(missing)}, extra_or_duplicate={len(extras)}"
        )

    empty_domains = [
        record.get("task_id", "")
        for record in classified_records
        if not record.get("domains")
    ]
    if empty_domains:
        raise ValueError(f"Found tasks with empty domains: {len(empty_domains)}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / OUTPUT_REL_PATH

    print("Loading MCP-Atlas dataset...")
    dataset = load_dataset("ScaleAI/mcp-atlas", split="train")
    tasks = [item for item in dataset if isinstance(item, Mapping)]

    print(f"Classifying {len(tasks)} tasks...")
    records = [classify_task(task) for task in tasks]
    write_jsonl(output_path, records)

    print(f"Wrote {len(records)} rows to {OUTPUT_REL_PATH}")
    print("\nVerifying output...")

    loaded_records = load_jsonl(output_path)
    verify_integrity(tasks, loaded_records)

    print_summary(loaded_records)


if __name__ == "__main__":
    main()
