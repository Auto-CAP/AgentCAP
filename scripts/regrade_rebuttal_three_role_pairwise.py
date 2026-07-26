#!/usr/bin/env python3
"""Blind pairwise LLM audit of FinanceBench baseline vs PVE outputs."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import aiohttp

from agent_cap.runner.unified_runner import _load_dataset_tasks

JUDGE_PROMPT = """You are a strict correctness judge for FinanceBench financial question answering. Judge each candidate independently against the question and reference answer.

A candidate is correct only when its final conclusion and all requested numerical quantities are semantically consistent with the reference answer. Allow equivalent units, percentages, signs, and reasonable rounding. Do not treat a company name containing digits, a year, or an unrelated overlapping number as the answer. If the question asks for a yes/no conclusion plus supporting quantity, require both to be consistent. Different valid reasoning is allowed.

Return ONLY one JSON object with exactly these fields: {"A_correct": true or false, "B_correct": true or false, "reason": "one concise sentence"}."""

COMPARISONS = {
    "strong_verifier": (
        "mini planner, GPT-5.4 executor",
        "mini planner, GPT-5.4 verifier, GPT-5.4 executor",
    ),
    "mini_verifier_control": (
        "GPT-5.4 planner, mini executor",
        "GPT-5.4 planner, mini verifier, mini executor",
    ),
}


def load_env() -> None:
    path = Path("/root/.hermes/.env")
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def load_output(root: Path, team: str) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(one(root / team, "output-data_*.jsonl"))
    return {str(row["task_id"]): row for row in rows}


def parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first < 0 or last < first:
        raise ValueError("no JSON object")
    obj = json.loads(cleaned[first : last + 1])
    if set(obj) != {"A_correct", "B_correct", "reason"}:
        raise ValueError(f"unexpected keys: {sorted(obj)}")
    if not isinstance(obj["A_correct"], bool) or not isinstance(obj["B_correct"], bool):
        raise ValueError("correctness fields must be booleans")
    return obj


async def judge(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
    }
    last_error: Exception | None = None
    for _ in range(3):
        try:
            async with session.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
                raw = json.loads(body)
                content = raw["choices"][0]["message"]["content"]
                return parse_json(content), raw.get("usage") or {}
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(2)
    raise RuntimeError(f"judge failed after retries: {last_error}")


async def main_async(args: argparse.Namespace) -> None:
    load_env()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing {args.api_key_env}")

    baseline_a = load_output(args.baseline_mini_gpt54, "team_plan_execute")
    baseline_b = load_output(args.baseline_gpt54_mini, "team_plan_execute")
    pve_c = load_output(args.pve_mini_gpt54_gpt54, "team_plan_verify_execute")
    pve_d = load_output(args.pve_gpt54_mini_mini, "team_plan_verify_execute")
    outputs = {
        "strong_verifier": (baseline_a, pve_c),
        "mini_verifier_control": (baseline_b, pve_d),
    }

    tasks = {str(task.task_id): task for task in _load_dataset_tasks("financebench", 0)}
    common_ids = list(baseline_a)
    for name, (baseline, pve) in outputs.items():
        if set(baseline) != set(pve) or set(baseline) != set(common_ids):
            raise SystemExit(f"task coverage mismatch for {name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.exists():
        for row in load_jsonl(args.output):
            existing[(str(row["comparison"]), str(row["task_id"]))] = row

    with args.output.open("a", encoding="utf-8") as file:
        async with aiohttp.ClientSession() as session:
            for comparison, (baseline, pve) in outputs.items():
                for task_id in common_ids:
                    key = (comparison, task_id)
                    if key in existing:
                        continue
                    task = tasks[task_id]
                    eval_config = task.eval_config or {}
                    question = str(eval_config.get("question", ""))
                    reference = str(eval_config.get("answer", ""))
                    base_answer = str(baseline[task_id].get("output_text", ""))
                    pve_answer = str(pve[task_id].get("output_text", ""))
                    swapped = int(hashlib.sha256(f"{comparison}:{task_id}".encode()).hexdigest(), 16) % 2 == 1
                    answer_a, answer_b = (pve_answer, base_answer) if swapped else (base_answer, pve_answer)
                    prompt = (
                        f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\n"
                        f"CANDIDATE A:\n{answer_a}\n\nCANDIDATE B:\n{answer_b}"
                    )
                    verdict, usage = await judge(
                        session,
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        user_prompt=prompt,
                    )
                    base_correct = verdict["B_correct"] if swapped else verdict["A_correct"]
                    pve_correct = verdict["A_correct"] if swapped else verdict["B_correct"]
                    row = {
                        "comparison": comparison,
                        "task_id": task_id,
                        "baseline_label": COMPARISONS[comparison][0],
                        "pve_label": COMPARISONS[comparison][1],
                        "swapped": swapped,
                        "baseline_correct": base_correct,
                        "pve_correct": pve_correct,
                        "reason": verdict["reason"],
                        "judge_model": args.model,
                        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
                        "usage": usage,
                        "credentials_serialized": False,
                    }
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    file.flush()

    rows = load_jsonl(args.output)
    summary: dict[str, Any] = {
        "judge_model": args.model,
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
        "comparisons": {},
    }
    for comparison in COMPARISONS:
        selected = [row for row in rows if row["comparison"] == comparison]
        if len(selected) != len(common_ids):
            raise SystemExit(f"incomplete judge results for {comparison}: {len(selected)}/{len(common_ids)}")
        baseline_passed = sum(bool(row["baseline_correct"]) for row in selected)
        pve_passed = sum(bool(row["pve_correct"]) for row in selected)
        fixed = sum(not row["baseline_correct"] and row["pve_correct"] for row in selected)
        broken = sum(row["baseline_correct"] and not row["pve_correct"] for row in selected)
        summary["comparisons"][comparison] = {
            "n": len(selected),
            "baseline_passed": baseline_passed,
            "pve_passed": pve_passed,
            "delta_percent": 100 * (pve_passed - baseline_passed) / len(selected),
            "fixed": fixed,
            "broken": broken,
        }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-mini-gpt54", type=Path, required=True)
    parser.add_argument("--baseline-gpt54-mini", type=Path, required=True)
    parser.add_argument("--pve-mini-gpt54-gpt54", type=Path, required=True)
    parser.add_argument("--pve-gpt54-mini-mini", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8317/v1")
    parser.add_argument("--api-key-env", default="CLIPROXYAPI_API_KEY")
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
