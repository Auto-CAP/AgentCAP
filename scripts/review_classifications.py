#!/usr/bin/env python3
"""Review task classifications: ask a model if it agrees with Qwen3-32B's labels.

Usage:
    python scripts/review_classifications.py \
        --api-key sk-xxx \
        --model gpt-5.4 \
        --base-url https://api.openai.com/v1 \
        --output results/task_classifications_gpt54_review.jsonl
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import aiohttp
from datasets import load_dataset


REVIEW_PROMPT = """A task classifier labeled this agentic AI benchmark task as "{qwen_label}".

Definition:
- plan-heavy: The agent needs to think hard before it can act. Complex reasoning, problem decomposition, multi-step logic, mathematical derivation, figuring out a non-obvious strategy. Even if execution is trivial, the planning is hard.
- execute-heavy: The agent knows what to do (or can figure it out quickly), but carrying it out is the hard part. Substantial tool interactions, processing many records, navigating complex APIs, long multi-turn tool-call chains.

Task: {prompt}

Do you agree with the "{qwen_label}" classification?
Respond in JSON only: {{"agree": true, "reasoning": "one sentence"}} or {{"agree": false, "classification": "plan-heavy|execute-heavy", "reasoning": "one sentence"}}"""


async def review_one(session, base_url, api_key, model, prompt, qwen_label, retries=3):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": REVIEW_PROMPT.format(prompt=prompt, qwen_label=qwen_label)}],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    for attempt in range(retries):
        try:
            async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                if resp.status != 200:
                    await asyncio.sleep(1)
                    continue
                result = await resp.json()
                content = result["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                return json.loads(content)
        except (json.JSONDecodeError, KeyError, aiohttp.ClientError) as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            await asyncio.sleep(1)
    return {"agree": True, "reasoning": "fallback after retries"}


async def run_review(tasks, qwen_labels, base_url, api_key, model, output_path, concurrency=5):
    sem = asyncio.Semaphore(concurrency)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing progress
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                d = json.loads(line)
                done.add(d["index"])
        print(f"Resuming: {len(done)} already done")

    agree_count = 0
    total = 0

    async with aiohttp.ClientSession() as session:
        with open(output_path, "a") as f:
            for i in range(len(tasks)):
                if i in done:
                    continue
                async with sem:
                    prompt = tasks[i]["PROMPT"]
                    ql = qwen_labels[i]["classification"]
                    result = await review_one(session, base_url, api_key, model, prompt, ql)

                    agrees = result.get("agree", True)
                    if agrees:
                        final_label = ql
                    else:
                        final_label = result.get("classification", ql)

                    agree_count += int(agrees)
                    total += 1

                    line = {
                        "index": i,
                        "task_id": qwen_labels[i]["task_id"],
                        "qwen_classification": ql,
                        "reviewer_agrees": agrees,
                        "reviewer_classification": final_label,
                        "reasoning": result.get("reasoning", ""),
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    f.flush()

                    status = "agree" if agrees else f"DISAGREE -> {final_label}"
                    print(f"[{i+1}/500] {status:20s} | {prompt[:70]}...")

    return agree_count, total


def main():
    parser = argparse.ArgumentParser(description="Review task classifications with a second model.")
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--qwen-labels", type=str, default="results/task_classifications.jsonl")
    parser.add_argument("--output", type=str, default="results/task_classifications_gpt54_review.jsonl")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    print("Loading dataset and Qwen labels...")
    ds = load_dataset("ScaleAI/mcp-atlas", split="train")
    with open(args.qwen_labels) as f:
        qwen_labels = [json.loads(line) for line in f]

    print(f"Reviewing 500 tasks with {args.model}...")
    agree, total = asyncio.run(run_review(list(ds), qwen_labels, args.base_url, args.api_key, args.model, args.output, args.concurrency))

    print(f"\n=== Review Results ===")
    print(f"  Agree: {agree}/{total} ({agree/total*100:.1f}%)")
    print(f"  Disagree: {total-agree}/{total} ({(total-agree)/total*100:.1f}%)")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
