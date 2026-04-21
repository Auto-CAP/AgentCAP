import argparse
import os
import re
import sqlite3
import sys
import time

from datasets import load_dataset
from math_verify import parse, verify
import aiohttp
import asyncio
import json


JUDGE_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemini-3.1-flash-lite-preview")
JUDGE_PROMPT = """You are grading an IMO mathematical answer.

GROUND TRUTH ANSWER: {gt}
SUBMITTED ANSWER: {pred}

Decide whether the submitted answer is mathematically equivalent to the ground truth.

CRITICAL RULES — NONE of the following are wrong answers:
- Different LaTeX formatting of the same value (e.g., `\\frac{{1}}{{2}}` vs `\\frac12`, `1/2`, `0.5`, `\\tfrac12`).
- Same multi-value answer written in different order or separators (comma, "or", `\\quad`, `\\qquad`, parentheses, "and", line breaks).
- Same formula with different variable names (e.g., `P(x)=x+1` vs `P(n)=n+1` vs `Q(x)=x+1`).
- Added labels or wrappers around the same value (e.g., `C_{{\\min}}=2^{{u-2}}` vs `2^{{u-2}}`; `T(m)=-4` vs `-4`).
- Extra restating / quantifiers that do not change the value (e.g., `P(x)=x+1 for all integers x` vs `P(x)=x+1`).
- Equivalent expressions (e.g., `\\lfloor \\log_2 a \\rfloor + 1` vs `\\lceil \\log_2(a+1) \\rceil` for positive integer `a`---treat as equivalent only if mathematically identical on the domain).
- Math identity: `2025^2 a(a-1)` vs `2025^2\\,a(a-1)`.

An answer IS wrong when:
- It gives a DIFFERENT numerical value, set, or formula from the ground truth.
- It is INCOMPLETE: ground truth has multiple distinct solutions and the submission gives only some (e.g., GT `{{-2/3, 0, 2/3}}`, pred `r=0` -> wrong, incomplete).
- It is EXTRANEOUS: submission includes values not in the ground truth.
- The expressions are not mathematically equivalent even after normalizing notation.

Return only `yes` (equivalent/correct) or `no` (different/incomplete/wrong). No other text."""


def extract_boxed(text):
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


async def judge_pair(session, gt, pred):
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    body = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "user", "content": JUDGE_PROMPT.format(gt=gt, pred=pred)}
        ],
        "max_tokens": 20,
        "temperature": 0,
    }
    for _ in range(3):
        try:
            async with session.post(JUDGE_URL, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as r:
                d = await r.json()
                text = (d.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
                if "yes" in text.lower():
                    return True
                if "no" in text.lower():
                    return False
        except Exception:
            await asyncio.sleep(2)
    return None


async def run(args):
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, task_id, exec_response FROM hybrid_runs").fetchall()

    ds = load_dataset("Hwilner/imo-answerbench", split="train")
    expected = {ex["Problem ID"]: (str(ex["Short Answer"]).strip(), ex.get("Category", "?")) for ex in ds}

    sem = asyncio.Semaphore(args.concurrency)
    results = {}
    async with aiohttp.ClientSession() as session:
        async def one(row):
            tid = row["task_id"]
            resp = row["exec_response"] or ""
            pred = extract_boxed(resp)
            if tid not in expected:
                return
            gt, cat = expected[tid]
            fast_match = False
            if pred:
                try:
                    fast_match = bool(verify(parse(gt), parse(pred)))
                except Exception:
                    fast_match = pred.strip() == gt.strip()
            if fast_match:
                results[tid] = (True, cat, "math_verify")
                return
            if not pred:
                results[tid] = (False, cat, "empty")
                return
            async with sem:
                j = await judge_pair(session, gt, pred)
            results[tid] = (bool(j), cat, "llm_judge" if j is not None else "judge_error")

        await asyncio.gather(*(one(r) for r in rows))

    from collections import defaultdict
    per_cat = defaultdict(lambda: [0, 0])
    tot_c, tot_n = 0, 0
    for tid, (ok, cat, why) in results.items():
        per_cat[cat][1] += 1
        tot_n += 1
        if ok:
            per_cat[cat][0] += 1
            tot_c += 1

    print(f"\n{args.db}")
    for cat in sorted(per_cat):
        c, n = per_cat[cat]
        pct = c / n * 100 if n else 0
        print(f"  {cat:20s}: {c}/{n} ({pct:.1f}%)")
    print(f"  {'TOTAL':20s}: {tot_c}/{tot_n} ({tot_c/max(1,tot_n)*100:.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
