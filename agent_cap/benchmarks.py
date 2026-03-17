"""Load public benchmarks (GSM8K, HumanEval, SWE-Bench Pro, etc.) as TaskDef lists."""

import random
import re
from typing import List

from agent_cap.runner.executor import TaskDef


def load_benchmark(name: str, num_tasks: int = 50, seed: int = 42) -> List[TaskDef]:
    """Load a public benchmark as a list of TaskDef.

    Args:
        name: "gsm8k", "humaneval", "gpqa", "mmlu_pro", or "bigcodebench"
        num_tasks: number of tasks to sample (0 = all)
        seed: random seed for reproducible sampling
    """
    loaders = {
        "gsm8k": _load_gsm8k,
        "humaneval": _load_humaneval,
        "gpqa": _load_gpqa,
        "mmlu_pro": _load_mmlu_pro,
        "bigcodebench": _load_bigcodebench,
        "swebench_pro": _load_swebench_pro,
    }
    if name not in loaders:
        raise ValueError(f"Unknown benchmark: {name}. Supported: {list(loaders)}")
    return loaders[name](num_tasks, seed)


def _sample(items: list, num: int, seed: int) -> list:
    if num <= 0 or num >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(list(items), num)


def _load_gsm8k(num_tasks: int, seed: int) -> List[TaskDef]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = _sample(list(ds), num_tasks, seed)

    tasks = []
    for i, ex in enumerate(samples):
        question = ex["question"]
        answer_text = ex["answer"]

        # Extract ground truth: number after "#### "
        m = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", answer_text)
        if not m:
            continue
        ground_truth = float(m.group(1).replace(",", ""))

        prompt = (
            f"{question}\n\n"
            "Solve this step by step. After your reasoning, write your final "
            "numerical answer on a new line in EXACTLY this format:\n"
            "#### <number>"
        )

        tasks.append(
            TaskDef(
                id=f"gsm8k-{i}",
                name=question[:60],
                messages=[{"role": "user", "content": prompt}],
                category="math",
                eval_config={
                    "type": "numerical",
                    "expected": ground_truth,
                    "tolerance": 0.01,
                },
            )
        )
    return tasks


def _load_humaneval(num_tasks: int, seed: int) -> List[TaskDef]:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    samples = _sample(list(ds), num_tasks, seed)

    tasks = []
    for ex in samples:
        task_id = ex["task_id"]
        func_prompt = ex["prompt"]
        test_code = ex["test"]
        entry_point = ex["entry_point"]

        prompt = (
            "Complete the following Python function. "
            "Write ONLY the complete function (including the def line), no explanation.\n\n"
            f"```python\n{func_prompt}```"
        )

        tasks.append(
            TaskDef(
                id=task_id,
                name=entry_point,
                messages=[{"role": "user", "content": prompt}],
                category="coding",
                eval_config={
                    "type": "humaneval",
                    "test_code": test_code,
                    "entry_point": entry_point,
                },
            )
        )
    return tasks


def _load_gpqa(num_tasks: int, seed: int) -> List[TaskDef]:
    from datasets import load_dataset

    ds = load_dataset("fingertap/GPQA-Diamond", split="test")
    samples = _sample(list(ds), num_tasks, seed)

    tasks = []
    for i, ex in enumerate(samples):
        question = ex["question"]
        answer = ex["answer"].strip().upper()

        prompt = (
            f"{question}\n\n"
            "Think step by step, then provide your final answer as a single letter "
            "(A, B, C, or D) on a new line in the format:\n"
            "The answer is X"
        )

        tasks.append(
            TaskDef(
                id=f"gpqa-{i}",
                name=question[:60],
                messages=[{"role": "user", "content": prompt}],
                category="science",
                eval_config={
                    "type": "multiple_choice",
                    "expected_answer": answer,
                },
            )
        )
    return tasks


def _load_mmlu_pro(num_tasks: int, seed: int) -> List[TaskDef]:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    samples = _sample(list(ds), num_tasks, seed)

    tasks = []
    letters = "ABCDEFGHIJ"
    for i, ex in enumerate(samples):
        question = ex["question"]
        options = ex["options"]
        answer = ex["answer"].strip().upper()
        category = ex.get("category", "general")

        options_text = "\n".join(
            f"{letters[j]}) {opt}" for j, opt in enumerate(options) if opt
        )

        prompt = (
            f"{question}\n\n{options_text}\n\n"
            "Think step by step, then provide your final answer as a single letter "
            "on a new line in the format:\n"
            "The answer is X"
        )

        tasks.append(
            TaskDef(
                id=f"mmlu_pro-{i}",
                name=f"[{category}] {question[:50]}",
                messages=[{"role": "user", "content": prompt}],
                category=category,
                eval_config={
                    "type": "multiple_choice",
                    "expected_answer": answer,
                },
            )
        )
    return tasks


def _load_bigcodebench(num_tasks: int, seed: int) -> List[TaskDef]:
    from datasets import load_dataset

    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
    samples = _sample(list(ds), num_tasks, seed)

    tasks = []
    for ex in samples:
        task_id = ex["task_id"]
        instruct = ex["instruct_prompt"]
        code_prompt = ex["code_prompt"]
        test_code = ex["test"]
        entry_point = ex["entry_point"]
        libs = ex["libs"]

        prompt = (
            f"Write a Python function called `{entry_point}` that does the following:\n\n"
            f"{instruct}\n\n"
            "The function signature and required imports are:\n"
            f"```python\n{code_prompt}\n```\n\n"
            "Write the COMPLETE function implementation including all imports. "
            "Put your code in a ```python code block."
        )

        packed_test = f"{code_prompt}|||TESTS|||{test_code}"

        tasks.append(
            TaskDef(
                id=f"bcb-{task_id.replace('/', '-')}",
                name=f"[{','.join(libs[:3])}] {instruct[:50]}",
                messages=[{"role": "user", "content": prompt}],
                category="coding",
                eval_config={
                    "type": "bigcodebench",
                    "test_code": packed_test,
                    "entry_point": entry_point,
                },
            )
        )
    return tasks


def _load_swebench_pro(
    num_tasks: int,
    seed: int,
    dataset_name: str = "ScaleAI/SWE-bench_Pro",
) -> List[TaskDef]:
    """Load SWE-Bench Pro (Verified) as a list of TaskDef.

    Each task presents a GitHub issue and asks the model to produce a
    unified-diff patch that resolves it.

    Args:
        num_tasks: Number of tasks to sample (0 = all).
        seed: Random seed for reproducible sampling.
        dataset_name: HuggingFace dataset identifier.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="test")
    samples = _sample(list(ds), num_tasks, seed)

    tasks: List[TaskDef] = []
    for i, ex in enumerate(samples):
        instance_id = ex.get("instance_id", f"swebench-{i}")
        repo = ex.get("repo", "unknown/repo")
        base_commit = ex.get("base_commit", "")
        problem_statement = ex.get("problem_statement", "")
        requirements = ex.get("requirements", "")
        interface = ex.get("interface", "")
        repo_language = ex.get("repo_language", "")
        gold_patch = ex.get("patch", "")
        test_patch = ex.get("test_patch", "")
        fail_to_pass = ex.get("fail_to_pass", "")

        extra_sections = ""
        if requirements and str(requirements).strip():
            extra_sections += f"\n\n## Requirements\n{str(requirements).strip()}\n"
        if interface and str(interface).strip():
            extra_sections += f"\n\n## Interface\n{str(interface).strip()}\n"

        lang_note = ""
        if repo_language:
            lang_note = f" (language: {repo_language})"

        prompt = (
            f"You are a software engineer working on the repository "
            f"**{repo}**{lang_note} "
            f"(base commit: `{base_commit}`).\n\n"
            f"## Issue\n{problem_statement.strip()}\n"
            f"{extra_sections}\n"
            "## Task\n"
            "Analyze the issue above and produce a **unified diff patch** "
            "(the output of `git diff`) that fixes it.\n\n"
            "Requirements:\n"
            "- Output ONLY the patch in a ```diff code block.\n"
            "- The patch must apply cleanly to the base commit.\n"
            "- Do not include unrelated changes.\n"
        )

        tasks.append(
            TaskDef(
                id=f"swebench_pro-{instance_id}",
                name=f"[{repo}] {problem_statement[:60]}",
                messages=[{"role": "user", "content": prompt}],
                category="software_engineering",
                eval_config={
                    "type": "swebench",
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": base_commit,
                    "expected_patch": gold_patch,
                    "test_patch": test_patch,
                    "fail_to_pass": fail_to_pass,
                    "repo_language": repo_language,
                    "dockerhub_tag": ex.get("dockerhub_tag", ""),
                },
            )
        )
    return tasks
