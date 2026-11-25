"""IMO-AnswerBench -> Harbor tasks.

Prompts, task selection and grading match ``python -m agent_cap.agents
--dataset imo-answerbench --evaluator imo`` (the TEASBench agentic setting):

* instruction.md is AgentCAP's user prompt for the problem, unchanged;
* the dataset-level system prompt is stored in task.toml metadata and passed
  to CAPAgent through its kwargs (Harbor gives agents only instruction.md);
* ``--limit N`` keeps the first N problems in dataset order, as AgentCAP does;
* tests/grade_imo.py applies AgentCAP's imo evaluator (boxed answer ->
  math-verify -> LLM judge) to ``/workspace/answer.txt``.

Agents other than CAPAgent do not know to write ``/workspace/answer.txt``;
pass ``answer_file_instruction=True`` (CLI: ``--answer-file-instruction``) to
append that request to instruction.md. It changes the prompt, so leave it off
for TEAS parity runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agent_cap.harbor.adapters import TEMPLATES, render, reset_dir, slugify, toml_str, write_executable

HF_DATASET = "Hwilner/imo-answerbench"
SYSTEM_PROMPT = (
    "You are an expert mathematical problem solver with "
    "access to a Python execution tool. Use it for calculations "
    "when useful. Put your final answer in \\boxed{}."
)
USER_SUFFIX = "\n\n" + r"Solve this step by step. Place your final numerical answer inside \boxed{}"
ANSWER_FILE = "/workspace/answer.txt"
ANSWER_FILE_INSTRUCTION = (
    "\n\nWhen you are done, write your final answer, including the \\boxed{...} "
    f"expression, to `{ANSWER_FILE}`."
)


def load_rows(source: Optional[str] = None) -> List[Dict[str, Any]]:
    """Rows of the dataset: a local .jsonl/.json file, or the Hugging Face dataset."""
    if source and Path(source).exists():
        path = Path(source)
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data if isinstance(data, list) else data.get("rows", []))
    from datasets import load_dataset

    return [dict(ex) for ex in load_dataset(source or HF_DATASET, split="train")]


def select_problems(
    rows: Iterable[Dict[str, Any]],
    *,
    limit: int = 0,
    indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Same filtering as agent_cap.runner.unified_runner for imo-answerbench."""
    problems: List[Dict[str, Any]] = []
    for i, ex in enumerate(rows):
        if not isinstance(ex, dict):
            continue
        problem = str(ex.get("Problem", "")).strip()
        if not problem:
            continue
        problems.append(
            {
                "index": i,
                "problem_id": str(ex.get("Problem ID", f"imo-{i}")),
                "problem": problem,
                "answer": str(ex.get("Short Answer", "")).strip(),
                "category": str(ex.get("Category", "math")),
            }
        )
    if indices:
        wanted = set(int(i) for i in indices)
        problems = [p for p in problems if p["index"] in wanted]
    elif limit > 0:
        problems = problems[:limit]
    return problems


def write_task(
    problem: Dict[str, Any],
    out_dir: Path,
    *,
    agent_timeout_sec: int = 7200,
    answer_file_instruction: bool = False,
) -> Path:
    slug = slugify(problem["problem_id"])
    task_dir = reset_dir(Path(out_dir) / slug)
    tmpl = TEMPLATES / "imo_answerbench"

    instruction = f"{problem['problem']}{USER_SUFFIX}"
    if answer_file_instruction:
        instruction += ANSWER_FILE_INSTRUCTION
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")

    (task_dir / "task.toml").write_text(
        render(
            tmpl / "task.toml",
            {
                "TASK_NAME": toml_str(f"imo-answerbench/{slug}"),
                "DESCRIPTION": toml_str(f"IMO-AnswerBench problem {problem['problem_id']}"),
                "PROBLEM_ID": toml_str(problem["problem_id"]),
                "CATEGORY": toml_str(problem["category"]),
                "DATASET_INDEX": int(problem["index"]),
                "SYSTEM_PROMPT": toml_str(SYSTEM_PROMPT),
                "AGENT_TIMEOUT_SEC": int(agent_timeout_sec),
            },
        ),
        encoding="utf-8",
    )

    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text((tmpl / "environment" / "Dockerfile").read_text(), encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    write_executable(tests_dir / "test.sh", (tmpl / "tests" / "test.sh").read_text())
    (tests_dir / "grade_imo.py").write_text((tmpl / "tests" / "grade_imo.py").read_text(), encoding="utf-8")
    (tests_dir / "expected.json").write_text(
        json.dumps({"problem_id": problem["problem_id"], "answer": problem["answer"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    # Oracle: the gold answer in boxed form. Written through Python so any
    # characters in the answer survive the shell.
    oracle = json.dumps("\\boxed{" + problem["answer"] + "}")
    write_executable(
        task_dir / "solution" / "solve.sh",
        "#!/bin/bash\nset -euo pipefail\n"
        f"python3 -c 'import json,sys; open(sys.argv[1],\"w\").write(json.loads(sys.argv[2]))' "
        f"{ANSWER_FILE} {_sh_quote(oracle)}\n",
    )
    return task_dir


def _sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def generate(
    out_dir: Path,
    *,
    source: Optional[str] = None,
    limit: int = 0,
    indices: Optional[Sequence[int]] = None,
    agent_timeout_sec: int = 7200,
    answer_file_instruction: bool = False,
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = select_problems(load_rows(source), limit=limit, indices=indices)
    dirs = [
        write_task(
            p,
            out_dir,
            agent_timeout_sec=agent_timeout_sec,
            answer_file_instruction=answer_file_instruction,
        )
        for p in problems
    ]
    (out_dir / "agentcard_dataset.json").write_text(
        json.dumps(
            {
                "benchmark": "imo-answerbench",
                "source": source or HF_DATASET,
                "n_tasks": len(dirs),
                "limit": limit,
                "indices": list(indices) if indices else None,
                "answer_file_instruction": answer_file_instruction,
                "tasks": [d.name for d in dirs],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return dirs


__all__ = ["generate", "select_problems", "load_rows", "write_task", "SYSTEM_PROMPT", "ANSWER_FILE"]
