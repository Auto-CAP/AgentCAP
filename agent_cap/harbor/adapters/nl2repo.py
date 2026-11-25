"""NL2RepoBench -> Harbor tasks.

Source: a checkout of upstream https://github.com/multimodal-art-projection/NL2RepoBench
(validated at commit 781a1da). For each ``test_files/<task>/`` it reads
``start.md``, ``test_commands.json``, ``test_files.json`` and the test-case count.

* The agent's container (environment/Dockerfile) is Python 3.12 with only
  ``start.md`` in ``/workspace``; instruction.md is upstream's OpenHands prompt.
* The verifier runs separately, in an image built from the official evaluation
  image ``ghcr.io/multimodal-art-projection/nl2repobench/<task>:1.0``;
  tests/nl2repo_eval.py reproduces upstream's post_processor scoring.
* No oracle: upstream ships no reference repositories.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from agent_cap.harbor.adapters import TEMPLATES, render, reset_dir, slugify, toml_str, write_executable

UPSTREAM_REPO = "https://github.com/multimodal-art-projection/NL2RepoBench"
VALIDATED_COMMIT = "781a1da"
EVAL_IMAGE = "ghcr.io/multimodal-art-projection/nl2repobench/{task}:1.0"
# openhands/openhands_app.py in upstream NL2RepoBench.
INSTRUCTION = (
    "According to the start.md in the workspace, implement the entire project as per the "
    "requirements specified in the document, ensuring that the final product can be directly "
    "run in the current directory. The running requirements should comply with the <API Usage "
    "Guide> section of the document. Please complete this task step by step."
)


def upstream_commit(source: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_task(task_dir: Path) -> Dict[str, object]:
    """Read one upstream task folder the way upstream's test_data_service does."""
    files = sorted(p.name for p in task_dir.iterdir() if p.is_file())
    txt = [f for f in files if f.endswith(".txt")]
    commands = [f for f in files if f.endswith(".json") and "commands" in f]
    test_files = [f for f in files if f.endswith(".json") and "files" in f]
    md = [f for f in files if f.endswith(".md")]
    if not (txt and commands and test_files and md):
        raise ValueError(f"{task_dir} is missing one of: count .txt, commands .json, files .json, .md")
    return {
        "task": task_dir.name,
        "test_case_count": int((task_dir / txt[0]).read_text(encoding="utf-8").strip()),
        "test_commands": json.loads((task_dir / commands[0]).read_text(encoding="utf-8")),
        "test_files": json.loads((task_dir / test_files[0]).read_text(encoding="utf-8")),
        "start_md": task_dir / md[0],
    }


def list_tasks(source: Path) -> List[str]:
    root = Path(source) / "test_files"
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def write_task(
    spec: Dict[str, object],
    out_dir: Path,
    *,
    commit: str,
    agent_timeout_sec: int = 7200,
    verifier_timeout_sec: int = 3600,
) -> Path:
    task = str(spec["task"])
    if task != task.lower():
        raise ValueError(f"task name {task!r} is not a valid image name (must be lowercase)")
    slug = slugify(task)
    task_dir = reset_dir(Path(out_dir) / slug)
    tmpl = TEMPLATES / "nl2repo"
    eval_image = EVAL_IMAGE.format(task=task)

    (task_dir / "instruction.md").write_text(INSTRUCTION, encoding="utf-8")
    (task_dir / "task.toml").write_text(
        render(
            tmpl / "task.toml",
            {
                "TASK_NAME": toml_str(f"nl2repo/{slug}"),
                "DESCRIPTION": toml_str(f"NL2Repo: build the {task} repository from start.md"),
                "TASK": toml_str(task),
                "UPSTREAM_COMMIT": toml_str(commit),
                "EVAL_IMAGE_JSON": toml_str(eval_image),
                "TEST_CASE_COUNT": int(spec["test_case_count"]),
                "AGENT_TIMEOUT_SEC": int(agent_timeout_sec),
                "VERIFIER_TIMEOUT_SEC": int(verifier_timeout_sec),
            },
        ),
        encoding="utf-8",
    )

    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(render(tmpl / "environment" / "Dockerfile", {"TASK": task}), encoding="utf-8")
    (env_dir / "start.md").write_bytes(Path(spec["start_md"]).read_bytes())

    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "Dockerfile").write_text(
        render(tmpl / "tests" / "Dockerfile", {"TASK": task, "EVAL_IMAGE": eval_image}),
        encoding="utf-8",
    )
    write_executable(tests_dir / "test.sh", (tmpl / "tests" / "test.sh").read_text())
    (tests_dir / "nl2repo_eval.py").write_text((tmpl / "tests" / "nl2repo_eval.py").read_text(), encoding="utf-8")
    (tests_dir / "meta.json").write_text(
        json.dumps(
            {
                "task": task,
                "upstream_commit": commit,
                "eval_image": eval_image,
                "test_case_count": spec["test_case_count"],
                "test_commands": spec["test_commands"],
                "test_files": spec["test_files"],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return task_dir


def generate(
    source: Path,
    out_dir: Path,
    *,
    tasks: Optional[Sequence[str]] = None,
    agent_timeout_sec: int = 7200,
    verifier_timeout_sec: int = 3600,
    require_commit: Optional[str] = None,
) -> List[Path]:
    source, out_dir = Path(source), Path(out_dir)
    commit = upstream_commit(source)
    if require_commit and not commit.startswith(require_commit):
        raise ValueError(f"{source} is at {commit}, expected {require_commit}")
    names = list(tasks) if tasks else list_tasks(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = []
    for name in names:
        spec = read_task(source / "test_files" / name)
        dirs.append(
            write_task(
                spec,
                out_dir,
                commit=commit,
                agent_timeout_sec=agent_timeout_sec,
                verifier_timeout_sec=verifier_timeout_sec,
            )
        )
    (out_dir / "agentcard_dataset.json").write_text(
        json.dumps(
            {
                "benchmark": "nl2repo",
                "upstream": UPSTREAM_REPO,
                "upstream_commit": commit,
                "n_tasks": len(dirs),
                "tasks": [d.name for d in dirs],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return dirs


__all__ = ["generate", "read_task", "list_tasks", "write_task", "INSTRUCTION", "VALIDATED_COMMIT"]
