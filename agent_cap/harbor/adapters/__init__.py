"""Harbor adapters: convert benchmarks into Harbor task directories.

Each adapter writes one directory per task under ``out_dir``::

    <task>/
      task.toml        Harbor task config (timeouts, resources, verifier mode)
      instruction.md   what the agent is told
      environment/     the agent's container (Dockerfile + build context)
      tests/           the verifier (test.sh writes /logs/verifier/reward.json)
      solution/        oracle, where the benchmark makes one possible

The resulting directory is a local Harbor dataset: ``datasets: [{path: <out_dir>}]``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

TEMPLATES = Path(__file__).with_name("templates")


def toml_str(value: Any) -> str:
    """A TOML basic string (JSON string escapes are valid TOML escapes)."""
    return json.dumps(str(value), ensure_ascii=False)


def render(template: Path, values: Mapping[str, Any]) -> str:
    text = template.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace(f"@@{key}@@", str(value))
    leftover = re.findall(r"@@[A-Z_]+@@", text)
    if leftover:
        raise ValueError(f"unfilled placeholders in {template}: {sorted(set(leftover))}")
    return text


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-").lower()
    return slug or "task"


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


__all__ = ["TEMPLATES", "toml_str", "render", "slugify", "reset_dir", "write_executable"]
