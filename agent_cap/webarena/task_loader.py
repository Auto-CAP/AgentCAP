"""Load WebArena tasks from config JSON files."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class WebArenaTask:
    task_id: int
    intent: str
    start_url: str
    sites: List[str]
    eval_config: Dict[str, Any]
    require_login: bool = False
    storage_state: str = ""
    raw: Dict[str, Any] = None


def load_webarena_tasks(
    config_dir: str = "config_files",
    limit: int = 0,
    task_ids: Optional[List[int]] = None,
) -> List[WebArenaTask]:
    config_path = Path(config_dir)

    if config_path.is_file() and config_path.suffix == ".json":
        with open(config_path) as f:
            raw_tasks = json.load(f)
        if isinstance(raw_tasks, dict):
            raw_tasks = [raw_tasks]
    elif config_path.is_dir():
        raw_tasks = []
        for f in sorted(config_path.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    raw_tasks.extend(data)
                else:
                    raw_tasks.append(data)
    else:
        raise FileNotFoundError(f"Config not found: {config_dir}")

    tasks = []
    for raw in raw_tasks:
        task_id = raw.get("task_id", len(tasks))
        if task_ids and task_id not in task_ids:
            continue

        tasks.append(
            WebArenaTask(
                task_id=task_id,
                intent=raw.get("intent", ""),
                start_url=raw.get("start_url", ""),
                sites=raw.get("sites", []),
                eval_config=raw.get("eval", {}),
                require_login=raw.get("require_login", False),
                storage_state=raw.get("storage_state", ""),
                raw=raw,
            )
        )

    if limit > 0:
        tasks = tasks[:limit]

    return tasks
