"""Build Harbor job configs for AgentCAP runs.

A job runs CAPAgent with one team (roles -> endpoints) over a local Harbor
dataset written by ``agent_cap.harbor.adapters``. Benchmark presets carry the
per-benchmark agent settings AgentCAP uses today (tool backend, system prompt,
step limit), so a TEASBench agent YAML maps onto the same job.

Secrets stay out of job files: role fields such as ``api_key: ${OPENROUTER_API_KEY}``
are written as-is and expanded by CAPAgent at run time; judge keys reach the
verifier through ``[verifier.env]`` templates in task.toml, resolved by Harbor
from the environment of ``harbor run``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

AGENT_IMPORT_PATH = "agent_cap.harbor.agent:CAPAgent"
DSH_IMPORT_PATH = "agent_cap.harbor.dsh:DSHAgent"


def _imo_preset() -> Dict[str, Any]:
    from agent_cap.harbor.adapters.imo_answerbench import ANSWER_FILE, SYSTEM_PROMPT

    return {
        "strategy": "single",
        "tool_backend": "math-python",
        "max_turns": 128,
        "system_prompt": SYSTEM_PROMPT,
        "answer_file": ANSWER_FILE,
        "workdir": "/workspace",
    }


def _nl2repo_preset() -> Dict[str, Any]:
    return {
        "strategy": "single",
        "tool_backend": "shell",
        "max_turns": 250,
        "workdir": "/workspace",
        "tool_options": {"shell_timeout": 450},
    }


PRESETS = {
    "imo-answerbench": _imo_preset,
    "nl2repo": _nl2repo_preset,
}

# TEASBench agent YAML keys -> CAPAgent kwargs.
_TEAS_DIRECT_KEYS = ("strategy", "max_turns", "tool_backend", "sequence")
# judge block -> verifier env vars read by tests/grade_imo.py
_JUDGE_ENV = {"base_url": "IMO_JUDGE_BASE_URL", "api_key": "IMO_JUDGE_API_KEY", "name": "IMO_JUDGE_MODEL"}


# DeepSeek Harness runs its own tools inside the container; per-benchmark settings
# only say where the task lives and where an answer must be written.
DSH_PRESETS = {
    "imo-answerbench": {"workdir": "/workspace", "answer_file": "/workspace/answer.txt"},
    "nl2repo": {"workdir": "/workspace"},
}


def dsh_kwargs(benchmark: str, settings: Mapping[str, Any]) -> Dict[str, Any]:
    """DSHAgent kwargs: the benchmark preset plus the given (non-None) settings."""
    if benchmark not in DSH_PRESETS:
        raise ValueError(f"unknown benchmark {benchmark!r}; known: {sorted(DSH_PRESETS)}")
    kwargs = dict(DSH_PRESETS[benchmark])
    kwargs.update({k: v for k, v in settings.items() if v is not None})
    for required in ("model", "endpoint_url"):
        if not kwargs.get(required):
            raise ValueError(f"DeepSeek Harness needs `{required}`")
    return kwargs


def preset(benchmark: str) -> Dict[str, Any]:
    try:
        return PRESETS[benchmark]()
    except KeyError as exc:
        raise ValueError(f"unknown benchmark {benchmark!r}; known: {sorted(PRESETS)}") from exc


def parse_role(spec: str) -> Tuple[str, Dict[str, Any]]:
    """``ROLE=k=v,k=v`` (as in ``python -m agent_cap.agents --agent``), env vars NOT expanded."""
    role, sep, rest = spec.partition("=")
    if not sep or not role.strip():
        raise ValueError(f"role spec must be ROLE=key=value,...; got {spec!r}")
    fields: Dict[str, Any] = {}
    for chunk in rest.split(","):
        chunk = chunk.strip()
        if chunk:
            key, _, value = chunk.partition("=")
            fields[key.strip()] = value.strip()
    return role.strip(), fields


def load_teas_agent_yaml(path: Path) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Read a TEASBench agent YAML (pipeline/configs/agents/*.yaml).

    Returns ``(capagent_kwargs, verifier_env)``. ``verifier_env`` maps the
    judge block onto the variables the IMO grader reads (values unexpanded).
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    kwargs: Dict[str, Any] = {k: data[k] for k in _TEAS_DIRECT_KEYS if k in data}
    agents = data.get("agents") or {}
    if not isinstance(agents, Mapping) or not agents:
        raise ValueError(f"{path}: no `agents` block")
    kwargs["roles"] = {str(role): dict(spec or {}) for role, spec in agents.items()}
    judge = data.get("judge") or {}
    verifier_env = {
        env: str(judge[key]) for key, env in _JUDGE_ENV.items() if isinstance(judge, Mapping) and judge.get(key)
    }
    return kwargs, verifier_env


def capagent_kwargs(
    benchmark: str,
    roles: Mapping[str, Mapping[str, Any]],
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    kwargs = preset(benchmark)
    kwargs["roles"] = {str(r): dict(v) for r, v in roles.items()}
    for key, value in (overrides or {}).items():
        if value is not None:
            kwargs[key] = copy.deepcopy(value)
    if not kwargs.get("roles"):
        raise ValueError("at least one role is required")
    return kwargs


def build_job_config(
    *,
    dataset_path: Path,
    agent_kwargs: Mapping[str, Any],
    job_name: str,
    jobs_dir: Path = Path("jobs"),
    environment: str = "docker",
    environment_import_path: Optional[str] = None,
    n_concurrent: int = 1,
    n_attempts: int = 1,
    task_names: Optional[Sequence[str]] = None,
    n_tasks: Optional[int] = None,
    agent_env: Optional[Mapping[str, str]] = None,
    agent_import_path: str = AGENT_IMPORT_PATH,
) -> Dict[str, Any]:
    """A Harbor JobConfig (as a plain dict, ready for YAML)."""
    dataset: Dict[str, Any] = {"path": str(Path(dataset_path))}
    if task_names:
        dataset["task_names"] = list(task_names)
    if n_tasks:
        dataset["n_tasks"] = int(n_tasks)
    env_cfg: Dict[str, Any] = {"type": environment}
    if environment_import_path:
        env_cfg = {"import_path": environment_import_path}
    agent: Dict[str, Any] = {"import_path": agent_import_path, "kwargs": dict(agent_kwargs)}
    if agent_env:
        agent["env"] = dict(agent_env)
    return {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir),
        "n_attempts": int(n_attempts),
        "n_concurrent_trials": int(n_concurrent),
        "skip_credential_checks": True,
        "environment": env_cfg,
        "datasets": [dataset],
        "agents": [agent],
    }


def validate_job_config(config: Mapping[str, Any]) -> None:
    """Raise if Harbor would reject the config (requires the harbor package)."""
    from harbor.models.job.config import JobConfig

    JobConfig.model_validate(dict(config))


def write_job_config(config: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def roles_from_specs(specs: List[str]) -> Dict[str, Dict[str, Any]]:
    return dict(parse_role(s) for s in specs)


__all__ = [
    "AGENT_IMPORT_PATH",
    "DSH_IMPORT_PATH",
    "DSH_PRESETS",
    "dsh_kwargs",
    "PRESETS",
    "preset",
    "parse_role",
    "load_teas_agent_yaml",
    "capagent_kwargs",
    "build_job_config",
    "validate_job_config",
    "write_job_config",
    "roles_from_specs",
]
