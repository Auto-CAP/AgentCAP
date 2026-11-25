"""DeepSeek Harness (DSH) as a Harbor agent, alone or as an Agent Team.

Harbor loads ``agent_cap.harbor.dsh:DSHAgent`` (or ``agent_cap.harbor:DSHAgent``).
The model comes from Harbor's standard connection, as for Codex or Claude Code::

    harbor run --agent agent_cap.harbor:DSHAgent --model deepseek/deepseek-chat \
        --ak teammates=2 -p datasets/nl2repo

(``--base-url``, ``--api-key-name``, ``--api-format`` and ``--effort`` work as for
any Harbor agent), or from the older ``model``/``endpoint_url``/``api_key_env``
kwargs. Unlike CAPAgent, DSH is a full
coding harness that runs *inside* the task container: setup installs Node and
``@deepseek-ai/dsh`` there, run writes a DSH profile and calls
``dsh --profile agentcard --json <prompt>`` in the task's working directory. The
model endpoint must therefore be reachable from the container.

Modes:

* ``teammates=0``  - one DSH agent (bundles dsh-base + dsh-headless).
* ``teammates=K``  - DSH Agent Teams: a lead plus K teammates that share the
  workspace (adds dsh-experimental-agent-team-profile). In DSH 0.2.0-rc.2 the
  teammates use the lead's model; ``spawn_teammate`` has no model argument.

DSH persists one ``session.v4.jsonl`` per agent under ``/logs/agent/dsh-home``,
which Harbor syncs to the trial's ``agent/`` directory. After the run they are
parsed into the same per-task row CAPAgent writes (``capagent_result.json``:
per-role usage, per-request ``turn_stats`` with TTFT and decode time from DSH's
stream timestamps), so ``agent_cap.harbor.export_teas`` works unchanged.

The API key is never written to files: the profile names an environment
variable, and its value is read from this process's environment (Harbor's
``--api-key-name``, or ``api_key_env``) and passed to ``dsh`` through
``environment.exec(env=...)``.
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.model_api_format import ModelApiFormat

from agent_cap.harbor import RESULT_FILENAME
from agent_cap.harbor.connection import connection_spec, resolve_endpoint

DSH_VERSION = "0.2.0-rc.2"
PROFILE = "agentcard"
PROTOCOLS = ("openai-completions", "openai-responses", "anthropic-messages", "deepseek-messages")
KEY_ENV_IN_CONTAINER = "AGENTCARD_DSH_API_KEY"
# Harbor --api-format -> DSH route api
FORMAT_TO_PROTOCOL = {
    ModelApiFormat.OPENAI_CHAT_COMPLETIONS: "openai-completions",
    ModelApiFormat.OPENAI_RESPONSES: "openai-responses",
    ModelApiFormat.ANTHROPIC_MESSAGES: "anthropic-messages",
}
BASE_BUNDLES = ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-headless"]
TEAM_BUNDLE = "@deepseek-ai/dsh-experimental-agent-team-profile"
# Plugins a batch run does not need (session titles, the DeepSeek account route,
# web access, workflow delegation, package inventory).
DISABLED_PLUGINS = (
    "session-title-llm",
    "llm-deepseek-account",
    "tool-web",
    "web-search-deepseek",
    "tool-workflow",
    "plugin-package-inventory-deepseek",
)
TEAM_PROMPT = (
    "\n\nWork as an Agent Team. Create exactly {teammates} teammates with spawn_teammate, "
    "giving each a complete, self-contained task description. Split the work between "
    "them, track it on the shared task board (team_task_create, team_task_update), and "
    "give each teammate clear file ownership: all of you share {workdir}. Use "
    "send_message to coordinate and to review each other's work. Before you finish, "
    "call wait_agent until every teammate has finished, check list_agents and "
    "team_task_list, and integrate the results. All teammates stop when you finish."
)


def build_profile(
    *,
    model: str,
    base_url: str,
    protocol: str,
    teammates: int,
    sessions_root: str,
    context_window: int = 131072,
    max_tokens: int = 16384,
    reasoning_effort: Optional[str] = None,
    compat: Optional[Mapping[str, Any]] = None,
    reasoning_efforts: Any = None,
) -> Dict[str, Any]:
    """The DSH profile as ``{"package.json": ..., "cordis.patch.yml": ...}`` (JSON is valid YAML)."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    if max_tokens > context_window:
        raise ValueError("max_tokens must not exceed context_window")
    native = protocol == "deepseek-messages"
    provider = "deepseek-official" if native else "agentcard-endpoint"
    selection: Dict[str, Any] = {"provider": provider, "model": model}
    if reasoning_effort:
        selection["reasoningEffort"] = reasoning_effort
    patches: List[Dict[str, Any]] = [
        {"id": "agent-default-model", "config": selection},
        {"id": "session-persistence-jsonl", "config": {"root": sessions_root, "compression": "none"}},
        {"id": "session-log-deepseek", "config": {"enabled": False}},
    ]
    if teammates > 0:
        patches += [
            {"id": "agent-team", "config": {"maxMembers": int(teammates)}},
            # Teammates run as subagent continuations and share this pool.
            {"id": "subagent", "config": {"maxActiveSubagents": int(teammates)}},
        ]
    patches += [{"id": plugin, "disabled": True} for plugin in DISABLED_PLUGINS]
    catalog_model: Dict[str, Any] = {"id": model, "contextWindow": int(context_window), "maxTokens": int(max_tokens)}
    if native:
        patches.append({"id": "llm-deepseek", "config": {
            "baseURL": base_url, "apiKeyEnv": KEY_ENV_IN_CONTAINER, "models": [catalog_model],
        }})
    else:
        if reasoning_efforts is not None:
            catalog_model["reasoningEfforts"] = reasoning_efforts
        route: Dict[str, Any] = {
            "api": protocol, "baseURL": base_url, "apiKeyEnv": KEY_ENV_IN_CONTAINER, "models": [catalog_model],
        }
        if compat:
            route["compat"] = dict(compat)
        patches += [
            {"id": "llm-deepseek", "disabled": True},
            {"id": "llm-pi-ai", "config": {"providers": {provider: route}}},
        ]
    bundles = BASE_BUNDLES + ([TEAM_BUNDLE] if teammates > 0 else [])
    package = {"name": f"{PROFILE}-profile", "private": True, "dsh": {"profile": {"bundles": bundles}}}
    return {"package.json": package, "cordis.patch.yml": patches}


# --------------------------------------------------------------------------- #
# Session logs -> AgentCAP row
# --------------------------------------------------------------------------- #

def _usage_value(usage: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def read_sessions(sessions_root: Path) -> List[Dict[str, Any]]:
    """Parse every session.v4.jsonl under ``sessions_root`` into per-agent records."""
    sessions = []
    for path in sorted(Path(sessions_root).rglob("session.v4.jsonl")):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines or lines[0].get("type") != "session":
            continue
        header = lines[0]
        record: Dict[str, Any] = {
            "id": str(header.get("id", "")),
            "parent": header.get("parentSession"),
            "members": {},
            "requests": [],
            "tool_calls": 0,
            "finish_reason": None,
        }
        pending_start: Optional[float] = None
        for event in lines[1:]:
            kind, data, t = event.get("type"), event.get("data") or {}, event.get("time")
            if kind == "request/header":
                pending_start = float(t) if isinstance(t, (int, float)) else None
            elif kind == "assistant/message":
                usage = data.get("usage") or {}
                stream = [c.get("time") for c in (data.get("stream") or []) if isinstance(c.get("time"), (int, float))]
                end = float(t) if isinstance(t, (int, float)) else None
                start = pending_start if pending_start is not None else (stream[0] if stream else end)
                first = stream[0] if stream else end
                last = stream[-1] if stream else end
                message = data.get("message") or {}
                blocks = ((message.get("source") or {}).get("replayState") or {}).get("blocks") or []
                content = message.get("content")
                if isinstance(content, list):  # fall back to content parts if present
                    blocks = blocks or content
                n_tools = sum(1 for b in blocks if isinstance(b, dict) and b.get("type") in ("tool-call", "toolCall"))
                record["requests"].append({
                    "start_ms": start,
                    "input_tokens": _usage_value(usage, "inputTokens", "input_tokens"),
                    "output_tokens": _usage_value(usage, "outputTokens", "output_tokens"),
                    "reasoning_tokens": _usage_value(usage, "reasoningTokens", "reasoning_tokens"),
                    "cached_tokens": _usage_value(usage, "cacheReadTokens", "cacheRead", "cachedTokens", "cached_tokens"),
                    "ttft_s": max(0.0, (first - start) / 1000.0) if first is not None and start is not None else 0.0,
                    "decode_s": max(0.0, (last - first) / 1000.0) if first is not None and last is not None else 0.0,
                    "latency_s": max(0.0, (end - start) / 1000.0) if end is not None and start is not None else 0.0,
                    "num_tool_calls": n_tools,
                })
                pending_start = None
            elif kind == "tool/call":
                record["tool_calls"] += 1
            elif kind == "turn/end":
                record["finish_reason"] = ((data.get("reason") or {}).get("kind"))
            elif kind == "team/member":
                member = data.get("member") or {}
                if member.get("id"):
                    record["members"][str(member["id"])] = str(member.get("name") or member["id"])
        sessions.append(record)
    return sessions


def sessions_to_row(
    sessions: List[Dict[str, Any]],
    *,
    task_id: str,
    output_text: str,
    e2e_latency_s: float,
    errors: List[str],
    meta: Mapping[str, Any],
) -> Dict[str, Any]:
    leads = [s for s in sessions if not s["parent"]]
    names: Dict[str, str] = {}
    for lead in leads:
        names.update(lead["members"])

    def role_of(session: Mapping[str, Any]) -> str:
        if not session["parent"]:
            return "lead" if len(leads) == 1 else f"lead-{session['id'][-8:]}"
        for member_id, name in names.items():
            if member_id in session["id"]:
                return name
        return f"subagent-{session['id'][-8:]}"

    per_role: Dict[str, Dict[str, int]] = {}
    turn_stats: List[Dict[str, Any]] = []
    tool_calls = 0
    for session in sessions:
        role = role_of(session)
        usage = per_role.setdefault(role, {
            "input_tokens": 0, "output_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": 0, "cached_tokens": 0, "requests": 0,
        })
        tool_calls += int(session["tool_calls"])
        for req in session["requests"]:
            usage["input_tokens"] += req["input_tokens"]
            usage["output_tokens"] += req["output_tokens"]
            usage["completion_tokens"] += req["output_tokens"]
            usage["reasoning_tokens"] += req["reasoning_tokens"]
            usage["cached_tokens"] += req["cached_tokens"]
            usage["requests"] += 1
            turn_stats.append({
                "role": role,
                "start_ms": req["start_ms"],
                "input_tokens": req["input_tokens"],
                "output_tokens": req["output_tokens"],
                "completion_tokens": req["output_tokens"],
                "reasoning_tokens": req["reasoning_tokens"],
                "cached_tokens": req["cached_tokens"],
                "latency_s": round(req["latency_s"], 6),
                "ttft_s": round(req["ttft_s"], 6),
                "decode_s": round(req["decode_s"], 6),
                "num_tool_calls": req["num_tool_calls"],
            })
    turn_stats.sort(key=lambda r: (r["start_ms"] is None, r["start_ms"] or 0))
    total = {k: sum(u[k] for u in per_role.values()) for k in (
        "input_tokens", "output_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "requests")}
    teammates = [s for s in sessions if s["parent"]]
    team = {
        "teammates_expected": int(meta.get("teammates") or 0),
        "teammates_spawned": len(names),
        "teammate_sessions": len(teammates),
        "teammates_completed": sum(1 for s in teammates if s["finish_reason"] == "completed"),
    }
    return {
        "task_id": task_id,
        "strategy": meta.get("strategy"),
        "output_text": output_text,
        "e2e_latency_s": round(e2e_latency_s, 4),
        "per_role_usage": per_role,
        "total_usage": total,
        "errors": errors,
        "num_turns": total["requests"],
        "tool_calls": tool_calls,
        "turn_stats": turn_stats,
        "agentcard": {**dict(meta), "team": team},
    }


def final_text_from_stdout(stdout_jsonl: str) -> str:
    final = ""
    for line in stdout_jsonl.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "final":
            final = str(event.get("text") or "")
    return final


# --------------------------------------------------------------------------- #
# Harbor agent
# --------------------------------------------------------------------------- #

class DSHAgent(BaseAgent):
    """DeepSeek Harness inside the task container; optional Agent Teams."""

    capabilities = AgentCapabilities(base_url=True, reasoning_effort=True)
    MODEL_CONNECTION = connection_spec(*FORMAT_TO_PROTOCOL)

    def __init__(
        self,
        logs_dir: Path,
        model_name: Optional[str] = None,
        *,
        model: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key_env: str = "DSH_API_KEY",
        protocol: Optional[str] = None,
        context_window: int = 131072,
        max_tokens: int = 16384,
        reasoning_effort: Optional[str] = None,
        compat: Optional[Mapping[str, Any]] = None,
        reasoning_efforts: Any = None,
        teammates: int = 0,
        dsh_version: Optional[str] = None,
        version: Optional[str] = None,
        workdir: str = "/workspace",
        answer_file: Optional[str] = None,
        team_prompt: Optional[str] = None,
        install_dir: str = "/opt/agentcard-dsh",
        node_major: int = 24,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Two ways to pick the model: Harbor's connection (model_name, base_url,
        # model_api_key_name, model_api_format), or the `model`/`endpoint_url`/
        # `api_key_env` kwargs (`base_url` itself is Harbor's field).
        super().__init__(logs_dir, model_name, **kwargs)
        if model and not endpoint_url:
            raise ValueError("DSHAgent: `model` needs `endpoint_url` (or use Harbor's --model/--base-url)")
        if not model and not model_name:
            raise ValueError("DSHAgent needs a model: Harbor's --model, or the `model` and `endpoint_url` kwargs")
        if protocol is not None and protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}")
        if int(teammates) < 0:
            raise ValueError("teammates must be >= 0")
        self.model = model
        self.dsh_base_url = endpoint_url
        self.api_key_env = api_key_env
        self.protocol = protocol
        self.context_window = int(context_window)
        self.max_tokens = int(max_tokens)
        self.reasoning_effort = reasoning_effort
        self.compat = dict(compat) if compat else None
        self.reasoning_efforts = reasoning_efforts
        self.teammates = int(teammates)
        self.dsh_version = dsh_version or version or DSH_VERSION
        self.workdir = workdir
        self.answer_file = answer_file
        self.team_prompt = team_prompt if team_prompt is not None else TEAM_PROMPT
        self.install_dir = install_dir
        self.node_major = int(node_major)
        self.task_id = task_id
        self._path_prefix = ""

    @staticmethod
    def name() -> str:
        return "deepseek-harness"

    def version(self) -> Optional[str]:
        return self.dsh_version

    @property
    def _env_home(self) -> PurePosixPath:
        return PurePosixPath(self.environment_logs_dir) / "dsh-home"

    async def _sh(self, environment: BaseEnvironment, command: str, *, timeout: Optional[int] = 1800,
                  env: Optional[Dict[str, str]] = None, cwd: str = "/"):
        return await environment.exec(command, cwd=cwd, env=env, timeout_sec=timeout)

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install Node (if missing or < 22) and the pinned DSH CLI in the container."""
        from harbor.agents.installed.node_install import nvm_node_install_snippet

        probe = await self._sh(environment, "node --version 2>/dev/null || true", timeout=60)
        version = (probe.stdout or "").strip().lstrip("v")
        major = int(version.split(".")[0]) if version[:1].isdigit() else 0
        if major < 22:
            res = await self._sh(
                environment,
                "command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates) ; "
                + nvm_node_install_snippet(self.node_major)
                + ' && dirname "$(command -v node)"',
                timeout=1800,
            )
            if res.return_code != 0:
                raise RuntimeError(f"Node install failed: {(res.stderr or res.stdout or '')[-2000:]}")
            self._path_prefix = (res.stdout or "").strip().splitlines()[-1]
        bin_path = f"{self.install_dir}/node_modules/.bin/dsh"
        path_env = self._path_env()
        current = await self._sh(environment, f"{bin_path} --version 2>/dev/null || true", timeout=120, env=path_env)
        if (current.stdout or "").strip() != self.dsh_version:
            pkg = shlex.quote(f"@deepseek-ai/dsh@{self.dsh_version}")
            res = await self._sh(
                environment,
                f"mkdir -p {shlex.quote(self.install_dir)} && "
                f"npm install --prefix {shlex.quote(self.install_dir)} --no-audit --no-fund --loglevel=error {pkg} "
                f"&& {bin_path} --version",
                timeout=1800,
                env=path_env,
            )
            if res.return_code != 0 or (res.stdout or "").strip().splitlines()[-1:] != [self.dsh_version]:
                raise RuntimeError(f"DSH install failed: {(res.stderr or res.stdout or '')[-2000:]}")

    def endpoint(self) -> Dict[str, Any]:
        """``{model, base_url, protocol, api_key}`` for the DSH profile and the dsh process."""
        if self.model:  # legacy kwargs; Harbor's base_url (set by the meter) takes precedence
            return {
                "model": self.model,
                "base_url": self._base_url or self.dsh_base_url,
                "protocol": self.protocol or "openai-completions",
                "api_key": os.environ.get(self.api_key_env, ""),
            }
        ep = resolve_endpoint(self)
        if not ep.base_url:
            raise ValueError(f"DSHAgent: no endpoint for model {self.model_name!r}; pass --base-url")
        protocol = self.protocol or FORMAT_TO_PROTOCOL.get(ep.api_format, "openai-completions")
        return {"model": ep.model, "base_url": ep.base_url, "protocol": protocol, "api_key": ep.api_key or ""}

    def upstream_base_url(self) -> Optional[str]:
        """The endpoint before any metering proxy (used by agent_cap.harbor.metered)."""
        if self.model:
            return self.dsh_base_url
        ep = resolve_endpoint(self)
        return ep.base_url if ep else None

    def _path_env(self) -> Dict[str, str]:
        if not self._path_prefix:
            return {}
        return {"PATH": f"{self._path_prefix}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}

    def _prompt(self, instruction: str) -> str:
        if self.teammates <= 0:
            return instruction
        return instruction + self.team_prompt.format(teammates=self.teammates, workdir=self.workdir)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        home = self._env_home
        profile_dir = home / "profiles" / PROFILE
        sessions_root = home / "sessions"
        endpoint = self.endpoint()
        files = build_profile(
            model=endpoint["model"],
            base_url=endpoint["base_url"],
            protocol=endpoint["protocol"],
            teammates=self.teammates,
            sessions_root=str(sessions_root),
            context_window=self.context_window,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            compat=self.compat,
            reasoning_efforts=self.reasoning_efforts,
        )
        await self._sh(environment, f"mkdir -p {shlex.quote(str(profile_dir))} /tmp/agentcard-dsh-user", timeout=60)
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in files.items():
                local = Path(tmp) / name
                local.write_text(json.dumps(content, indent=1), encoding="utf-8")
                await environment.upload_file(local, str(profile_dir / name))

        key = endpoint["api_key"]
        run_env = {
            **self._path_env(),
            "DSH_HOME": str(home),
            "HOME": "/tmp/agentcard-dsh-user",
            "DSH_PERMISSION_MODE": "danger-full-access",
            "DSH_TELEMETRY_MODE": "DISABLED",
            KEY_ENV_IN_CONTAINER: key or "EMPTY",
        }
        stdout_path = PurePosixPath(self.environment_logs_dir) / "dsh-stdout.jsonl"
        stderr_path = PurePosixPath(self.environment_logs_dir) / "dsh-stderr.log"
        command = (
            f"{self.install_dir}/node_modules/.bin/dsh --profile {PROFILE} --json "
            f"{shlex.quote(self._prompt(instruction))} < /dev/null "
            f"> {shlex.quote(str(stdout_path))} 2> {shlex.quote(str(stderr_path))}"
        )
        started = time.perf_counter()
        errors: List[str] = []
        result = None
        try:
            result = await self._sh(environment, command, timeout=None, env=run_env, cwd=self.workdir)
            if result.return_code != 0:
                errors.append(f"dsh exited with status {result.return_code}")
        except BaseException as exc:  # Harbor cancels run() on agent timeout
            errors.append(f"{type(exc).__name__}: {exc}"[:500])
            self._finish(context, started, errors, endpoint)
            raise
        await self._ensure_logs_on_host(environment)
        output_text = self._finish(context, started, errors, endpoint)
        if self.answer_file:
            await self._write_answer(environment, output_text)

    async def _ensure_logs_on_host(self, environment: BaseEnvironment) -> None:
        """Docker mounts /logs/agent; other environments may not, so download it."""
        host_logs = Path(self.logs_dir)
        if (host_logs / "dsh-stdout.jsonl").exists() and any((host_logs / "dsh-home").rglob("session.v4.jsonl")):
            return
        try:
            await environment.download_dir(str(self.environment_logs_dir), host_logs)
        except Exception as exc:  # keep going: the row records the missing logs
            self.logger.warning("could not download DSH logs: %s", exc)

    def _finish(self, context: AgentContext, started: float, errors: List[str], endpoint: Mapping[str, Any]) -> str:
        host_logs = Path(self.logs_dir)
        stdout = host_logs / "dsh-stdout.jsonl"
        output_text = final_text_from_stdout(stdout.read_text(encoding="utf-8")) if stdout.exists() else ""
        sessions = read_sessions(host_logs / "dsh-home" / "sessions")
        if not sessions:
            errors.append("no DSH session logs found under agent/dsh-home/sessions")
        meta = {
            "agent": self.name(),
            "dsh_version": self.dsh_version,
            "strategy": "dsh-agent-team" if self.teammates > 0 else "dsh-single",
            "teammates": self.teammates,
            "protocol": endpoint["protocol"],
            "roles": {"lead": endpoint["model"], **({"teammates": endpoint["model"]} if self.teammates > 0 else {})},
        }
        row = sessions_to_row(
            sessions,
            task_id=self.task_id or _trial_task_id(host_logs),
            output_text=output_text,
            e2e_latency_s=time.perf_counter() - started,
            errors=errors,
            meta=meta,
        )
        host_logs.mkdir(parents=True, exist_ok=True)
        (host_logs / RESULT_FILENAME).write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
        usage = row["total_usage"]
        context.n_input_tokens = usage["input_tokens"]
        context.n_output_tokens = usage["output_tokens"]
        context.n_cache_tokens = usage["cached_tokens"]
        context.metadata = {
            "per_role_usage": row["per_role_usage"],
            "num_requests": usage["requests"],
            "team": row["agentcard"]["team"],
            "errors": errors,
            "capagent_result": RESULT_FILENAME,
        }
        return output_text

    async def _write_answer(self, environment: BaseEnvironment, text: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "answer.txt"
            local.write_text(text, encoding="utf-8")
            parent = str(PurePosixPath(self.answer_file).parent)
            await self._sh(environment, f"mkdir -p {shlex.quote(parent)}", timeout=60)
            await environment.upload_file(local, self.answer_file)


def _trial_task_id(logs_dir: Path) -> str:
    trial = logs_dir.parent.name if logs_dir.name == "agent" else logs_dir.name
    return trial.rsplit("__", 1)[0] if "__" in trial else trial


__all__ = ["DSHAgent", "build_profile", "read_sessions", "sessions_to_row", "DSH_VERSION", "PROTOCOLS"]
