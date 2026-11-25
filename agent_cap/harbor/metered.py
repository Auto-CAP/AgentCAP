"""Any Harbor agent, with every model request recorded by AgentCAP's metering proxy.

``agent_cap.harbor.metered:<agent>`` is the Harbor agent ``<agent>`` (``codex``,
``claude-code``, ``openhands``, ``mini-swe-agent``, ...; or ``dsh`` /
``capagent``) with one change: before the agent starts, its ``--base-url`` is
pointed at the proxy in :mod:`agent_cap.harbor.meter`, which forwards to the
endpoint the agent would have used. Install, flags, tools, prompts and logs are
the agent's own::

    harbor run --agent agent_cap.harbor.metered:codex --model openai/gpt-5.4 -p datasets/nl2repo

Each trial then also writes ``agent/meter_result.json``: the AgentCAP row (usage,
per-request ``turn_stats`` with TTFT and decode time) that
``agent_cap.harbor.export_teas`` turns into TEAS files.

Where the proxy is reachable from: the task container must reach this host.
``AGENTCAP_METER_HOST`` (or the ``meter_host`` agent kwarg) names the address;
otherwise, for Harbor's ``docker`` environment, the container's default gateway
is used. With neither, or for an agent that cannot take ``--base-url``, the
agent runs unmetered and ``meter_result.json`` says why.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName

from agent_cap.harbor import METER_FILENAME
from agent_cap.harbor.meter import MeterServer, MeterSession, records_to_row

# Short names for AgentCAP's own agents, next to Harbor's agent names.
OWN_AGENTS = {
    "capagent": "agent_cap.harbor.agent:CAPAgent",
    "dsh": "agent_cap.harbor.dsh:DSHAgent",
}
_CACHE: Dict[str, type] = {}
METER_KWARGS = ("meter_host", "meter")


def gateway_from_proc_route(text: str) -> Optional[str]:
    """Default gateway from /proc/net/route (little-endian hex)."""
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) > 2 and parts[1] == "00000000" and parts[2] != "00000000":
            raw = bytes.fromhex(parts[2])
            return ".".join(str(b) for b in reversed(raw))
    return None


def _trial_name(logs_dir: Path) -> str:
    return logs_dir.parent.name if logs_dir.name == "agent" else logs_dir.name


def metered(base: type, agent_name: Optional[str] = None) -> type:
    """A subclass of the Harbor agent class ``base`` that runs behind the metering proxy."""
    key = agent_name or f"{base.__module__}:{base.__qualname__}"
    if key in _CACHE:
        return _CACHE[key]

    class Metered(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, meter_host: Optional[str] = None, meter: bool = True, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._meter_host = meter_host
            self._meter_enabled = bool(meter) and str(meter).lower() not in ("false", "0", "no")
            self._meter_session: Optional[MeterSession] = None
            self._meter_note: Optional[str] = None
            self._meter_upstream: Optional[str] = None

        @classmethod
        def parse_options(cls, kwargs=None, env=None):  # the agent's own options exclude ours
            kwargs = {k: v for k, v in (kwargs or {}).items() if k not in METER_KWARGS}
            return super().parse_options(kwargs, env)

        @classmethod
        def preflight(cls, kwargs=None, env=None) -> None:
            kwargs = {k: v for k, v in (kwargs or {}).items() if k not in METER_KWARGS}
            return super().preflight(kwargs=kwargs, env=env)

        async def setup(self, environment: BaseEnvironment) -> None:
            await self._meter_attach(environment)
            await super().setup(environment)

        async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
            if self._meter_session is None and self._meter_note is None:
                await self._meter_attach(environment)
            started = time.perf_counter()
            errors: List[str] = []
            try:
                await super().run(instruction, environment, context)
            except BaseException as exc:  # includes Harbor's cancellation on timeout
                errors.append(f"{type(exc).__name__}: {exc}"[:500])
                raise
            finally:
                self._meter_write(time.perf_counter() - started, errors)

        # ------------------------------------------------------------------ #

        async def _meter_attach(self, environment: BaseEnvironment) -> None:
            if not self._meter_enabled:
                self._meter_note = "metering disabled (meter=false)"
                return
            try:
                upstream = self._meter_upstream_url()
            except Exception as exc:  # e.g. missing API key variable; the agent reports it itself
                self._meter_note = f"no endpoint to meter: {type(exc).__name__}: {exc}"[:300]
                return
            if not upstream:
                self._meter_note = "no endpoint to meter: the agent has no base URL; pass --base-url"
                return
            host = self._meter_host or os.environ.get("AGENTCAP_METER_HOST")
            if not host and str(environment.type()) in ("docker", "EnvironmentType.DOCKER"):
                res = await environment.exec("cat /proc/net/route", timeout_sec=30)
                host = gateway_from_proc_route(res.stdout or "")
            if not host:
                self._meter_note = (
                    f"no address for the proxy in a {environment.type()} environment; "
                    "set AGENTCAP_METER_HOST to a host the task container can reach"
                )
                return
            self._meter_upstream = upstream
            self._meter_session = MeterServer.shared().open(upstream, label=_trial_name(Path(self.logs_dir)))
            url = self._meter_session.url(host)
            self._base_url = url
            options = getattr(self, "options", None)
            if options is not None and hasattr(options, "base_url"):
                try:
                    options.base_url = url
                except Exception:
                    pass
            self.logger.info("metering %s -> %s via %s", self.name(), upstream, url)

        def _meter_upstream_url(self) -> Optional[str]:
            if hasattr(self, "upstream_base_url"):  # DSHAgent
                return self.upstream_base_url()
            if not type(self).capabilities.base_url or self.MODEL_CONNECTION is None:
                raise ValueError(f"agent {self.name()!r} does not support --base-url")
            conn = self.model_connection
            if not conn.can_configure_base_url:
                raise ValueError(f"this {self.name()} configuration does not support --base-url")
            return conn.execution_base_url or conn.inference_base_url

        def _meter_write(self, elapsed_s: float, errors: List[str]) -> None:
            logs = Path(self.logs_dir)
            logs.mkdir(parents=True, exist_ok=True)
            meta: Dict[str, Any] = {
                "agent": self.name(),
                "agent_version": _safe(self.version),
                "strategy": self.name(),
                "model": self.model_name,
            }
            if self._meter_session is None:
                payload: Dict[str, Any] = {"metered": False, "reason": self._meter_note, "agentcard": meta}
            else:
                records = self._meter_session.snapshot()
                MeterServer.shared().close(self._meter_session)
                payload = records_to_row(
                    records, task_id=_trial_name(logs).rsplit("__", 1)[0], e2e_latency_s=elapsed_s,
                    errors=errors, meta={**meta, "upstream": self._meter_upstream},
                )
                payload["metered"] = True
                payload["requests_raw"] = records
            (logs / METER_FILENAME).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # import_path() is "<module>:<__name__>"; naming the class after the agent keeps
    # it importable as agent_cap.harbor.metered:<agent>.
    Metered.__name__ = agent_name or f"Metered{base.__name__}"
    Metered.__qualname__ = Metered.__name__
    Metered.__module__ = __name__
    Metered.__doc__ = f"{base.__doc__ or base.__name__}\n\nMetered by agent_cap.harbor.meter."
    _CACHE[key] = Metered
    return Metered


def _safe(fn) -> Optional[str]:
    try:
        return fn()
    except Exception:
        return None


def resolve_agent_class(name: str) -> type:
    """Harbor agent name, AgentCAP short name, or ``module:Class`` -> agent class."""
    if name in OWN_AGENTS:
        name = OWN_AGENTS[name]
    if ":" in name:
        from harbor.utils.import_path import import_class

        return import_class(name, base=BaseAgent, label="agent")
    return AgentFactory.get_agent_class(AgentName(name))


def __getattr__(name: str) -> type:
    """``agent_cap.harbor.metered:codex`` -> the metered Codex agent class."""
    if name.startswith("__"):
        raise AttributeError(name)
    if name in _CACHE:
        return _CACHE[name]
    try:
        base = resolve_agent_class(name)
    except ValueError as exc:
        raise AttributeError(f"unknown Harbor agent {name!r}: {exc}") from None
    return metered(base, agent_name=name)


__all__ = ["metered", "resolve_agent_class", "gateway_from_proc_route", "OWN_AGENTS"]
