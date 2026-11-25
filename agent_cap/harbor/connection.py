"""Harbor's model connection, shared by every harness AgentCAP runs on Harbor.

Harbor gives every agent the same connection fields (``harbor run`` flags in
brackets): ``model_name`` (``--model``), ``base_url`` (``--base-url``),
``model_api_key_name`` (``--api-key-name``, the *name* of the environment
variable that holds the key), ``model_api_format`` (``--api-format``) and
``reasoning_effort`` (``--effort``). Codex, Claude Code, OpenHands and the other
built-in agents read them; CAPAgent and DSHAgent read them through
:func:`resolve_endpoint`, so one set of flags selects the model for any harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from harbor.agents.agent_auth_resolution import ModelConnectionSpec
from harbor.models.model_api_format import ModelApiFormat


def connection_spec(*formats: ModelApiFormat) -> ModelConnectionSpec:
    """Spec for an agent that talks to any endpoint in ``formats`` (first is the default).

    Unknown provider prefixes fall back to an OpenAI-compatible endpoint, so a
    bare ``Qwen/Qwen3.5-4B`` with ``--base-url`` resolves to that model id.
    """
    default = formats[0]
    by_provider = {"default": default}
    if ModelApiFormat.ANTHROPIC_MESSAGES in formats:
        by_provider["anthropic"] = ModelApiFormat.ANTHROPIC_MESSAGES
    return ModelConnectionSpec(
        default_provider="openai",
        supported_api_formats=list(formats),
        default_api_format_by_provider=by_provider,
    )


@dataclass(frozen=True)
class Endpoint:
    """What a harness needs to call the model."""

    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    api_key_env: Optional[str]  # variable the key came from, if any
    api_format: Optional[ModelApiFormat]
    provider: Optional[str]


def resolve_endpoint(agent: Any) -> Optional[Endpoint]:
    """The model endpoint Harbor resolved for ``agent``; None without ``--model``."""
    if not getattr(agent, "model_name", None):
        return None
    conn = agent.model_connection
    return Endpoint(
        model=conn.model_id or agent.model_name,
        base_url=conn.execution_base_url or conn.inference_base_url,
        api_key=conn.api_key,
        api_key_env=conn.api_key_source_name,
        api_format=conn.model_api_format,
        provider=conn.provider,
    )


__all__ = ["Endpoint", "connection_spec", "resolve_endpoint"]
