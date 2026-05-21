"""Core dataclasses for the multi-agent module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelEndpoint:
    """OpenAI-compatible model endpoint."""

    name: str
    base_url: str = "http://localhost:30000/v1"
    api_key: str = "dummy"
    max_tokens: int = 4096
    temperature: float = 0.0
    use_streaming: bool = False
    openrouter_provider: str = ""
    protocol: str = ""
    engine: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEndpoint":
        return cls(
            name=str(data.get("name") or data.get("model") or ""),
            base_url=str(data.get("base_url", "http://localhost:30000/v1")),
            api_key=str(data.get("api_key", "dummy")),
            max_tokens=int(data.get("max_tokens", 4096) or 4096),
            temperature=float(data.get("temperature", 0.0) or 0.0),
            use_streaming=bool(data.get("use_streaming", False)),
            openrouter_provider=str(data.get("openrouter_provider", "")),
            protocol=str(data.get("protocol", "")),
            engine=str(data.get("engine", "")),
        )


@dataclass
class AgentSpec:
    """Declarative spec for one agent in a team."""

    role: str
    endpoint: ModelEndpoint
    system_prompt: str = ""
    can_call_tools: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, role: str, data: Dict[str, Any]) -> "AgentSpec":
        endpoint = ModelEndpoint.from_dict(data)
        return cls(
            role=role,
            endpoint=endpoint,
            system_prompt=str(data.get("system_prompt", "")),
            can_call_tools=bool(data.get("can_call_tools", True)),
            extra={k: v for k, v in data.items() if k not in {
                "name", "model", "base_url", "api_key", "max_tokens", "temperature",
                "use_streaming", "openrouter_provider", "protocol", "engine",
                "system_prompt", "can_call_tools",
            }},
        )


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    requests: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        self.requests += other.requests


@dataclass
class TurnRecord:
    """One LLM call in the trace."""

    role: str
    model: str
    messages_in: List[Dict[str, Any]]
    assistant: Dict[str, Any]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0


@dataclass
class RunResult:
    """Final result of a strategy run on one task."""

    task_id: str
    strategy: str
    output_text: str
    e2e_latency_s: float
    per_role_usage: Dict[str, Usage] = field(default_factory=dict)
    turns: List[TurnRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_usage(self) -> Usage:
        agg = Usage()
        for u in self.per_role_usage.values():
            agg.add(u)
        return agg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "output_text": self.output_text,
            "e2e_latency_s": round(self.e2e_latency_s, 4),
            "per_role_usage": {
                role: {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cached_tokens": u.cached_tokens,
                    "requests": u.requests,
                }
                for role, u in self.per_role_usage.items()
            },
            "total_usage": {
                "input_tokens": self.total_usage.input_tokens,
                "output_tokens": self.total_usage.output_tokens,
                "cached_tokens": self.total_usage.cached_tokens,
                "requests": self.total_usage.requests,
            },
            "errors": list(self.errors),
            "num_turns": len(self.turns),
            "extras": dict(self.extras),
        }


@dataclass
class Task:
    """A task to run through the multi-agent system."""

    task_id: str
    user_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        if "user_prompt" in data:
            prompt = str(data["user_prompt"])
        elif "messages" in data and data["messages"]:
            prompt = str(data["messages"][-1].get("content", ""))
        elif "question" in data:
            prompt = str(data["question"])
        else:
            prompt = ""
        return cls(
            task_id=str(data.get("task_id") or data.get("id") or "task-0"),
            user_prompt=prompt,
            metadata={k: v for k, v in data.items()
                      if k not in {"user_prompt", "messages", "question", "task_id", "id"}},
        )


__all__ = [
    "ModelEndpoint",
    "AgentSpec",
    "Usage",
    "TurnRecord",
    "RunResult",
    "Task",
]
