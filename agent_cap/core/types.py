"""Core data types for Agent-CAP benchmarking."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List
import json
from datetime import datetime


class StepType(Enum):
    """
    Types of steps in an agentic workflow.

    Based on the CAP Decomposition Principle:
    - Planning and Reasoning: GPU memory bandwidth bound during token generation
    - Retrieval: Storage I/O and embedding computation bound
    - Tool Calling: CPU bound for parsing, serialization, and network I/O
    - Code Execution: CPU compute and sandbox overhead
    """
    PLANNING = auto()      # High-level task decomposition
    REASONING = auto()     # Chain-of-thought, inference
    RETRIEVAL = auto()     # RAG, document fetch, embedding lookup
    TOOL_CALLING = auto()  # External API calls, function execution
    CODE_EXECUTION = auto()  # Running generated code
    PREFILL = auto()       # LLM prefill phase (compute-bound)
    DECODE = auto()        # LLM decode phase (memory-bandwidth-bound)
    EMBEDDING = auto()     # Embedding computation
    OTHER = auto()         # Miscellaneous steps

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class Step:
    """
    Represents a single atomic operation in an agentic workflow.

    Attributes:
        name: Human-readable name for the step
        step_type: Category of the step (planning, retrieval, etc.)
        start_time: Timestamp when step started (seconds since trace start)
        end_time: Timestamp when step ended (seconds since trace start)
        metadata: Additional information about the step
        parent_id: ID of parent step if nested
        step_id: Unique identifier for this step
        thread_id: Thread/process identifier (for parallel execution)
    """
    name: str
    step_type: StepType
    start_time: float
    end_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    step_id: str = ""
    thread_id: str = "main"

    @property
    def duration(self) -> float:
        """Duration of the step in seconds."""
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> float:
        """Duration of the step in milliseconds."""
        return self.duration * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "name": self.name,
            "step_type": str(self.step_type),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "step_id": self.step_id,
            "thread_id": self.thread_id,
        }


@dataclass
class Trace:
    """
    A complete trace of an agentic workflow execution.

    Contains all steps and metadata for a single workflow run.
    """
    name: str
    steps: List[Step] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    trace_id: str = ""

    @property
    def total_duration(self) -> float:
        """Total duration of all steps in seconds."""
        if not self.steps:
            return 0.0
        return max(s.end_time for s in self.steps) - min(s.start_time for s in self.steps)

    @property
    def total_duration_ms(self) -> float:
        """Total duration in milliseconds."""
        return self.total_duration * 1000

    def get_steps_by_type(self, step_type: StepType) -> List[Step]:
        """Get all steps of a specific type."""
        return [s for s in self.steps if s.step_type == step_type]

    def get_duration_by_type(self, step_type: StepType) -> float:
        """Get total duration for a step type in seconds."""
        return sum(s.duration for s in self.get_steps_by_type(step_type))

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the trace."""
        type_durations = {}
        type_counts = {}

        for step_type in StepType:
            steps = self.get_steps_by_type(step_type)
            if steps:
                type_durations[str(step_type)] = sum(s.duration_ms for s in steps)
                type_counts[str(step_type)] = len(steps)

        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "total_steps": len(self.steps),
            "total_duration_ms": self.total_duration_ms,
            "duration_by_type_ms": type_durations,
            "count_by_type": type_counts,
            "metadata": self.metadata,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
            "summary": self.summary(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert trace to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str) -> None:
        """Save trace to a JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: str) -> "Trace":
        """Load trace from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        steps = []
        for s in data.get("steps", []):
            step_type = StepType[s["step_type"].upper()]
            steps.append(Step(
                name=s["name"],
                step_type=step_type,
                start_time=s["start_time"],
                end_time=s["end_time"],
                metadata=s.get("metadata", {}),
                parent_id=s.get("parent_id"),
                step_id=s.get("step_id", ""),
                thread_id=s.get("thread_id", "main"),
            ))

        trace = cls(
            name=data["name"],
            steps=steps,
            metadata=data.get("metadata", {}),
            trace_id=data.get("trace_id", ""),
        )

        if data.get("start_time"):
            trace.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            trace.end_time = datetime.fromisoformat(data["end_time"])

        return trace
