"""
Tracer for recording agentic workflow steps.

Provides decorators and context managers for easy instrumentation.
"""

import functools
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List, Union

from agent_cap.core.types import Step, Trace, StepType
from agent_cap.core.timer import Timer, TimeTimer, CudaTimer, create_timer


class Tracer:
    """
    Records steps in an agentic workflow for benchmarking.

    Usage:
        # As context manager
        tracer = Tracer("my-workflow")
        with tracer.step("planning", StepType.PLANNING):
            plan = agent.plan(task)

        # As decorator
        @tracer.trace_step("reasoning", StepType.REASONING)
        def reason(context):
            return llm.generate(context)

        # Get results
        trace = tracer.get_trace()
        trace.save("trace.json")
    """

    def __init__(
        self,
        name: str = "trace",
        use_cuda: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a new Tracer.

        Args:
            name: Name for this trace
            use_cuda: If True, use CUDA events for timing (requires PyTorch)
            metadata: Additional metadata to attach to the trace
        """
        self.name = name
        self.use_cuda = use_cuda
        self._steps: List[Step] = []
        self._metadata = metadata or {}
        self._trace_id = str(uuid.uuid4())[:8]
        self._step_counter = 0
        self._lock = threading.Lock()
        self._start_time: Optional[datetime] = None
        self._trace_start: Optional[float] = None  # Reference time for relative timestamps
        self._step_stack: List[str] = []  # For tracking nested steps
        self._active = False

    def start(self) -> "Tracer":
        """Start the trace recording."""
        self._start_time = datetime.now()
        self._trace_start = TimeTimer()
        self._trace_start.start()
        self._active = True
        return self

    def stop(self) -> Trace:
        """Stop the trace and return the Trace object."""
        self._active = False
        end_time = datetime.now()

        trace = Trace(
            name=self.name,
            steps=self._steps.copy(),
            metadata=self._metadata.copy(),
            start_time=self._start_time,
            end_time=end_time,
            trace_id=self._trace_id,
        )
        return trace

    def reset(self) -> None:
        """Reset the tracer for a new trace."""
        self._steps = []
        self._step_counter = 0
        self._step_stack = []
        self._trace_id = str(uuid.uuid4())[:8]
        self._start_time = None
        self._trace_start = None
        self._active = False

    def _get_relative_time(self) -> float:
        """Get time relative to trace start."""
        if self._trace_start is None:
            return 0.0
        return self._trace_start.elapsed()

    def _generate_step_id(self) -> str:
        """Generate a unique step ID."""
        with self._lock:
            self._step_counter += 1
            return f"{self._trace_id}-{self._step_counter}"

    @contextmanager
    def step(
        self,
        name: str,
        step_type: StepType = StepType.OTHER,
        metadata: Optional[Dict[str, Any]] = None,
        thread_id: str = "main",
    ):
        """
        Context manager to record a step.

        Args:
            name: Name of the step
            step_type: Type of step (planning, retrieval, etc.)
            metadata: Additional metadata for this step
            thread_id: Thread identifier for parallel steps

        Yields:
            The Step object being recorded

        Example:
            with tracer.step("fetch_documents", StepType.RETRIEVAL) as s:
                docs = retriever.get(query)
                s.metadata["doc_count"] = len(docs)
        """
        if not self._active:
            self.start()

        step_id = self._generate_step_id()
        parent_id = self._step_stack[-1] if self._step_stack else None

        step_obj = Step(
            name=name,
            step_type=step_type,
            start_time=self._get_relative_time(),
            metadata=metadata or {},
            parent_id=parent_id,
            step_id=step_id,
            thread_id=thread_id,
        )

        # Push to stack for nested steps
        self._step_stack.append(step_id)

        timer = create_timer(self.use_cuda)
        timer.start()

        try:
            yield step_obj
        finally:
            timer.stop()
            step_obj.end_time = self._get_relative_time()

            with self._lock:
                self._steps.append(step_obj)
                self._step_stack.pop()

    def trace_step(
        self,
        name: Optional[str] = None,
        step_type: StepType = StepType.OTHER,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator to trace a function as a step.

        Args:
            name: Name of the step (defaults to function name)
            step_type: Type of step
            metadata: Additional metadata

        Returns:
            Decorated function

        Example:
            @tracer.trace_step("generate_response", StepType.REASONING)
            def generate(prompt):
                return llm(prompt)
        """
        def decorator(func: Callable) -> Callable:
            step_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.step(step_name, step_type, metadata):
                    return func(*args, **kwargs)

            return wrapper
        return decorator

    def record_step(
        self,
        name: str,
        step_type: StepType,
        start_time: float,
        end_time: float,
        metadata: Optional[Dict[str, Any]] = None,
        thread_id: str = "main",
    ) -> Step:
        """
        Manually record a step with explicit timestamps.

        Useful when timing was done externally or for post-hoc recording.

        Args:
            name: Name of the step
            step_type: Type of step
            start_time: Start time in seconds (relative to trace start)
            end_time: End time in seconds (relative to trace start)
            metadata: Additional metadata
            thread_id: Thread identifier

        Returns:
            The recorded Step
        """
        if not self._active:
            self.start()

        step = Step(
            name=name,
            step_type=step_type,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata or {},
            step_id=self._generate_step_id(),
            thread_id=thread_id,
        )

        with self._lock:
            self._steps.append(step)

        return step

    def get_trace(self) -> Trace:
        """
        Get the current trace without stopping.

        Returns:
            Trace object with all recorded steps
        """
        return Trace(
            name=self.name,
            steps=self._steps.copy(),
            metadata=self._metadata.copy(),
            start_time=self._start_time,
            trace_id=self._trace_id,
        )

    def get_steps(self) -> List[Step]:
        """Get all recorded steps."""
        return self._steps.copy()

    def __enter__(self) -> "Tracer":
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()


# Global tracer instance for convenience
_global_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the global tracer instance, creating one if needed."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer("global")
    return _global_tracer


def set_tracer(tracer_instance: Tracer) -> None:
    """Set the global tracer instance."""
    global _global_tracer
    _global_tracer = tracer_instance


# Convenience decorator using global tracer
def tracer(
    name: Optional[str] = None,
    step_type: StepType = StepType.OTHER,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to trace a function using the global tracer.

    Args:
        name: Name of the step (defaults to function name)
        step_type: Type of step
        metadata: Additional metadata

    Example:
        @tracer("fetch_data", StepType.RETRIEVAL)
        def fetch_data(query):
            return db.query(query)
    """
    def decorator(func: Callable) -> Callable:
        step_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t = get_tracer()
            with t.step(step_name, step_type, metadata):
                return func(*args, **kwargs)

        return wrapper
    return decorator
