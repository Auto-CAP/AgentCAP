"""
Timer implementations for Agent-CAP benchmarking.

Supports both CPU timing (time.time) and GPU timing (CUDA events).
"""

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class Timer(ABC):
    """Abstract base class for timers."""

    @abstractmethod
    def start(self) -> None:
        """Record the start time."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Record the end time."""
        pass

    @abstractmethod
    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        pass

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


class TimeTimer(Timer):
    """
    CPU timer using time.perf_counter() for high-resolution timing.

    This is the default timer for most use cases. Uses time.perf_counter()
    which provides the highest resolution timer available on the platform.
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self) -> None:
        """Record start time using perf_counter."""
        self._start_time = time.perf_counter()
        self._end_time = None

    def stop(self) -> None:
        """Record end time using perf_counter."""
        self._end_time = time.perf_counter()

    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        if self._start_time is None:
            raise RuntimeError("Timer was never started")
        if self._end_time is None:
            # Timer still running, return current elapsed time
            return time.perf_counter() - self._start_time
        return self._end_time - self._start_time

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed() * 1000

    def get_timestamps(self) -> Tuple[float, float]:
        """Return (start_time, end_time) tuple."""
        if self._start_time is None:
            raise RuntimeError("Timer was never started")
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return (self._start_time, end)


class CudaTimer(Timer):
    """
    GPU timer using CUDA events for accurate GPU timing.

    Uses torch.cuda.Event for precise GPU timing that accounts for
    asynchronous kernel execution. Falls back to TimeTimer if CUDA
    is not available.

    Note: CUDA event timing measures GPU time, which may differ from
    wall-clock time due to GPU-CPU synchronization.
    """

    def __init__(self):
        self._torch_available = False
        self._cuda_available = False
        self._start_event = None
        self._end_event = None
        self._fallback_timer: Optional[TimeTimer] = None

        try:
            import torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

        if not self._cuda_available:
            self._fallback_timer = TimeTimer()

    def start(self) -> None:
        """Record start time using CUDA event or fallback to CPU timer."""
        if self._fallback_timer is not None:
            self._fallback_timer.start()
            return

        import torch
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def stop(self) -> None:
        """Record end time using CUDA event or fallback to CPU timer."""
        if self._fallback_timer is not None:
            self._fallback_timer.stop()
            return

        if self._end_event is not None:
            self._end_event.record()

    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        if self._fallback_timer is not None:
            return self._fallback_timer.elapsed()

        import torch
        if self._start_event is None or self._end_event is None:
            raise RuntimeError("Timer was never started")

        # Synchronize to ensure the events have been recorded
        torch.cuda.synchronize()

        # elapsed_time returns milliseconds
        return self._start_event.elapsed_time(self._end_event) / 1000.0

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed() * 1000

    @property
    def is_cuda(self) -> bool:
        """Return True if using actual CUDA timing."""
        return self._cuda_available and self._fallback_timer is None


class MonotonicTimer(Timer):
    """
    Timer using time.monotonic() for measuring intervals.

    Monotonic time cannot go backwards, making it suitable for
    measuring elapsed time even if system time changes.
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._end_time = None

    def stop(self) -> None:
        self._end_time = time.monotonic()

    def elapsed(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Timer was never started")
        if self._end_time is None:
            return time.monotonic() - self._start_time
        return self._end_time - self._start_time


def create_timer(use_cuda: bool = False) -> Timer:
    """
    Factory function to create appropriate timer.

    Args:
        use_cuda: If True, attempt to use CUDA timing. Falls back to CPU
                  timing if CUDA is not available.

    Returns:
        A Timer instance (CudaTimer or TimeTimer).
    """
    if use_cuda:
        return CudaTimer()
    return TimeTimer()
