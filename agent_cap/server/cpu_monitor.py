"""CPU utilization monitor using /proc/stat (Linux).

Mirrors the GPUMonitor pattern: background thread polls at configurable
intervals, ``stop()`` returns an aggregated summary dataclass.

Falls back gracefully on non-Linux systems (returns zero-value summaries).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import List, Optional, Tuple


@dataclass
class CPUSnapshot:
    """Single point-in-time CPU measurement."""

    timestamp: float
    cpu_util_pct: float
    per_core_util_pct: List[float]
    memory_used_mb: float
    memory_total_mb: float


@dataclass
class CPUMetricsSummary:
    """Aggregated CPU metrics over a monitoring window."""

    avg_cpu_util_pct: float
    max_cpu_util_pct: float
    avg_memory_used_mb: float
    peak_memory_used_mb: float
    duration_s: float
    num_samples: int
    snapshots: List[CPUSnapshot] = field(default_factory=list)


class CPUMonitor:
    """Background CPU utilisation monitor.

    Usage::

        monitor = CPUMonitor(interval=0.5)
        monitor.start()
        # ... workload ...
        summary = monitor.stop()
        print(f"Avg CPU: {summary.avg_cpu_util_pct:.1f}%")
    """

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self._stop_event = Event()
        self._lock = Lock()
        self._thread: Optional[Thread] = None
        self._snapshots: List[CPUSnapshot] = []
        self._start_ts: Optional[float] = None
        self._end_ts: Optional[float] = None
        # Previous /proc/stat values for delta calculation
        self._prev_cpu_times: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background polling thread."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("CPUMonitor is already running")
        self._stop_event.clear()
        self._snapshots = []
        self._prev_cpu_times = None
        self._start_ts = time.monotonic()
        self._end_ts = None
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> CPUMetricsSummary:
        """Stop the monitor and return aggregated metrics."""
        if not self._thread:
            return CPUMetricsSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0, [])

        self._stop_event.set()
        self._thread.join(timeout=self.interval * 2 + 1)
        self._end_ts = self._end_ts or time.monotonic()

        with self._lock:
            snapshots = list(self._snapshots)

        duration = 0.0
        if self._start_ts is not None:
            duration = max(0.0, self._end_ts - self._start_ts)

        if not snapshots:
            return CPUMetricsSummary(0.0, 0.0, 0.0, 0.0, duration, 0, [])

        count = len(snapshots)
        avg_util = sum(s.cpu_util_pct for s in snapshots) / count
        max_util = max(s.cpu_util_pct for s in snapshots)
        avg_mem = sum(s.memory_used_mb for s in snapshots) / count
        peak_mem = max(s.memory_used_mb for s in snapshots)

        return CPUMetricsSummary(
            avg_cpu_util_pct=avg_util,
            max_cpu_util_pct=max_util,
            avg_memory_used_mb=avg_mem,
            peak_memory_used_mb=peak_mem,
            duration_s=duration,
            num_samples=count,
            snapshots=snapshots,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self._poll_once()
            if snapshot is not None:
                with self._lock:
                    self._snapshots.append(snapshot)
            self._stop_event.wait(self.interval)
        self._end_ts = time.monotonic()

    def _poll_once(self) -> Optional[CPUSnapshot]:
        cpu_util, per_core = self._read_cpu_util()
        mem_used, mem_total = self._read_meminfo()

        if cpu_util is None:
            return None

        return CPUSnapshot(
            timestamp=time.monotonic(),
            cpu_util_pct=cpu_util,
            per_core_util_pct=per_core,
            memory_used_mb=mem_used,
            memory_total_mb=mem_total,
        )

    # ------------------------------------------------------------------
    # /proc readers
    # ------------------------------------------------------------------

    def _read_cpu_util(self) -> Tuple[Optional[float], List[float]]:
        """Read CPU utilisation from /proc/stat.

        Returns (overall_pct, [per_core_pct]).  On non-Linux, returns
        (None, []).
        """
        stat_path = Path("/proc/stat")
        if not stat_path.exists():
            return None, []

        try:
            text = stat_path.read_text()
        except OSError:
            return None, []

        overall_pct: Optional[float] = None
        per_core: List[float] = []

        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            label = parts[0]

            if label == "cpu":
                # Overall aggregate
                idle, total = self._parse_cpu_line(parts)
                overall_pct = self._calc_delta_pct(idle, total, is_overall=True)
            elif label.startswith("cpu") and label[3:].isdigit():
                idle, total = self._parse_cpu_line(parts)
                pct = self._calc_delta_pct(idle, total, is_overall=False)
                if pct is not None:
                    per_core.append(pct)

        return overall_pct, per_core

    @staticmethod
    def _parse_cpu_line(parts: List[str]) -> Tuple[float, float]:
        """Parse a ``cpu`` line from /proc/stat into (idle, total) jiffies."""
        # Fields: user nice system idle iowait irq softirq steal ...
        values = [float(v) for v in parts[1:]]
        total = sum(values)
        idle = values[3] if len(values) > 3 else 0.0
        return idle, total

    def _calc_delta_pct(
        self, idle: float, total: float, *, is_overall: bool
    ) -> Optional[float]:
        """Compute CPU % from delta of idle/total jiffies (overall line only).

        For per-core lines we compute a simple snapshot-based estimate.
        """
        if is_overall:
            if self._prev_cpu_times is not None:
                prev_idle, prev_total = self._prev_cpu_times
                d_total = total - prev_total
                d_idle = idle - prev_idle
                if d_total > 0:
                    pct = (1.0 - d_idle / d_total) * 100.0
                else:
                    pct = 0.0
            else:
                # First sample — compute absolute utilisation
                pct = (1.0 - idle / total) * 100.0 if total > 0 else 0.0
            self._prev_cpu_times = (idle, total)
            return max(0.0, min(pct, 100.0))

        # Per-core: simple absolute snapshot (delta tracking per-core would
        # need a dict keyed by core id — keep it simple for now).
        pct = (1.0 - idle / total) * 100.0 if total > 0 else 0.0
        return max(0.0, min(pct, 100.0))

    @staticmethod
    def _read_meminfo() -> Tuple[float, float]:
        """Read memory stats from /proc/meminfo.

        Returns (used_mb, total_mb).  Returns (0, 0) on non-Linux.
        """
        meminfo_path = Path("/proc/meminfo")
        if not meminfo_path.exists():
            return 0.0, 0.0

        try:
            text = meminfo_path.read_text()
        except OSError:
            return 0.0, 0.0

        mem_total_kb = 0.0
        mem_available_kb = 0.0

        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            key = parts[0].rstrip(":")
            if key == "MemTotal" and len(parts) >= 2:
                mem_total_kb = float(parts[1])
            elif key == "MemAvailable" and len(parts) >= 2:
                mem_available_kb = float(parts[1])

        mem_total_mb = mem_total_kb / 1024.0
        mem_used_mb = (mem_total_kb - mem_available_kb) / 1024.0
        return mem_used_mb, mem_total_mb
