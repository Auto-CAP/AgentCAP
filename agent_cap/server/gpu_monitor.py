from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import List, Optional
import subprocess
import time


@dataclass
class GPUSnapshot:
    timestamp: float
    gpu_util_pct: float
    memory_used_mb: float
    memory_total_mb: float
    power_draw_w: float
    temperature_c: float


@dataclass
class GPUMetricsSummary:
    avg_gpu_util_pct: float
    max_gpu_util_pct: float
    avg_memory_used_mb: float
    peak_memory_used_mb: float
    avg_power_w: float
    duration_s: float
    num_samples: int
    snapshots: List[GPUSnapshot] = field(default_factory=list)


class GPUMonitor:
    def __init__(self, interval: float = 1.0, gpu_index: int = 0):
        self.interval = interval
        self.gpu_index = gpu_index
        self._stop_event = Event()
        self._lock = Lock()
        self._thread: Optional[Thread] = None
        self._snapshots: List[GPUSnapshot] = []
        self._start_ts: Optional[float] = None
        self._end_ts: Optional[float] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("GPUMonitor is already running")
        self._stop_event.clear()
        self._snapshots = []
        self._start_ts = time.monotonic()
        self._end_ts = None
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> GPUMetricsSummary:
        if not self._thread:
            return GPUMetricsSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, [])

        self._stop_event.set()
        self._thread.join(timeout=self.interval * 2 + 1)
        self._end_ts = self._end_ts or time.monotonic()

        with self._lock:
            snapshots = list(self._snapshots)

        duration = 0.0
        if self._start_ts is not None:
            duration = max(0.0, self._end_ts - self._start_ts)

        if not snapshots:
            return GPUMetricsSummary(0.0, 0.0, 0.0, 0.0, 0.0, duration, 0, [])

        count = len(snapshots)
        avg_util = sum(s.gpu_util_pct for s in snapshots) / count
        max_util = max(s.gpu_util_pct for s in snapshots)
        avg_mem = sum(s.memory_used_mb for s in snapshots) / count
        peak_mem = max(s.memory_used_mb for s in snapshots)
        avg_power = sum(s.power_draw_w for s in snapshots) / count

        return GPUMetricsSummary(
            avg_gpu_util_pct=avg_util,
            max_gpu_util_pct=max_util,
            avg_memory_used_mb=avg_mem,
            peak_memory_used_mb=peak_mem,
            avg_power_w=avg_power,
            duration_s=duration,
            num_samples=count,
            snapshots=snapshots,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self._poll_once()
            if snapshot is not None:
                with self._lock:
                    self._snapshots.append(snapshot)
            self._stop_event.wait(self.interval)
        self._end_ts = time.monotonic()

    def _poll_once(self) -> Optional[GPUSnapshot]:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (subprocess.SubprocessError, OSError):
            return None

        line = result.stdout.strip().splitlines()
        if not line:
            return None

        parts = [p.strip() for p in line[0].split(",")]
        if len(parts) != 5:
            return None

        try:
            gpu_util, mem_used, mem_total, power, temp = [float(v) for v in parts]
        except ValueError:
            return None

        return GPUSnapshot(
            timestamp=time.monotonic(),
            gpu_util_pct=gpu_util,
            memory_used_mb=mem_used,
            memory_total_mb=mem_total,
            power_draw_w=power,
            temperature_c=temp,
        )
