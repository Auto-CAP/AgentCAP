"""Metrics aggregation for single-agent benchmarking.

Collects per-request streaming responses and hardware snapshots, then
computes the full metrics suite:

- **User-facing**: E2E latency (avg/p50/p99), Requests Per Second (RPS)
- **Inference engine**: TTFT avg/p99, TPOT avg/p99
- **Agentic-specific**: total input/output tokens, tool call count, avg tool
  call latency
- **Hardware**: avg/max GPU utilisation, avg/max CPU utilisation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_cap.server.streaming_client import StreamingChatResponse


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * pct / 100.0), len(s) - 1)
    return s[idx]


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark results for one (batch_size, tool_mode) run."""

    # Identification
    batch_size: int = 0
    tool_mode: str = "no_tools"  # "no_tools" | "with_tools"
    num_requests: int = 0

    # User-facing
    e2e_latency_avg_ms: float = 0.0
    e2e_latency_p50_ms: float = 0.0
    e2e_latency_p99_ms: float = 0.0
    requests_per_second: float = 0.0

    # Inference engine
    ttft_avg_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tpot_avg_ms: float = 0.0
    tpot_p99_ms: float = 0.0

    # Agentic-specific
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0
    avg_tool_call_latency_ms: float = 0.0

    # Hardware
    avg_gpu_util_pct: float = 0.0
    max_gpu_util_pct: float = 0.0
    avg_cpu_util_pct: float = 0.0
    max_cpu_util_pct: float = 0.0

    # Errors
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "tool_mode": self.tool_mode,
            "num_requests": self.num_requests,
            "e2e_latency_avg_ms": round(self.e2e_latency_avg_ms, 2),
            "e2e_latency_p50_ms": round(self.e2e_latency_p50_ms, 2),
            "e2e_latency_p99_ms": round(self.e2e_latency_p99_ms, 2),
            "requests_per_second": round(self.requests_per_second, 3),
            "ttft_avg_ms": round(self.ttft_avg_ms, 2),
            "ttft_p99_ms": round(self.ttft_p99_ms, 2),
            "tpot_avg_ms": round(self.tpot_avg_ms, 2),
            "tpot_p99_ms": round(self.tpot_p99_ms, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tool_calls": self.total_tool_calls,
            "avg_tool_call_latency_ms": round(self.avg_tool_call_latency_ms, 2),
            "avg_gpu_util_pct": round(self.avg_gpu_util_pct, 1),
            "max_gpu_util_pct": round(self.max_gpu_util_pct, 1),
            "avg_cpu_util_pct": round(self.avg_cpu_util_pct, 1),
            "max_cpu_util_pct": round(self.max_cpu_util_pct, 1),
            "error_count": self.error_count,
        }


def aggregate_metrics(
    responses: List[StreamingChatResponse],
    batch_size: int,
    tool_mode: str,
    wall_clock_s: float,
    gpu_avg_util: float = 0.0,
    gpu_max_util: float = 0.0,
    cpu_avg_util: float = 0.0,
    cpu_max_util: float = 0.0,
    tool_call_latencies_ms: Optional[List[float]] = None,
) -> BenchmarkMetrics:
    """Aggregate a batch of streaming responses into a single metrics record.

    Args:
        responses: List of per-request streaming responses.
        batch_size: Concurrency level used.
        tool_mode: ``"no_tools"`` or ``"with_tools"``.
        wall_clock_s: Total wall-clock time for the batch.
        gpu_avg_util: Average GPU utilisation during the batch.
        gpu_max_util: Peak GPU utilisation during the batch.
        cpu_avg_util: Average CPU utilisation during the batch.
        cpu_max_util: Peak CPU utilisation during the batch.
        tool_call_latencies_ms: Per-tool-call latencies (if measured).

    Returns:
        Populated BenchmarkMetrics.
    """
    if not responses:
        return BenchmarkMetrics(batch_size=batch_size, tool_mode=tool_mode)

    # Separate successful and failed
    ok = [r for r in responses if r.error is None]
    errors = [r for r in responses if r.error is not None]

    latencies = [r.latency_ms for r in ok]
    ttfts = [r.ttft_ms for r in ok]
    tpot_avgs = [r.tpot_ms_avg for r in ok if r.tpot_ms_avg > 0]
    tpot_p99s = [r.tpot_ms_p99 for r in ok if r.tpot_ms_p99 > 0]

    total_input = sum(r.input_tokens for r in ok)
    total_output = sum(r.output_tokens for r in ok)
    total_tc = sum(r.tool_call_count for r in ok)

    rps = len(ok) / wall_clock_s if wall_clock_s > 0 else 0.0

    tc_lats = tool_call_latencies_ms or []
    avg_tc_lat = _mean(tc_lats) if tc_lats else 0.0

    return BenchmarkMetrics(
        batch_size=batch_size,
        tool_mode=tool_mode,
        num_requests=len(responses),
        # User-facing
        e2e_latency_avg_ms=_mean(latencies),
        e2e_latency_p50_ms=_percentile(latencies, 50),
        e2e_latency_p99_ms=_percentile(latencies, 99),
        requests_per_second=rps,
        # Inference engine
        ttft_avg_ms=_mean(ttfts),
        ttft_p99_ms=_percentile(ttfts, 99),
        tpot_avg_ms=_mean(tpot_avgs),
        tpot_p99_ms=_percentile(tpot_p99s, 99)
        if tpot_p99s
        else _percentile(tpot_avgs, 99),
        # Agentic-specific
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tool_calls=total_tc,
        avg_tool_call_latency_ms=avg_tc_lat,
        # Hardware
        avg_gpu_util_pct=gpu_avg_util,
        max_gpu_util_pct=gpu_max_util,
        avg_cpu_util_pct=cpu_avg_util,
        max_cpu_util_pct=cpu_max_util,
        # Errors
        error_count=len(errors),
    )
