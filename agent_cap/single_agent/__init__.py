"""Single-agent benchmarking module for AgentCAP.

Benchmarks a single LLM agent (e.g. GPT-OSS-120b via vLLM) across
different batch sizes, with and without tool calls, collecting:

- User-facing: E2E latency per request, Requests per second (RPS)
- Inference engine: Avg / P99 TTFT and TPOT
- Agentic-specific: Total input / output tokens, tool call count, avg tool latency
- Hardware: GPU utilisation, CPU utilisation

Tool calls execute inside Docker containers (SWE-bench Pro pre-built images).
"""

from agent_cap.single_agent.config import SingleAgentBenchConfig
from agent_cap.single_agent.runner import SingleAgentRunner
from agent_cap.single_agent.metrics import aggregate_metrics, BenchmarkMetrics
from agent_cap.single_agent.tool_executor import ToolExecutor

__all__ = [
    "SingleAgentBenchConfig",
    "SingleAgentRunner",
    "aggregate_metrics",
    "BenchmarkMetrics",
    "ToolExecutor",
]
