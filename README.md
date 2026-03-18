# AgentCAP

Benchmarking framework for AI agent systems — measuring **Cost**, **Accuracy**, and **Performance** across single-agent and multi-agent workloads.

## Architecture

```
agent_cap/
├── core/                  # Shared core
│   ├── agentic_loop.py    # LLM → tool calls → metrics (TTFT, TPOT, tokens)
│   ├── tool_backend.py    # Abstract ToolBackend interface
│   ├── evaluator.py       # Abstract Evaluator interface
│   └── metrics.py         # BenchmarkMetrics aggregation
├── backends/              # Tool execution backends (pluggable)
│   ├── swebench_backend   # Docker/Modal sandbox (SWE-bench Pro)
│   └── mcp_backend        # MCP server HTTP API (MCP-ATLAS)
├── evaluators/            # Evaluation methods (pluggable)
│   ├── swebench_eval      # run_script.sh + parser.py
│   └── gtfa_eval          # GTFA claims
├── workloads/             # Workload types
│   ├── single_agent       # 1 model autonomous loop
│   └── agent_team         # Planner + Executor (plan-execute)
└── server/
    └── streaming_client   # aiohttp SSE streaming (TTFT/TPOT)
```

## Supported Benchmarks

| Benchmark | Dataset | Tools | Evaluation |
|---|---|---|---|
| **SWE-bench Pro** | `ScaleAI/SWE-bench_Pro` | read/write/shell/search in Docker/Modal | run_script.sh per instance |
| **MCP-ATLAS** | `ScaleAI/mcp-atlas` | GitHub, fetch, whois via MCP server | GTFA claims |

## Workload Types

| Workload | Description |
|---|---|
| **Single Agent** | One model autonomously reads code, writes fixes, runs tests |
| **Agent Team** | Planner model creates step-by-step plan, Executor model follows it with tool calls |

## Metrics Collected

- **User-facing**: E2E latency (avg/p50/p99), Requests per second
- **Inference**: TTFT avg/p99, TPOT avg/p99
- **Agentic**: Total input/output tokens, tool call count, avg tool call latency
- **Hardware**: GPU utilization, CPU utilization

## Quick Start

### SWE-bench Pro (Single Agent)

```bash
pip install -e ".[all]"

# Start vLLM
vllm serve unsloth/GPT-OSS-120b --port 8000

# Run benchmark (Modal runtime for RunPod)
agent-cap single-agent configs/single_agent_with_tools.yaml \
    --runtime modal --limit 5 --max-turns 10
```

### MCP-ATLAS

```bash
# Start MCP server
docker run --rm -d --name mcp-atlas-env -p 1984:1984 mcp-atlas-env

# Run benchmark
agent-cap mcp-atlas configs/mcp_atlas.yaml --limit 10
```

### Agent Team (Plan-Execute)

```bash
# Start two models
vllm serve planner-model --port 8000
vllm serve executor-model --port 8001

# Run via workloads/agent_team.py
python scripts/run_hybrid_experiment.py --config configs/hybrid_config.yaml
```

## Adding a New Benchmark

1. Implement `ToolBackend` (how tools execute)
2. Implement `Evaluator` (how to judge results)
3. Both workloads (`single_agent`, `agent_team`) work automatically

## Acknowledgement

This project is supported by the Advanced Research and Invention Agency (ARIA)'s grant "Scaling Compute: AI at 1/1000th the cost. Technical Area 4 Benchmarking".
