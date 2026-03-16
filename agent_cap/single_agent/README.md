# Single-Agent Benchmark

Benchmark a single LLM agent on SWE-Bench Pro, sweeping batch sizes with and without tool calls.

## Quick Start

```bash
# 1. Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model unsloth/GPT-OSS-120b --tensor-parallel-size 8 --port 30000

# 2. Run benchmark (dry-run first)
agent-cap single-agent configs/single_agent_gpt_oss_120b.yaml --dry-run

# 3. Run for real
agent-cap single-agent configs/single_agent_gpt_oss_120b.yaml
```

Results → `results/single_agent/metrics.csv` + `metrics.json`

## What It Measures

| Category | Metrics |
|---|---|
| User-facing | E2E latency (avg/p50/p99), RPS |
| Inference | TTFT avg/p99, TPOT avg/p99 |
| Agentic | Input/output tokens, tool call count, tool call latency |
| Hardware | GPU util%, CPU util% (nvidia-smi + /proc/stat) |

## Config

Edit `configs/single_agent_gpt_oss_120b.yaml`:

```yaml
model_id: unsloth/GPT-OSS-120b
serving_engine: vllm
base_url: http://localhost:30000
dataset: swebench_pro
batch_sizes: [1, 2, 4, 8, 16, 32]
enable_tool_calls: true
```

## Architecture

```
StreamingChatClient  ──→  vLLM (SSE streaming)
        │                     │
        │ TTFT/TPOT timing    │ token chunks
        ▼                     ▼
  SingleAgentRunner  ←── GPUMonitor + CPUMonitor
        │
        ▼
  BenchmarkMetrics  ──→  CSV / JSON
```

`batch_sizes` controls concurrency — each size runs all tasks in parallel at that level, both with and without tool calls.
