# Experiment Start Guide

How to run AgentCAP experiments on MCP-Atlas and SWE-bench Pro.

## Prerequisites

- Python 3.11+
- `uv` (for MCP server setup)
- Docker (for SWE-bench Pro)
- API keys for LLM providers

## 1. MCP-Atlas

### 1.1 Start MCP Tool Server

Option A: Docker (recommended, stable)
```bash
docker run --rm -d --name mcp-atlas-env -p 1984:1984 \
    --env-file third_party/mcp-atlas/.env \
    agent-environment:latest
```

Option B: Without Docker
```bash
cd mcp-server && bash start.sh
```

First run will prompt you to fill API keys in `third_party/mcp-atlas/.env`.

Verify:
```bash
curl http://localhost:1984/health
# Should return: {"status":"health_and_client_connection_ok"}
```

### 1.2 Run Experiment

```bash
python -m agent_cap.runner.unified_runner \
    --model-name gpt-5.4 \
    --dataset mcp-atlas \
    --backend mcp \
    --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data \
    --no-local
```

### 1.3 All 6 API Models

GPT-5.4 (direct OpenAI):
```bash
python -m agent_cap.runner.unified_runner \
    --model-name gpt-5.4 \
    --dataset mcp-atlas \
    --backend mcp \
    --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data \
    --no-local
```

Claude 4.6 Opus (direct Anthropic):
```bash
python -m agent_cap.runner.unified_runner \
    --model-name claude-4.6-opus \
    --dataset mcp-atlas \
    --backend mcp \
    --base-url https://api.anthropic.com/v1 \
    --api-key $ANTHROPIC_API_KEY \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data \
    --no-local
```

DeepSeek V3.2 (direct DeepSeek):
```bash
python -m agent_cap.runner.unified_runner \
    --model-name deepseek-reasoner \
    --dataset mcp-atlas \
    --backend mcp \
    --base-url https://api.deepseek.com/v1 \
    --api-key $DEEPSEEK_API_KEY \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data \
    --no-local
```

Kimi K2.5 (OpenRouter, pin to Moonshot AI):
```bash
python -m agent_cap.runner.unified_runner \
    --model-name moonshotai/kimi-k2.5 \
    --dataset mcp-atlas \
    --backend mcp \
    --base-url https://openrouter.ai/api/v1 \
    --api-key $OPENROUTER_API_KEY \
    --openrouter-provider "Moonshot AI" \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data \
    --no-local
```

GLM-5 Turbo (OpenRouter, pin to Z.AI):
```bash
python -m agent_cap.runner.unified_runner \
    --model-name z-ai/glm-5-turbo \
    --dataset mcp-atlas \
    --backend mcp \
    --base-url https://openrouter.ai/api/v1 \
    --api-key $OPENROUTER_API_KEY \
    --openrouter-provider "Z.AI" \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data \
    --no-local
```

MiniMax M2.7 (OpenRouter, pin to Minimax):
```bash
python -m agent_cap.runner.unified_runner \
    --model-name minimax/minimax-m2.7 \
    --dataset mcp-atlas \
    --backend mcp \
    --base-url https://openrouter.ai/api/v1 \
    --api-key $OPENROUTER_API_KEY \
    --openrouter-provider "Minimax" \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data \
    --no-local
```

### 1.4 Local Models (SGLang)

Start SGLang server first:
```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-32B --port 30000
```

Then run:
```bash
python -m agent_cap.runner.unified_runner \
    --model-name Qwen/Qwen3-32B \
    --dataset mcp-atlas \
    --backend mcp \
    --base-url http://localhost:30000/v1 \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data
```

Note: no `--no-local` flag for local models (enables GPU monitoring).

## 2. SWE-bench Pro

### 2.1 With Docker

No separate server needed. Each task pulls its own Docker image.

```bash
python -m agent_cap.runner.unified_runner \
    --model-name gpt-5.4 \
    --dataset swe-bench-pro \
    --backend swebench-docker \
    --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data \
    --no-local
```

### 2.2 With Modal (no Docker needed)

Requires Modal account: `pip install modal && modal setup`

```bash
python -m agent_cap.runner.unified_runner \
    --model-name gpt-5.4 \
    --dataset swe-bench-pro \
    --backend swebench-modal \
    --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data \
    --no-local
```

### 2.3 SWE-bench Tools

SWE-bench provides 4 tools to the agent:
- `read_file` — read a file in the repo
- `write_file` — write content to a file
- `run_shell` — execute a shell command
- `search_code` — grep for a pattern

The agent reads code, understands the issue, and makes fixes using these tools.

## 3. Output Structure

All experiments produce the same output format:

```
/data/sicheng/agent-team-data/{model-name}/
    metadata_{dataset}_{timestamp}.json
    metrics_{dataset}_{timestamp}.json
    detailed_results_{dataset}_{timestamp}.jsonl
    output_data_{dataset}_{timestamp}.jsonl
    trajectories_{dataset}_{timestamp}/
        task_000/
            turn_000_request.json
            turn_000_response.json
            turn_000_tool_calls.json
            turn_000_tool_results.json
            turn_001_request.json
            ...
            summary.json
        task_001/
            ...
```

## 4. API Keys Summary

| Variable | Used by | How to get |
|---|---|---|
| OPENAI_API_KEY | GPT-5.4 | api.openai.com |
| ANTHROPIC_API_KEY | Claude 4.6 | console.anthropic.com |
| DEEPSEEK_API_KEY | DeepSeek V3.2 | platform.deepseek.com |
| OPENROUTER_API_KEY | Kimi, GLM-5, MiniMax | openrouter.ai |
| MOONSHOT_API_KEY | Kimi (direct, optional) | platform.moonshot.cn |

## 5. Provider Pinning (OpenRouter)

When using OpenRouter, always pin the provider to ensure prompt caching works:

| Model | Provider | Flag |
|---|---|---|
| Kimi K2.5 | Moonshot AI | `--openrouter-provider "Moonshot AI"` |
| GLM-5 Turbo | Z.AI | `--openrouter-provider "Z.AI"` |
| MiniMax M2.7 | Minimax | `--openrouter-provider "Minimax"` |

Without pinning, OpenRouter may route to different providers each request, breaking prompt cache.

## 6. Cost Tracking

The cost model accounts for prompt caching automatically. Cache prices:

| Model | Input $/1M | Cache $/1M | Output $/1M |
|---|---|---|---|
| GPT-5.4 | $2.50 | $0.25 | $15.00 |
| Claude 4.6 Opus | $5.00 | $0.50 | $25.00 |
| GLM-5 Turbo | $1.20 | $0.24 | $4.00 |
| Kimi K2.5 | $0.56 | $0.10 | $2.94 |
| DeepSeek V3.2 | $0.28 | $0.028 | $0.42 |
| MiniMax M2.7 | $0.30 | $0.06 | $1.20 |

Cached tokens are read from `usage.prompt_tokens_details.cached_tokens` in API responses.

For local models, cost = total_per_hour * gpu_seconds / 3600 (measured at runtime).

## 7. Checking Progress

While experiments run in tmux:
```bash
# Count completed tasks
wc -l /data/sicheng/agent-team-data/{model}/output_data_{dataset}_*.jsonl

# Check latest result
tail -1 /data/sicheng/agent-team-data/{model}/output_data_{dataset}_*.jsonl | python3 -m json.tool

# Check metrics so far (after experiment completes)
python3 -m json.tool /data/sicheng/agent-team-data/{model}/metrics_{dataset}_*.json
```
