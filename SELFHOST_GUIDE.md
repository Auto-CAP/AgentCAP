# Self-Hosted Model Experiment Guide

Run AgentCAP experiments with locally deployed models via SGLang.

## Models

| Model | Type | GPUs | SGLang flags |
|---|---|---|---|
| Qwen3-32B | Dense | 1x H100 | `--tp 1` |
| Gemma 4 26B-A4B | MoE (4B active) | 1x H100 | `--tp 1` |
| GPT-OSS-120B | MoE (5.1B active) | 8x H100 | `--tp 8` |

## 1. Start SGLang Server

1 GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --port 30000 \
    --mem-fraction-static 0.85
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path google/gemma-4-31B-it \
    --port 30000 \
    --mem-fraction-static 0.85
```

8 GPU:
```bash
python -m sglang.launch_server \
    --model-path LGAI-EXAONE/EXAONE-Deep-32B \
    --port 30000 \
    --tp 8 \
    --mem-fraction-static 0.85
```

Verify:
```bash
curl http://localhost:30000/v1/models
```

## 2. Start MCP Tool Server (for MCP-Atlas)

```bash
cd mcp-server && bash start.sh
```

Or with Docker:
```bash
docker run --rm -d --name mcp-atlas-env -p 1984:1984 \
    --env-file third_party/mcp-atlas/.env \
    agent-environment:latest
```

## 3. Run MCP-Atlas

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

Note: no `--no-local` and no `--api-key` for self-hosted models. GPU monitoring is enabled by default.

## 4. Run SWE-bench Pro

```bash
python -m agent_cap.runner.unified_runner \
    --model-name Qwen/Qwen3-32B \
    --dataset swe-bench-pro \
    --backend swebench-docker \
    --base-url http://localhost:30000/v1 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data
```

## 5. Run All 3 Models on MCP-Atlas

```bash
# Terminal 1: Qwen3-32B
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path Qwen/Qwen3-32B --port 30000

# Terminal 2: run experiment
python -m agent_cap.runner.unified_runner \
    --model-name Qwen/Qwen3-32B \
    --dataset mcp-atlas --backend mcp \
    --base-url http://localhost:30000/v1 \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data
```

Kill SGLang, start next model:
```bash
# Terminal 1: Gemma 4
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path google/gemma-4-31B-it --port 30000

# Terminal 2: run experiment
python -m agent_cap.runner.unified_runner \
    --model-name google/gemma-4-31B-it \
    --dataset mcp-atlas --backend mcp \
    --base-url http://localhost:30000/v1 \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data
```

Kill SGLang, start next model:
```bash
# Terminal 1: GPT-OSS-120B (8 GPU)
python -m sglang.launch_server --model-path LGAI-EXAONE/EXAONE-Deep-32B --port 30000 --tp 8

# Terminal 2: run experiment
python -m agent_cap.runner.unified_runner \
    --model-name LGAI-EXAONE/EXAONE-Deep-32B \
    --dataset mcp-atlas --backend mcp \
    --base-url http://localhost:30000/v1 \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --output-dir /data/sicheng/agent-team-data
```

## 6. Output

```
/data/sicheng/agent-team-data/{model-name}/
    metadata_{dataset}_{timestamp}.json       # hardware info (GPU name, count)
    metrics_{dataset}_{timestamp}.json        # performance, agentic, quality, hardware
    detailed_results_{dataset}_{timestamp}.jsonl
    output_data_{dataset}_{timestamp}.jsonl
    trajectories_{dataset}_{timestamp}/
        task_000/
            turn_000_request.json
            turn_000_response.json
            turn_000_tool_calls.json
            turn_000_tool_results.json
            ...
            summary.json
```

## 7. Cost Model

For self-hosted models, cost is measured at runtime:

```
cost = total_per_hour × gpu_seconds / 3600
```

- `gpu_seconds` = sum of TTFT + decode time from streaming responses
- `total_per_hour` = capex (GPU depreciation) + opex (electricity)
- Default: H100 SXM = $1.46/hr per GPU

No prompt caching pricing needed — SGLang handles prefix caching automatically, which reduces TTFT and thus gpu_seconds.

## 8. Checking Progress

```bash
# Count completed tasks
wc -l /data/sicheng/agent-team-data/Qwen/Qwen3-32B/output_data_mcp-atlas_*.jsonl

# Check GPU utilization while running
nvidia-smi

# View latest trajectory
ls /data/sicheng/agent-team-data/Qwen/Qwen3-32B/trajectories_mcp-atlas_*/task_*/
```
