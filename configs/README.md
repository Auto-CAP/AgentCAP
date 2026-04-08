# Experiment Configurations

20 configs across 4 categories for the AgentCAP paper.

## Models

### API Models (via OpenRouter)
| Model | Input $/1M | Output $/1M | Note |
|---|---|---|---|
| Claude 4.6 Opus | $5.00 | $25.00 | Free (API credits) |
| GPT-5.4 | $2.50 | $15.00 | Free (API credits) |
| Gemini 3.1 Pro | $2.00 | $12.00 | |
| Kimi K2.5 | $0.60 | $3.00 | |
| GLM-5 Turbo | $1.20 | $4.00 | |
| DeepSeek V3.2 | $0.28 | $0.42 | |
| MiniMax M2.7 | $0.30 | $1.20 | |

### Local Models (SGLang, 1x or 8x H100)
| Model | Type | Active Params |
|---|---|---|
| Qwen3-32B | Dense | 32B |
| Gemma 4 26B-A4B | MoE | 4B |
| GPT-OSS-120B | MoE | 5.1B |

---

## A. Self-self Baselines (5)

Single model as both planner and executor.

| # | Model | Type |
|---|---|---|
| 1 | Claude 4.6 Opus | API |
| 2 | GPT-5.4 | API |
| 3 | DeepSeek V3.2 | API |
| 4 | Qwen3-32B | Local |
| 5 | Gemma 4 26B-A4B | Local |

---

## B. Pure API Plan-Execute (5)

Both planner and executor are API models.

| # | Planner | Executor | Insight |
|---|---|---|---|
| 6 | Claude 4.6 | DeepSeek V3.2 | Same executor, most expensive planner |
| 7 | GPT-5.4 | DeepSeek V3.2 | Same executor, different planner -> planner quality impact |
| 8 | Claude 4.6 | Kimi K2.5 | Same planner, different executor -> executor quality impact |
| 9 | DeepSeek V3.2 | Claude 4.6 | Reverse of #6 -> which role matters more |
| 10 | Kimi K2.5 | DeepSeek V3.2 | Mid-tier plan + budget exec |

---

## C. Pure Local Plan-Execute (5)

Both planner and executor are self-hosted models.

| # | Planner | Executor | Insight |
|---|---|---|---|
| 11 | Qwen3-32B | Gemma 4 26B-A4B | Dense plan + MoE exec |
| 12 | Gemma 4 26B-A4B | Qwen3-32B | Reverse of #11 -> who is better planner |
| 13 | Qwen3-32B | GPT-OSS-120B | Dense plan + large MoE exec |
| 14 | GPT-OSS-120B | Qwen3-32B | Reverse of #13 -> does large MoE plan well |
| 15 | GPT-OSS-120B | Gemma 4 26B-A4B | MoE plan + MoE exec |

---

## D. Hybrid: API + Local (5)

One model is API, the other is self-hosted.

| # | Planner | Executor | Insight |
|---|---|---|---|
| 16 | Claude 4.6 (API) | Qwen3-32B (Local) | API plan + local dense exec |
| 17 | Claude 4.6 (API) | Gemma 4 26B-A4B (Local) | API plan + local MoE exec |
| 18 | GPT-5.4 (API) | Qwen3-32B (Local) | Different API planner, same local exec |
| 19 | Qwen3-32B (Local) | DeepSeek V3.2 (API) | Local plan + API exec |
| 20 | Gemma 4 26B-A4B (Local) | DeepSeek V3.2 (API) | Different local planner, same API exec |

---

## Key Comparisons

| Comparison | Configs | Question |
|---|---|---|
| Planner quality | 6 vs 7 | Claude vs GPT-5.4 as planner, same DeepSeek executor |
| Executor quality | 6 vs 8 | DeepSeek vs Kimi as executor, same Claude planner |
| Role importance | 6 vs 9 | Claude->DeepSeek vs DeepSeek->Claude: who should plan? |
| Local planner fit | 11 vs 12 | Qwen->Gemma vs Gemma->Qwen: which local model plans better? |
| API vs local exec | 6 vs 16 | Claude->DeepSeek(API) vs Claude->Qwen(Local): API or local executor? |
| Hybrid direction | 16 vs 19 | Claude->Qwen vs Qwen->DeepSeek: API plan+local exec vs local plan+API exec |
| Dense vs MoE exec | 16 vs 17 | Qwen(dense) vs Gemma(MoE) as executor under same Claude planner |
| Full cost spectrum | 1 vs 6 vs 16 | Claude self vs Claude->DeepSeek vs Claude->Qwen: full API vs cheap API vs hybrid |

---

## Running

```bash
# Start MCP server
docker run --rm -d --name mcp-atlas-env -p 1984:1984 \
    --env-file /path/to/.env agent-environment:latest

# Start LLM server (for local models)
python -m sglang.launch_server --model-path Qwen/Qwen3-32B --port 30000

# Run a single config
python -m agent_cap.runner.unified_runner \
    --model-name Qwen/Qwen3-32B \
    --dataset mcp-atlas \
    --base-url http://localhost:30000 \
    --mcp-server-url http://localhost:1984

# Output structure
results/{model_name}/
    metadata_{dataset}_{timestamp}.json
    metrics_{dataset}_{timestamp}.json
    detailed_results_{dataset}_{timestamp}.jsonl
    output_data_{dataset}_{timestamp}.jsonl
```
