# AgentCAP Budget & Experiment Plan
## Paper: "Plan Smart, Execute Cheap: Cost-Optimal Hybrid Delegation in Agentic AI"
## Target: NeurIPS 2026
## Budget: £2,000 (~$2,520 USD)

---

## 1. Experiment Overview

### 1.1 Three Benchmarks — Why Each One

#### MCP-Atlas (Tool Orchestration — Balanced)
- Dataset: `ScaleAI/MCP-Atlas` on HuggingFace
- 50 tasks (tool-use, API orchestration)
- Mix of plan-heavy and execute-heavy
- Already integrated in the codebase

#### SWE-bench Pro (Code Modification — Execute-Heavy)
- Dataset: `ScaleAI/SWE-bench_Pro` on HuggingFace (public subset: 731 tasks)
- We'll use 30 tasks from the public subset
- Average: 107 lines changed, 4.1 files modified per task
- WHY: Tests the extreme case where execution dominates. A strong plan (locate the bug, identify the fix strategy) paired with a weak executor (apply multi-file code changes) tests whether plan-execute separation helps for software engineering.
- This is the EXECUTE-HEAVY extreme of the plan-execute spectrum.

#### IMO-AnswerBench (Mathematical Reasoning — Plan-Heavy)
- Dataset: `google-deepmind/superhuman` repo, file `imobench/answerbench_v2.csv`
- 400 IMO-level math problems with verified answers
- Categories: Algebra (100), Combinatorics (100), Geometry (100), Number Theory (100)
- Difficulty: Pre-IMO, IMO-Easy, IMO-Medium, IMO-Hard
- We'll use 30 tasks (stratified sample across categories and difficulties)
- WHY: Tests the extreme case where planning/reasoning IS the task. Execution is trivial (just output the answer). This proves WHEN plan-execute separation doesn't help — pure reasoning tasks don't benefit from splitting.
- This is the PLAN-HEAVY extreme of the plan-execute spectrum.

#### The Three Benchmarks Form a Complete Spectrum
```
Plan-Heavy <-------------------------------------> Execute-Heavy
  IMO-AnswerBench      MCP-Atlas (mixed)       SWE-bench Pro
  (pure reasoning)     (tool orchestration)     (code modification)
```

This allows the paper to answer: "For which types of tasks does plan-execute separation provide the most value?"

### 1.2 Models (via OpenRouter)

| Model | Provider | Input $/1M | Output $/1M | Cache Read $/1M | Role |
|---|---|---|---|---|---|
| MiniMax M2.7 | MiniMax | $0.30 | $1.20 | $0.06 | Budget executor |
| GLM-5 Turbo | Z.ai | $1.20 | $4.00 | $0.24 | Mid-tier |
| DeepSeek V3.2 | DeepSeek | $0.28 | $0.42 | free | Budget executor |
| Kimi K2.5 | Moonshot | $0.60 | $3.00 | TBD | Mid-tier |
| GPT-5.4 | OpenAI | $2.50 | $15.00 | $0.25 | Premium planner |
| Claude 4.6 Opus | Anthropic | $5.00 | $25.00 | $0.50 | Premium planner |
| Gemini 3.1 Pro | Google | $2.00 | $12.00 | $0.20 | Premium planner |

### 1.3 Task Classification: Plan-Heavy vs Execute-Heavy

After running baseline experiments, classify each task by:
- `plan_ratio = plan_output_tokens / (plan_output_tokens + exec_output_tokens)`
- `tool_intensity = exec_tool_calls / exec_output_tokens * 1000`

Categories:
- Plan-Heavy: plan_ratio > 0.4 (reasoning-dominated)
- Execute-Heavy: tool_intensity > 2 AND plan_ratio < 0.2 (tool-call-dominated)
- Balanced: everything else

---

## 2. Experiment Matrix

### 2.1 MCP-Atlas (50 tasks)

#### Self-Self Baselines (7 configs)
Each model as both planner and executor.

| Config | Planner | Executor | Input $/task | Output $/task | Est Cost/task | 50 tasks |
|---|---|---|---|---|---|---|
| minimax-self | M2.7 | M2.7 | 17K x $0.30/1M=$0.005 | 6.5K x $1.20/1M=$0.008 | $0.013 | $0.65 |
| glm5-self | GLM-5 | GLM-5 | 17K x $1.20/1M=$0.020 | 6.5K x $4.00/1M=$0.026 | $0.046 | $2.32 |
| deepseek-self | DS-V3.2 | DS-V3.2 | 17K x $0.28/1M=$0.005 | 6.5K x $0.42/1M=$0.003 | $0.008 | $0.38 |
| kimi-self | K2.5 | K2.5 | 17K x $0.60/1M=$0.010 | 6.5K x $3.00/1M=$0.020 | $0.030 | $1.50 |
| gpt54-self | GPT-5.4 | GPT-5.4 | 17K x $2.50/1M=$0.043 | 6.5K x $15.0/1M=$0.098 | $0.140 | $7.01 |
| claude-self | Claude 4.6 | Claude 4.6 | 17K x $5.00/1M=$0.085 | 6.5K x $25.0/1M=$0.163 | $0.248 | $12.38 |
| gemini-self | Gemini 3.1 | Gemini 3.1 | 17K x $2.00/1M=$0.034 | 6.5K x $12.0/1M=$0.078 | $0.112 | $5.60 |

**Self-self subtotal: $29.84**

#### Execute-Only Baselines (7 configs, no planner)
Same 7 models, experiment_type=execute-only. Similar cost to self-self.
**Execute-only subtotal: ~$29.84**

#### Hybrid Configs -- Premium Planner -> Budget Executor (12 configs)
Plan phase: ~2K input, ~1.5K output (planner)
Exec phase: ~15K input, ~5K output (executor)

| Config | Planner cost | Executor cost | Cost/task | 50 tasks |
|---|---|---|---|---|
| gpt54->minimax | $0.026 | $0.011 | $0.036 | $1.82 |
| gpt54->deepseek | $0.026 | $0.006 | $0.032 | $1.60 |
| gpt54->kimi | $0.026 | $0.024 | $0.050 | $2.50 |
| gpt54->glm5 | $0.026 | $0.038 | $0.064 | $3.18 |
| claude->minimax | $0.048 | $0.011 | $0.058 | $2.91 |
| claude->deepseek | $0.048 | $0.006 | $0.054 | $2.68 |
| claude->kimi | $0.048 | $0.024 | $0.072 | $3.58 |
| claude->glm5 | $0.048 | $0.038 | $0.086 | $4.28 |
| gemini->minimax | $0.036 | $0.011 | $0.047 | $2.36 |
| gemini->deepseek | $0.036 | $0.006 | $0.042 | $2.12 |
| gemini->kimi | $0.036 | $0.024 | $0.060 | $3.00 |
| gemini->glm5 | $0.036 | $0.038 | $0.074 | $3.68 |

**Hybrid subtotal: $33.71**

#### MCP-Atlas Total: ~$93.39

---

### 2.2 SWE-bench Pro (30 tasks)

Token profile per task (much larger than MCP-Atlas):
- Plan phase: ~5K input, ~3K output (reading repo context + planning)
- Exec phase: ~80K input (repo code context), ~15K output (code changes, multi-turn)
- Total: ~85K input, ~18K output per task

NOTE: SWE-bench Pro tasks are 5-10x heavier than MCP-Atlas due to large codebase context.

#### Self-Self Baselines (7 configs x 30 tasks)

| Config | Cost/task | 30 tasks |
|---|---|---|
| minimax-self | $0.047 | $1.42 |
| glm5-self | $0.174 | $5.22 |
| deepseek-self | $0.031 | $0.94 |
| kimi-self | $0.105 | $3.15 |
| gpt54-self | $0.483 | $14.48 |
| claude-self | $0.875 | $26.25 |
| gemini-self | $0.386 | $11.58 |

**Self-self subtotal: $63.04**

#### Execute-Only Baselines (7 configs x 30 tasks): ~$63.04

#### Hybrid Configs (12 configs x 30 tasks)
Plan phase cost similar to MCP-Atlas. Exec phase much heavier.

| Config | Cost/task | 30 tasks |
|---|---|---|
| gpt54->minimax | $0.066 | $1.99 |
| gpt54->deepseek | $0.050 | $1.51 |
| gpt54->kimi | $0.123 | $3.69 |
| gpt54->glm5 | $0.183 | $5.50 |
| claude->minimax | $0.089 | $2.66 |
| claude->deepseek | $0.072 | $2.17 |
| claude->kimi | $0.145 | $4.36 |
| claude->glm5 | $0.205 | $6.16 |
| gemini->minimax | $0.078 | $2.34 |
| gemini->deepseek | $0.062 | $1.85 |
| gemini->kimi | $0.134 | $4.03 |
| gemini->glm5 | $0.195 | $5.84 |

**Hybrid subtotal: $42.10**

#### SWE-bench Pro Total: ~$168.18

---

### 2.3 IMO-AnswerBench (30 tasks)

Token profile per task:
- Plan phase: ~3K input, ~5K output (long reasoning chain)
- Exec phase: ~8K input (plan + problem), ~3K output (shorter execution)
- Total: ~11K input, ~8K output per task
- NOTE: Output-heavy due to long reasoning chains. No tool calls.

#### Self-Self Baselines (7 configs x 30 tasks)

| Config | Cost/task | 30 tasks |
|---|---|---|
| minimax-self | $0.013 | $0.39 |
| glm5-self | $0.045 | $1.36 |
| deepseek-self | $0.006 | $0.19 |
| kimi-self | $0.031 | $0.92 |
| gpt54-self | $0.148 | $4.43 |
| claude-self | $0.255 | $7.65 |
| gemini-self | $0.118 | $3.54 |

**Self-self subtotal: $18.48**

#### Execute-Only Baselines: ~$18.48

#### Hybrid Configs (12 configs x 30 tasks)
For IMO, exec phase is lightweight (just outputting the answer).

| Config | Cost/task (approx) | 30 tasks |
|---|---|---|
| All 12 hybrid configs | $0.02-$0.12 each | ~$18-25 total |

**Hybrid subtotal: ~$22.00**

#### IMO-AnswerBench Total: ~$58.96

---

## 3. Cost Model Updates Needed

### 3.1 Runtime Throughput Measurement (DONE)
- Local cost model now uses `compute_local_cost_runtime()` 
- Measures TTFT (prefill time) and decode time from streaming responses
- `cost = total_per_hour x gpu_seconds / 3600`
- Automatically correct for all concurrency scenarios

### 3.2 Prompt Caching (TODO)
API providers offer 90% discount on cached input tokens (system prompt, tool definitions, conversation history prefix).

In multi-turn execution phases, ~60-70% of input tokens are cached (previous conversation context).

Impact estimate:
- Without caching: API costs as listed above
- With caching: API input costs reduced by ~40-50% overall
- Estimated savings: ~$40-60 across all experiments

Need to add `cache_read_price_per_1m` to `APICostConfig` and track cached vs uncached input tokens.

### 3.3 OpenRouter Pass-Through Pricing
OpenRouter charges the same as the underlying provider for most models. No markup needed in the cost model. Use the prices listed in section 1.2.

---

## 4. Total Budget Summary

| Category | Estimated Cost |
|---|---|
| **MCP-Atlas** (50 tasks x 26 configs) | $93.39 |
| **SWE-bench Pro** (30 tasks x 26 configs) | $168.18 |
| **IMO-AnswerBench** (30 tasks x 26 configs) | $58.96 |
| **Subtotal** | **$320.53** |
| Debug/retry buffer (25%) | $80.13 |
| Smoke tests & iteration (10 tasks x all configs x 2 rounds) | ~$50 |
| Task classification run (1 reference model x all tasks) | ~$15 |
| Scoring/evaluation (GPT-4o-mini for GTFA scoring) | ~$10 |
| **Total Estimated** | **~$476** |
| **Budget Available** | **$2,520** |
| **Remaining Buffer** | **$2,044** |

### Why So Much Buffer?
The estimates above are CONSERVATIVE (assume all tokens are uncached, no failures, max token usage). Real costs will likely be 30-50% lower due to:
1. Prompt caching (40-50% reduction on input costs)
2. Many tasks will use fewer tokens than the max estimate
3. Cheaper models (DeepSeek, MiniMax) dominate the experiment matrix

The £2,000 budget is very comfortable for this experiment plan. Even with 3x cost overrun, we'd still be within budget.

### Budget Risk Scenarios

| Scenario | Total Cost | Within Budget? |
|---|---|---|
| Best case (with caching) | ~$300 | Yes, $2,220 buffer |
| Expected case | ~$476 | Yes, $2,044 buffer |
| Worst case (2x overrun) | ~$952 | Yes, $1,568 buffer |
| Catastrophic (5x, should not happen) | ~$2,380 | Barely, but still within £2,000 |

---

## 5. Execution Timeline (6 weeks)

### Week 1: Infrastructure
- [ ] Integrate SWE-bench Pro dataset loader
- [ ] Integrate IMO-AnswerBench dataset loader
- [ ] Create all YAML configs for 7 models via OpenRouter
- [ ] Add prompt caching support to cost model
- [ ] Smoke test: 3 tasks x 3 configs on each benchmark

### Week 2: MCP-Atlas Experiments
- [ ] Run all 26 MCP-Atlas configs (50 tasks each)
- [ ] Score results with GTFA evaluator
- [ ] Classify tasks as plan-heavy/execute-heavy/balanced

### Week 3: SWE-bench Pro Experiments
- [ ] Run all 26 SWE-bench Pro configs (30 tasks each)
- [ ] Evaluate with run_script.sh + parser
- [ ] Analyze plan-quality vs code-change correlation

### Week 4: IMO-AnswerBench Experiments
- [ ] Run all 26 IMO-AnswerBench configs (30 tasks each)
- [ ] Evaluate with answer matching
- [ ] Analyze: does plan-execute help for pure reasoning?

### Week 5: Analysis & Figures
- [ ] Pareto frontier: Cost vs Accuracy across all configs
- [ ] Plan-heavy vs Execute-heavy breakdown
- [ ] Cross-benchmark comparison
- [ ] Cost model validation: predicted vs actual

### Week 6: Writing
- [ ] Draft paper
- [ ] Generate all figures and tables
- [ ] Internal review and iteration

---

## 6. Collaboration Plan

(To be filled based on team composition)

---

## 7. Key Research Questions

1. **RQ1 (Economic Value)**: How much does plan-execute separation save compared to full-API on each benchmark?
2. **RQ2 (Task Dependency)**: For which task types (plan-heavy, execute-heavy, balanced) does separation help most?
3. **RQ3 (Planner Quality)**: Does a better planner (Claude > GPT > Gemini) consistently improve execution quality across all benchmarks?
4. **RQ4 (Executor Threshold)**: What is the minimum executor capability (DeepSeek < MiniMax < GLM < Kimi) needed for acceptable performance?
5. **RQ5 (Cross-Benchmark)**: Does the optimal planner-executor pairing generalize across MCP-Atlas, SWE-bench Pro, and IMO-AnswerBench?
