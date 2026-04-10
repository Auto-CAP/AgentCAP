# AgentCAP: Cost-Accuracy Pareto Frontiers for Heterogeneous LLM Agent Teams

## 1. Introduction and Motivation

The way we deploy LLM agents is fundamentally wasteful. Today's default is to run a single frontier model for every cognitive function: planning what to do, deciding which tools to call, executing routine API requests, recovering from errors. A $15/M-token model spends 80% of its tokens on mechanical tool calls that a $0.10/M-token model handles equally well. This is the organizational equivalent of having the CEO personally draft every email.

Human organizations solved this problem centuries ago through division of labor. Cognitive science explains why: Kahneman's dual-process theory shows that humans allocate expensive deliberative reasoning (System 2) only to planning and judgment, then switch to cheap automatic processing (System 1) for execution. Vygotsky's zone of proximal development demonstrates that structured guidance — scaffolding — enables less capable agents to perform beyond their independent ability. A detailed plan from a frontier model may be all a 9B-parameter executor needs to accomplish tasks it would fail at alone.

This is no longer theoretical. Anthropic's Claude Code ships an `opusplan` mode — Opus for planning, Sonnet for execution — and practitioners report 60-80% cost reduction at comparable quality. OpenAI Codex, OpenCode, Aider, and Cursor all support similar model-pair configurations. The industry is already building heterogeneous agent teams through intuition and ad-hoc testing.

**The problem: nobody has measured any of this systematically.** No benchmark evaluates heterogeneous model teams. No framework tracks cost alongside accuracy across API and self-hosted deployments. No dataset enables domain-stratified analysis of when delegation helps and when it hurts. Every deployment decision — which model plans, which executes, when to self-host — is made by guesswork.

| Benchmark | Heterogeneous Teams | Cost Tracking | Plan-Execute | API vs Local | Domains |
|---|---|---|---|---|---|
| AgentBench | No | No | No | No | 8 envs |
| MultiAgentBench | Homogeneous only | No | No | No | 5 tasks |
| TheAgentCompany | No | No | No | No | 1 |
| tau-bench | No | No | No | No | 2 |
| MAESTRO | Role-based | No | No | No | 6 |
| HAL | No | No | No | No | 9 |
| **AgentCAP** | **Yes** | **Yes** | **Yes** | **Yes** | **5** |

AgentCAP fills this gap. We evaluate 64 planner-executor combinations (8 models x 8 models, spanning API and self-hosted) across 5 domain-stratified benchmarks (coding, medicine, finance, open-domain, math), tracking cost, accuracy, and latency per task. We introduce an empirical task taxonomy that classifies tasks as plan-critical or execute-critical based on model-swap sensitivity — not human judgment. The result is the first cost-accuracy Pareto frontier for heterogeneous LLM agent teams, with actionable guidance for practitioners on which model does what, when, and at what price.

---

## 2. Research Questions

**RQ1: To what extent can a cheap executor, guided by a frontier planner, close the gap with a frontier model running end-to-end?**
We quantify the plan leverage ratio — the accuracy gain a weak executor obtains from a strong plan — across model sizes, deployment types, and domains. This directly measures whether delegation is a viable cost reduction strategy or merely an accuracy sacrifice.

**RQ2: How does the plan-critical vs. execute-critical nature of a task reshape the optimal team composition?**
We introduce an empirical task taxonomy based on model-swap sensitivity (Section 4.3) and analyze whether these two task categories demand fundamentally different resource allocation between planner and executor. The hypothesis: plan-critical tasks tolerate cheap executors but demand expensive planners, while execute-critical tasks show the reverse — and the boundary between them shifts across domains.

**RQ3: Where on the cost-accuracy Pareto frontier do hybrid teams (API planner + self-hosted executor) sit relative to pure-API and pure-local alternatives?**
We map the full Pareto surface across 64 model combinations and three deployment regimes. The practical question: at what quality threshold and operational scale does each regime dominate, and what is the break-even point for self-hosting the executor?

**RQ4: How does domain structure (coding, medical, finance, open-domain, math) modulate the delegation benefit and the shape of the Pareto frontier?**
We run identical experiments across five domain-stratified benchmarks to isolate domain-specific effects. The hypothesis: domains that require deep sequential reasoning (medicine, math) are plan-critical and see large delegation gains, while domains that require precise multi-step tool use (coding) are execute-critical and see diminishing returns from stronger planners.

---

## 3. Benchmark Design

### 3.1 Datasets

| Dataset | Domain | Tasks | Tool Backend | Key Characteristics |
|---|---|---|---|---|
| MCP-Atlas | Open-domain (10 domains) | 500 | 30+ MCP tool servers | Multi-hop cross-domain queries; 98.2% of tasks span multiple domains |
| MCP-Atlas-Finance | Finance subset | 130 | MCP servers (twelvedata, etc.) | Financial data queries extracted from MCP-Atlas |
| SWE-bench Pro | Software engineering | 731 | Docker sandbox | Real GitHub issues; code editing, testing, debugging |
| MedAgentBench | Clinical medicine | 300 | FHIR API (in-memory mock server) | 100 patients, 700K+ medical records, 10 clinical task categories |
| IMO-AnswerBench | Mathematics | TBD | Minimal (computation) | Competition-level math; pure reasoning with minimal tool use |

**Design rationale.** Three closed-domain benchmarks (coding, medical, math) plus one open-domain benchmark plus one domain subset. All are agentic with tool use except IMO-AnswerBench, which tests pure reasoning to isolate the planning contribution. All are unsaturated: the best current models achieve 30 to 70 percent accuracy, leaving room for improvement and differentiation.

**MCP-Atlas domain distribution.** The 500 tasks span software engineering (26.0%), data analysis (21.0%), knowledge and research (20.8%), geography and travel (16.0%), productivity (5.2%), finance (4.8%), arts and culture (4.0%), biomedical (1.6%), and others. The average task touches 3.9 domains.

### 3.2 Models

**API Models (4):**

| Model | Provider | Input / Output Price (per M tokens) |
|---|---|---|
| Claude 4.6 Opus | Anthropic (direct) | $15 / $75 |
| GPT-5.4 | OpenAI (direct) | $2.50 / $10 |
| MiniMax M2.7 | MiniMax via OpenRouter | ~$1 / $3 |
| Kimi K2.5 | Moonshot via OpenRouter | ~$0.80 / $2 |

**Local Models (4):**

| Model | Size | Hosting | Approx. Cost per Task |
|---|---|---|---|
| Qwen3.5-9B | 9B | 1x H100 | ~$0.02 |
| Qwen3.5-27B | 27B | 1x H100 | ~$0.05 |
| GPT-OSS-20B | 20B | 1x H100 | ~$0.03 |
| GPT-OSS-120B | 120B | 2x H100 | ~$0.15 |

**Design rationale.** Four API models spanning a 20x cost range ($0.80 to $75 per million output tokens). Four local models spanning a 13x size range (9B to 120B parameters). This enables three categories of team: API-API, local-local, and hybrid API-local.

### 3.3 Team Architecture: Plan-Execute

The plan-execute strategy operates in two phases:

**Phase 1 (Planning).** The planner model receives the task instruction and generates a detailed step-by-step plan. No tools are provided. This is a single LLM call.

**Phase 2 (Execution).** The executor model receives the original task instruction concatenated with the plan from Phase 1. It follows the plan using available tools in a multi-turn agentic loop, with a maximum of 8 to 15 turns depending on the dataset.

This is the simplest team architecture and serves as the foundation for systematic evaluation. The framework supports extensible strategies via a `register_strategy()` pattern (critic-loop, router, etc.), but this paper focuses on plan-execute as the first systematic evaluation.

### 3.4 Cost Model

**API cost per request:**

```
cost = uncached_input_tokens * input_price
     + cached_input_tokens * cache_read_price
     + output_tokens * output_price
```

All four API models have prompt caching. Cache read prices are typically 10 to 50 percent of input prices.

**Local cost per request:**

```
cost = gpu_seconds * (capex_per_second + opex_per_second)

capex_per_second = hardware_cost / (useful_life_seconds * utilization)
opex_per_second  = TDP_watts * PUE * electricity_price_per_wh / 3600
```

Reference hardware: NVIDIA H100 80GB SXM at $35,000, 3-year useful life, 700W TDP, PUE of 1.3, electricity at $0.10 per kWh. This yields approximately $0.37 per GPU-hour for capital and $0.09 per GPU-hour for electricity, totaling $0.46 per GPU-hour.

**Derived cost metrics:**

| Metric | Formula | Interpretation |
|---|---|---|
| Cost per Correct Answer (CpCA) | total_cost / num_correct | Normalizes cost by quality |
| Delegation Savings | 1 - Cost(team at threshold) / Cost(single at threshold) | Savings from delegation at equal accuracy |
| Break-Even Volume | hardware_daily_cost / (api_cost_per_task - local_cost_per_task) | Daily tasks needed for self-hosting to beat API |
| Plan Leverage | Acc(S_plan + W_exec) / Acc(W_exec solo) | How much a good plan boosts a weak executor |

---

## 4. Experimental Design

### 4.1 The 64-Combination Matrix

Eight planners multiplied by eight executors yields 64 combinations. This includes 8 self-self baselines along the diagonal (e.g., Claude planning for Claude executing) and 56 cross-model teams.

Each combination is evaluated on 50 randomly sampled tasks per dataset, for a total of 250 tasks per combination across 5 datasets.

**Total experiment size:** 64 combinations x 250 tasks = 16,000 task runs.

### 4.2 Metrics Collected per Task Run

| Metric | Description |
|---|---|
| Accuracy | Binary correct/incorrect, dataset-specific evaluation |
| Plan cost ($) | Planner input and output tokens times price |
| Execution cost ($) | Executor tokens across all turns times price |
| Total cost ($) | Plan cost plus execution cost |
| End-to-end latency (s) | Wall clock from task start to final answer |
| Plan tokens | Input and output token counts for the planner call |
| Execution tokens | Total input and output across all executor turns |
| Tool call count | Number of tool invocations by the executor |
| Executor turns | Number of LLM calls in the agentic loop |

### 4.3 Empirical Task Classification

Instead of human annotation, we classify tasks empirically using model-swap sensitivity analysis. Select a strong model S (e.g., Claude 4.6 Opus) and a weak model W (e.g., Qwen3.5-9B):

```
plan_sensitivity(t)  = Acc(S_plan, W_exec, t) - Acc(W_plan, W_exec, t)
exec_sensitivity(t)  = Acc(W_plan, S_exec, t) - Acc(W_plan, W_exec, t)
```

Classification rule:
- **Plan-critical:** plan_sensitivity > exec_sensitivity
- **Execute-critical:** exec_sensitivity > plan_sensitivity
- **Balanced:** |plan_sensitivity - exec_sensitivity| < epsilon

The three required combinations (WW, SW, WS) are already contained within the 64-combination matrix. No additional runs are needed.

To ensure robustness, we repeat with two to three different S/W pairs (e.g., Claude/Qwen-9B, GPT-5.4/GPT-OSS-20B) and take majority vote across pairs. Tasks where pairs disagree are labeled "ambiguous."

This produces a model-grounded task taxonomy that is:
- **Reproducible:** any researcher can replicate with the same models
- **Empirically justified:** based on actual performance rather than human intuition
- **Granular:** per-task classification, not per-dataset

### 4.4 Analysis Plan

**A. Pareto Frontier Analysis.**
For each dataset, plot cost (log scale, x-axis) versus accuracy (y-axis) for all 64 combinations. Identify Pareto-optimal teams. Color points by deployment type: blue for pure API, green for pure local, red for hybrid. The central question: does the Pareto frontier of teams dominate all single-model baselines?

**B. Plan Leverage Heatmap.**
For each (planner, executor) pair, compute plan_leverage = Acc(planner, executor) / Acc(executor solo). Display as a heatmap with planners on the y-axis and executors on the x-axis. The question: which planners provide the most leverage, and does leverage depend on executor size?

**C. Domain-Stratified Breakdown.**
Repeat analysis A and B for each dataset separately. Compare optimal teams across domains. The question: is the Pareto-optimal team the same for coding, medical, finance, and open-domain tasks?

**D. Task-Type Interaction.**
Within each dataset, separate plan-critical from execute-critical tasks (using the classification from Section 4.3). Draw separate Pareto frontiers for each type. The question: do plan-critical and execute-critical tasks have fundamentally different optimal team compositions?

**E. Scaling Analysis.**
Fix the planner (e.g., Claude 4.6 Opus) and vary executor size across 9B, 20B, 27B, and 120B. Then fix the executor and vary planner capability. The question: at what executor size does performance plateau? Is there a "good enough" threshold?

**F. Break-Even Analysis.**
For each hybrid team that Pareto-dominates pure alternatives, compute the daily request volume at which self-hosting the executor breaks even against pure API deployment. The question: at what operational scale does hybrid deployment become cost-optimal?

---

## 5. Expected Contributions

1. **AgentCAP Benchmark Framework.** The first systematic benchmark for evaluating heterogeneous LLM agent teams across cost, accuracy, and latency dimensions, spanning five domains with both API and self-hosted models.

2. **Empirical Task Taxonomy.** A model-grounded classification of agentic tasks into plan-critical, execute-critical, and balanced categories, based on swap-sensitivity analysis rather than human judgment.

3. **Pareto Frontier Analysis.** Comprehensive cost-accuracy Pareto frontiers for 64 model combinations across 5 domains, quantifying when and by how much delegation helps.

4. **Practical Deployment Guidance.** Actionable recommendations for practitioners on model team selection given their domain, budget, latency constraints, and quality requirements.

---

## 6. Timeline and Budget

### 6.1 API Cost Breakdown for P1

Token assumptions per task: planner receives ~1.5K input and generates ~1K output (single call). Executor runs ~7 turns with growing context, totaling ~35K uncached input + ~21K cached input + ~3.5K output across all turns.

| Role | Model | Estimated Cost per Task |
|---|---|---|
| Planner | Claude 4.6 Opus | $0.098 |
| Planner | GPT-5.4 | $0.014 |
| Planner | MiniMax M2.7 | $0.005 |
| Planner | Kimi K2.5 | $0.003 |
| Executor | Claude 4.6 Opus | $0.82 |
| Executor | GPT-5.4 | $0.13 |
| Executor | MiniMax M2.7 | $0.05 |
| Executor | Kimi K2.5 | $0.04 |
| Any role | Local models | ~$0 (GPU time only) |

Claude as executor dominates cost: 8 combos with Claude executor account for $1,670 of the $2,330 total. The most expensive single combination is Claude-to-Claude at $0.92 per task ($229 for 250 tasks). The cheapest API combination is Kimi-to-Kimi at $0.04 per task ($11 for 250 tasks). All 16 local-to-local combinations cost effectively $0 in API spend.

### 6.2 Schedule

| Phase | Description | Duration | API Cost |
|---|---|---|---|
| P0 | Infrastructure: runners, backends, cost model, datasets | Complete | $0 |
| P1 | Full 64 matrix: 64 combos x 50 samples x 5 datasets = 16,000 runs | 5-7 days | ~$2,300 |
| P2 | Analysis: Pareto frontiers, task classification, heatmaps | 3-4 days | $0 |
| P3 | Paper draft: all sections, figures, tables | 5-7 days | $0 |
| P4 | Validation: top 10 Pareto-optimal combos on full datasets (excl. Claude executor) | 3-5 days | ~$1,500 |
| P5 | Camera-ready: revisions, ablations | 3-5 days | ~$500 |
| **Total** | | **~4 weeks** | **~$4,300** |
