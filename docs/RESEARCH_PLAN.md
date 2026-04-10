# AgentCAP: Cost-Accuracy Pareto Frontiers for Heterogeneous LLM Agent Teams

## 1. Background

### 1.1 The Single-Model Default

Current LLM agent deployments use one model for everything: planning, reasoning, tool calling, and error recovery. This is the default because it is simple. But it is inefficient. A frontier model priced at $15 per million tokens spends most of its tokens on routine tool calls that a model at $0.10 per million tokens could handle equally well. The analogy is straightforward: no organization has the CEO write every email, yet the dominant LLM agent paradigm does exactly this.

### 1.2 Cognitive Science Foundations

Three frameworks from cognitive science predict that separating planning from execution should be optimal.

**Dual Process Theory (Kahneman, 2011).** Human cognition operates two systems. System 2 is slow, deliberative, and resource-intensive; it handles planning and complex reasoning. System 1 is fast, automatic, and cheap; it handles routine execution. Crucially, humans do not run System 2 continuously. They plan with System 2, then switch to System 1 for execution. This is not laziness; it is optimal resource allocation. The mapping to LLMs is direct: expensive frontier models serve as System 2 (the planner), and cheap models serve as System 1 (the executor).

**Zone of Proximal Development (Vygotsky, 1978).** A weaker agent can accomplish tasks beyond its solo capability when given structured guidance, which Vygotsky called scaffolding. In our framework, the plan IS the scaffolding. A 9-billion-parameter model that succeeds at 30% accuracy when working alone may achieve 70% or higher when following a detailed step-by-step plan from a frontier model. The plan does not reduce the difficulty of the task. It expands the executor's effective capability boundary, moving tasks from outside to inside the executor's zone of proximal development.

**Division of Cognitive Labor (Simon, 1947; Smith, 1776).** Herbert Simon's concept of bounded rationality holds that any single agent has limited cognitive resources. Organizations exist to overcome individual cognitive bottlenecks through specialization. Adam Smith's pin factory demonstrated that 18 specialists each performing one step outperform 18 generalists each making whole pins, achieving a 240-fold improvement. In LLM terms: you should not use a $15-per-million-token model to call `read_file`, but you should use it to decide WHICH file to read.

### 1.3 Industry Practice

This is not hypothetical. Major AI providers are already operationalizing heterogeneous model teams in production:

- **Anthropic Claude Code** offers an `opusplan` mode that uses Opus (frontier) for planning and automatically delegates execution to Sonnet (efficient). Community practitioners report 60 to 80 percent cost reduction with comparable quality. One practitioner noted: "the biggest win wasn't even cost, it was context window management. Opus burns through 80% of the window just reading files."
- **OpenAI Codex** uses a frontier model for orchestration with smaller models for sub-tasks.
- **Open-source agent tools** such as OpenCode, Aider, and Cursor support configurable planner and executor model pairs.

Yet these systems are designed through intuition and ad-hoc testing. No systematic benchmark exists to evaluate when delegation helps, by how much, and at what cost.

### 1.4 Limitations of Existing Benchmarks

| Benchmark | Multi-Model Teams | Cost Tracking | Plan-Execute Separation | Cross-Deployment (API vs Local) | Domain Coverage |
|---|---|---|---|---|---|
| AgentBench | No | No | No | No | 8 environments |
| MultiAgentBench | Yes (homogeneous) | No | No | No | 5 tasks |
| TheAgentCompany | No | No | No | No | Simulated company |
| tau-bench | No | No | No | No | Retail + airline |
| MAESTRO | Yes (roles) | No | No | No | 6 categories |
| HAL | No | No | No | No | 9 benchmarks |
| **AgentCAP (ours)** | **Yes (heterogeneous)** | **Yes** | **Yes** | **Yes** | **5 domains** |

No existing benchmark evaluates all four dimensions that matter for practical deployment: heterogeneous model teams, cost tracking, plan-execute separation, and cross-deployment comparison.

---

## 2. Research Questions

**RQ1: When does delegation beat single-model?**
Under what conditions does a planner-plus-executor team Pareto-dominate the best single model at the same cost? We hypothesize that delegation is beneficial when the task has separable planning and execution phases and when execution does not require the same level of reasoning as planning.

**RQ2: Is the optimal team task-dependent?**
Do plan-critical and execute-critical tasks demand different team configurations? We introduce an empirical task taxonomy (Section 4.3) based on model-swap sensitivity analysis and hypothesize that plan-critical tasks benefit from expensive planners paired with cheap executors, while execute-critical tasks show the opposite pattern.

**RQ3: Does hybrid deployment dominate pure deployments?**
Can cross-deployment teams (API planner plus self-hosted executor) outperform both pure-API and pure-local teams on the cost-accuracy Pareto frontier? This is the most practically actionable question: if hybrid dominates, practitioners should deploy this way.

**RQ4: Does the optimal team generalize across domains?**
Is the best planner-executor pair for coding also the best for medical and finance tasks? Our five domain-stratified datasets enable domain-specific analysis. We hypothesize that coding tasks are more execute-critical (requiring precise tool use) while medical tasks are more plan-critical (requiring diagnostic reasoning chains).

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

| Phase | Description | Duration | Estimated API Cost |
|---|---|---|---|
| P0 | Infrastructure: runners, backends, cost model, dataset loaders | Complete | $0 |
| P1 | Full 64-combination matrix: 64 x 50 samples x 5 datasets = 16,000 runs | 5-7 days | ~$6,000 |
| P2 | Analysis: Pareto frontiers, task classification, heatmaps, scaling curves | 3-4 days | $0 |
| P3 | Paper draft: Sections 1 through 6, all figures and tables | 5-7 days | $0 |
| P4 | Validation: top 15 combinations on full datasets | 3-5 days | ~$4,000 |
| P5 | Camera-ready: revisions, ablation studies, appendices | 3-5 days | ~$1,000 |
| **Total** | | **Approximately 4 weeks** | **Approximately $11,000** |
