# AgentCAP Experiment Plan (v4)

**Paper**: "When Two Heads Are Cheaper Than One: Cost-Optimal Multi-Agent Combinations on Local GPU Clusters"
**Target**: NeurIPS 2026
**Hardware**: H100, H200, H20, A100, B200 cluster (GPUs sufficient for any model)
**Serving**: SGLang, all local inference, no API calls

---

## Core Thesis

On local GPU clusters, multi-agent combination strategies create a three-dimensional tradeoff between **Accuracy**, **Cost**, and **System Performance**. We provide: (1) a CapEx+OpEx cost model grounded in real hardware economics, (2) systematic evaluation of 9 combination strategies (including 3 novel >2-model strategies) across 5 model families and 2 benchmarks, and (3) evidence that the optimal strategy depends on deployment scenario — cluster utilization, task type (agentic vs non-agentic), and scale.

---

## Contributions

1. **Cost Model for Local GPU Clusters** (Section 3) — A CapEx+OpEx cost model that captures GPU depreciation, electricity, CPU orchestration overhead, and tool execution cost. Unlike API-based $/token pricing, this reflects the real economics of self-hosted inference.

2. **Systematic Multi-Agent Strategy Evaluation** (Section 4-5) — 9 strategies (6 two-model + 3 novel multi-model) tested across 5 model families (Qwen, OpenAI, NVIDIA, DeepSeek, Moonshot) on code generation and agentic benchmarks.

3. **Three-Dimensional Tradeoff Analysis** (Section 6) — Accuracy × Cost × Performance analysis revealing that the optimal strategy shifts based on deployment scenario (idle GPUs → parallel strategies; saturated cluster → cascade; agentic tasks → higher CPU cost share).

---

## Three Evaluation Dimensions

### Dimension 1: Accuracy

| Metric | Definition | Benchmark |
|---|---|---|
| **Pass@1** (primary) | % of tasks solved correctly on first attempt | Both |
| **Coverage Score** | Mean fraction of ground-truth claims covered (0-1) | MCP-Atlas |
| **Oracle Ceiling** | Best accuracy achievable by any strategy (task-level OR) | Both |

### Dimension 2: Cost (CapEx + OpEx)

The cost model decomposes total cost per task into four components:

```
Total_Cost(task) = C_gpu + C_cpu + C_tool + C_idle

Where:
  C_gpu  = Σ_i (gpu_hours_i × (CapEx_rate_i + OpEx_rate_i))     # GPU inference
  C_cpu  = cpu_hours × CPU_cost_rate                              # Orchestration
  C_tool = tool_exec_seconds × CPU_cost_rate                      # Tool execution
  C_idle = idle_gpu_hours × CapEx_rate  (if dedicated allocation)  # GPU idle during tool calls
```

#### CapEx Rate (hardware depreciation)

```
CapEx_rate = GPU_purchase_price / (useful_life_years × 8760 hours/year)
```

| GPU | Purchase Price | Useful Life | CapEx/GPU-hour |
|---|---|---|---|
| A100 80GB | $15,000 | 3 years | $0.57 |
| H100 80GB | $30,000 | 3 years | $1.14 |
| H200 141GB | $40,000 | 3 years | $1.52 |
| H20 96GB | $13,000 | 3 years | $0.50 |
| B200 192GB | $45,000 | 3 years | $1.71 |

#### OpEx Rate (electricity + cooling)

```
OpEx_rate = (GPU_TDP × avg_utilization + CPU_power_share) × PUE × electricity_price / 1000
```

| GPU | TDP | @80% util + 50W CPU | ×1.3 PUE | @$0.10/kWh | OpEx/GPU-hour |
|---|---|---|---|---|---|
| A100 | 400W | 370W | 481W | — | $0.048 |
| H100 | 700W | 610W | 793W | — | $0.079 |
| H200 | 700W | 610W | 793W | — | $0.079 |
| H20 | 400W | 370W | 481W | — | $0.048 |
| B200 | 1000W | 850W | 1105W | — | $0.111 |

#### Key Cost Insight: Strategy Changes Optimal Under Different Scenarios

| Deployment Scenario | How CapEx Counts | Likely Winner |
|---|---|---|
| **Idle GPUs** (spare capacity) | CapEx = sunk cost, marginal cost ≈ OpEx only | Parallel strategies (Vote, Best-of-N) — GPUs are free anyway |
| **Saturated cluster** (all GPUs busy) | CapEx = opportunity cost of displacing other work | Cascade — uses fewest GPU-hours |
| **Cloud rental** | Cost = cloud $/GPU-hour (CapEx+OpEx bundled) | Minimize total GPU-hours |

#### Sensitivity Analysis Parameters

The cost model must be evaluated under varied assumptions:

| Parameter | Default | Sensitivity Range | Why |
|---|---|---|---|
| GPU useful life | 3 years | 2-5 years | Shorter life → higher CapEx → cascade favored |
| Electricity price | $0.10/kWh | $0.05-$0.20 | Higher price → OpEx dominates → power-efficient models favored |
| PUE | 1.3 | 1.1-1.6 | Data center efficiency |
| GPU utilization | 80% | 50%-95% | Affects both OpEx and opportunity cost |
| Cluster utilization | 70% | 30%-100% | Determines if CapEx is sunk or opportunity cost |

### Dimension 3: System Performance

| Metric | Definition | Why |
|---|---|---|
| **Throughput** (primary) | tasks / GPU-hour at given concurrency | Deployment scalability |
| **Latency** | wall-clock seconds per task (end-to-end) | User-facing responsiveness |
| **Scaling Efficiency** | throughput(N) / (N × throughput(1)) | How well strategy parallelizes |
| **CPU Utilization** | % CPU during inference | Bottleneck detector |

#### CPU Bottleneck Experiment

```
X-axis: concurrent requests (batch size: 1, 2, 4, 8, 16, 32, 64)
Y-axis: latency per task (seconds)
Lines:  (a) pure LLM inference (no tools)
        (b) with tool calls (agentic)
        (c) multi-agent cascade (routing + possible escalation)
```

**Expected insight**: Line (b) and (c) flatten at lower batch sizes than (a), indicating CPU orchestration bottleneck. The gap between (a) and (b)/(c) quantifies the "orchestration tax" of agentic multi-agent strategies.

---

## Research Questions

### RQ1: Which multi-agent combinations dominate the accuracy-cost Pareto frontier?

- **Dimensions**: Accuracy × Cost
- **What we test**: 9 strategies × 4+ model pairs × 2 benchmarks
- **Cost model**: Full CapEx+OpEx per task, compared across 3 deployment scenarios
- **Expected finding**: Cascade is Pareto-optimal under saturated-cluster/cloud scenarios; parallel strategies competitive under idle-GPU scenario
- **Key figure**: 3D Pareto frontier (accuracy vs $/task) with deployment scenario overlays

### RQ2: Does cross-family model diversity improve the frontier?

- **Dimensions**: Accuracy × Cost
- **What we test**: Homogeneous strategies (3× same model) vs heterogeneous (3 different families). Is diversity's value in answer aggregation or uncertainty-based routing?
- **Expected finding**: Cross-family disagreement is a stronger (and cheaper) escalation signal than single-model confidence
- **Key figure**: Diversity-Gated Cascade vs Standard Cascade cost-accuracy comparison

### RQ3: How do scaling and task type shift the optimal strategy?

- **Dimensions**: Cost × Performance (with accuracy held constant)
- **What we test**: Throughput scaling curves for each strategy; agentic vs non-agentic cost decomposition; CPU bottleneck detection
- **Expected finding**: (1) Parallel strategies have higher throughput at scale; (2) agentic tasks have higher CPU cost share, shifting the Pareto frontier; (3) tool orchestration creates measurable CPU bottleneck at moderate concurrency
- **Key figure**: Batch size × latency curves (with/without tools) + cost decomposition stacked bars

---

## Strategies (9 Total)

### 2-Model Strategies (6, implemented)

| Strategy | Type | Mechanism | Cost Profile |
|---|---|---|---|
| **Cascade** | Multi-agent | Small first → escalate to large if low confidence | Sequential, low avg cost |
| **Adaptive-Cascade** | Multi-agent | Cascade with dynamic threshold | Sequential, adaptive cost |
| **Vote** | Multi-agent | 3× small model, majority vote | Parallel, 3× small cost |
| **Generate-Verify** | Multi-agent | Small generates, large verifies | Sequential, 1 small + 1 large |
| **Best-of-N** | Single-agent | N× same model, pick best | Parallel, N× cost |
| **Self-Critique** | Single-agent | Model critiques and revises own output | Sequential, 2× cost |

### >2 Model Strategies (3, new — Contribution 2)

| Strategy | Models | Mechanism | Tests (RQ) |
|---|---|---|---|
| **Diversity-Gated Cascade** ⭐ | 2 cheap (different families) + 1 large | Two cheap models run in parallel. Agree → accept. Disagree → escalate to large. | RQ2: Cross-family disagreement as routing signal |
| **Cross-Family Vote** | 3 models (different families) | Three different-family models vote. Compare vs 3× same model at equal cost. | RQ2: Does architectural diversity improve vote accuracy? |
| **3-Tier Cascade** | small + medium + large | Small → medium → large with confidence routing at each tier. | RQ1+RQ3: Do returns plateau after 2 tiers? |

---

## Models

### Kept (data already collected)

| Model | Family | Architecture | Active Params | Status |
|---|---|---|---|---|
| Qwen3-4B | Qwen | Dense Transformer | 4B | ✅ Pair A done, Pair B running |
| Qwen3-30B-A3B | Qwen | MoE | 3B active | ✅ Pair A done, Pair B running |
| Qwen3-32B | Qwen | Dense Transformer | 32B | ✅ Pair B running |

### New (to download)

| Model | Family | Architecture | Active Params | GPU Requirement | Why |
|---|---|---|---|---|---|
| GPT-OSS-20b | OpenAI | MoE | 3.6B active | 1× H100 | Cross-family small model |
| Nemotron-3-Nano | NVIDIA | MoE + Mamba hybrid | 3B active | 1× H100 | Different architecture (linear attention) |
| DeepSeek-V3.2 | DeepSeek | MoE | 37B active | TP=4-8 H100 | Flagship large model, most capable open MoE |
| Kimi-K2.5 | Moonshot | MoE | 32B active | TP=8 H100 | 5th family, trillion-scale MoE |

### Why These Models

- **5 families**: Qwen, OpenAI, NVIDIA, DeepSeek, Moonshot — maximum diversity
- **3 architecture types**: Dense Transformer, MoE, MoE+Mamba — tests architecture sensitivity
- **Small models all ~3-4B active**: Fair cost comparison at similar inference speed
- **Large models are real flagships**: DeepSeek-V3.2 and Kimi-K2.5 are the models people actually deploy — not toy benchmarks

---

## Experiment Design

### Benchmarks

| Benchmark | Type | Tasks | Evaluation | What it tests |
|---|---|---|---|---|
| **BigCodeBench** (ICML 2024) | Non-agentic code generation | 50 tasks | Execution-based (unittest) | Pure reasoning, no tool overhead |
| **MCP-Atlas** (Scale AI, 2025-2026) | Agentic multi-tool orchestration | 50 tasks | Claims-based, GPT-4o judge | Multi-turn tool-use, CPU orchestration cost |

### Phase 1: 2-Model Experiments (existing + cross-family)

| Pair | Small | Large | Key Question | Status |
|---|---|---|---|---|
| **A** | Qwen3-4B | Qwen3-30B-A3B | Baseline: same family, big capability gap | ✅ Done (400 runs) |
| **B** | Qwen3-30B-A3B | Qwen3-32B | Same family, small capability gap | 🔄 Running |
| **C** | GPT-OSS-20b | DeepSeek-V3.2 | Cross-family: OpenAI small → DeepSeek large | ⏳ Next |
| **D** | Nemotron-3-Nano | Kimi-K2.5 | Cross-architecture+family: Mamba → Moonshot MoE | ⏳ Next |

### Phase 2: >2 Model Experiments

| Experiment | Models (families) | GPUs | Key Question |
|---|---|---|---|
| **Diversity-Gated Cascade** | Qwen3-4B + GPT-OSS-20b → DeepSeek-V3.2 (3 families) | 3+ GPUs | Cross-family disagreement as routing signal (RQ2) |
| **Diversity-Gated Cascade** | Qwen3-4B + Nemotron-3-Nano → Kimi-K2.5 (3 families) | 3+ GPUs | Same, with Mamba hybrid (RQ2) |
| **Cross-Family Vote** | Qwen3-4B + GPT-OSS-20b + Nemotron-3-Nano (3 families) | 3 GPUs | Diverse vote vs homogeneous BoN (RQ2) |
| **3-Tier Cascade** | Qwen3-4B → Qwen3-32B → DeepSeek-V3.2 (2 families) | 3+ GPUs | Does 3rd tier add value? (RQ1+RQ3) |

### Phase 3: System Performance Experiments (RQ3)

| Experiment | Setup | Metric |
|---|---|---|
| **Throughput scaling** | Each strategy at batch sizes 1,2,4,8,16,32,64 | Latency per task, GPU util, CPU util |
| **Agentic vs non-agentic** | Same model pair on BigCodeBench vs MCP-Atlas | Cost decomposition (C_gpu, C_cpu, C_tool, C_idle) |
| **CPU bottleneck detection** | Compare throughput saturation point with/without tools | Batch size at which latency stops improving |

---

## Confirmed Results

### Pair A: Qwen3-4B vs Qwen3-30B-A3B × BigCodeBench (400 runs)

| Strategy | Pass@1 | GPU-s/task | $/task (H100) | $/correct |
|---|---|---|---|---|
| Best-of-N-Small (4B×3) | 50% | 104.4 | $0.035 | $0.070 |
| Cascade | 44% | 23.9 | $0.008 | **$0.018** |
| Best-of-N-Large (30B×3) | 44% | 74.7 | $0.025 | $0.057 |
| Adaptive-Cascade | 42% | 41.6 | $0.014 | $0.033 |
| Vote | 36% | 25.3 | $0.008 | $0.023 |
| Generate-Verify | 30% | 25.1 | $0.008 | $0.028 |
| Self-Critique-Small | 24% | 52.4 | $0.017 | $0.073 |
| Self-Critique-Large | 14% | 23.6 | $0.008 | $0.056 |

*$/task calculated at H100 rate: $1.22/GPU-hour = $0.000339/GPU-second*

**Key findings**:
- Cascade: lowest $/correct ($0.018) — Pareto-optimal
- Best-of-N-Small: highest accuracy (50%) — cheap×3 > expensive×1
- Self-critique: consistently harmful (drops accuracy 6-24pp)
- Oracle ceiling: 56% (44% tasks unsolvable by any strategy)

### MCP-Atlas Baselines + Cascade Combo

| Configuration | Pass@1 | Coverage | Notes |
|---|---|---|---|
| Qwen3-30B-A3B single | 0% | 4.4% | Too weak for agentic |
| Qwen3-32B single | 8% | 16.1% | Marginal |
| Cascade combo (30B-A3B → 32B) | 14% | 13.6% | 2× baseline pass rate |

---

## Paper Figures (6 Main + Supplementary)

| # | Figure | RQ | Core Message |
|---|---|---|---|
| 1 | **Pareto Frontier: Accuracy vs $/task** (all pairs, 3 deployment scenarios) | RQ1 | Cascade dominates under saturated/cloud; parallel competitive under idle GPUs |
| 2 | **Cost Decomposition Stacked Bars** (C_gpu + C_cpu + C_tool + C_idle per strategy) | RQ1 | Agentic strategies have higher CPU/tool cost share |
| 3 | **Diversity-Gated vs Standard Cascade** | RQ2 | Cross-family disagreement is a stronger routing signal |
| 4 | **Homogeneous vs Heterogeneous Vote** (budget-matched) | RQ2 | Diversity value: routing > aggregation |
| 5 | **Throughput Scaling: batch size × latency** (with/without tools) | RQ3 | Tool orchestration creates CPU bottleneck at moderate concurrency |
| 6 | **Task Decomposition: Easy / Boundary / Unsolvable** | RQ3 | Narrow boundary zone explains why returns saturate after 2 agents |
| S1 | **3-Tier vs 2-Tier Cascade** | RQ1 | Diminishing returns from finer-grained routing |
| S2 | **Cost Model Sensitivity Analysis** (GPU life, electricity, utilization) | RQ1 | Strategy ranking stability under parameter variation |

---

## Execution Timeline

```
[DONE]  Pair A × BigCodeBench (400 runs, analysis complete)
[DONE]  MCP-Atlas baselines + cascade combo scored
[RUN]   Pair B × BigCodeBench (62/300, GPUs 0+1 on RTX PRO 6000)
[RUN]   Pair B × MCP-Atlas combos (56/250, GPUs 2+3 on RTX PRO 6000)
         |
         v  (~12h)
Phase 3: Pair B analysis + remaining MCP-Atlas strategy scoring
         |
         v
Phase 4: Download new models
   - GPT-OSS-20b (~42GB)
   - Nemotron-3-Nano (~57GB)
   - DeepSeek-V3.2 (multi-GPU, TP=4-8)
   - Kimi-K2.5 (multi-GPU, TP=8)
         |
         v
Phase 5: Cross-family 2-model experiments (Pair C + Pair D)
   Pair C: GPT-OSS-20b vs DeepSeek-V3.2 × both benchmarks
   Pair D: Nemotron-3-Nano vs Kimi-K2.5 × both benchmarks
         |
         v  (~16-24h)
Phase 6: >2 Model experiments
   6a: Diversity-Gated Cascade (4B + GPT-OSS → DeepSeek-V3.2)
   6b: Diversity-Gated Cascade (4B + Nemotron → Kimi-K2.5)
   6c: Cross-Family Vote (4B + GPT-OSS + Nemotron)
   6d: 3-Tier Cascade (4B → 32B → DeepSeek-V3.2)
         |
         v  (~8-12h)
Phase 7: System Performance experiments (RQ3)
   7a: Throughput scaling curves (batch 1-64, each strategy)
   7b: Cost decomposition (C_gpu, C_cpu, C_tool, C_idle)
   7c: CPU bottleneck detection (agentic vs non-agentic)
         |
         v  (~4-8h)
Phase 8: Final analysis
   - Cost model calculations with CapEx+OpEx
   - Sensitivity analysis (GPU life, electricity, utilization)
   - All paper figures
   - 3 deployment scenario comparisons
```

---

## What Gets Dropped (Appendix or Omit)

| Item | Disposition | Reason |
|---|---|---|
| Self-critique detailed analysis | Appendix (1 paragraph + table) | Negative baseline, not core story |
| Qwen3.5 model family | Removed from plan | Too many Qwen variants, replaced by DeepSeek/Kimi |
| Per-pair leaderboard tables | Supplementary | Useful but not paper-body worthy |
| Architecture-specific quirks | Fold into RQ2 diversity analysis | Only if it supports/contradicts diversity thesis |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| DeepSeek-V3.2 requires too many GPUs | Can't run Pair C | Use GPT-OSS-120b (TP=4) as alternative large model |
| Kimi-K2.5 SGLang support issues | Can't run Pair D | Use DeepSeek-V3.2 for both Pair C and D |
| CPU bottleneck not measurable | RQ3 weakened | Focus on cost decomposition instead of throughput curves |
| Cross-family vote inconclusive | RQ2 weakened | Negative result still publishable (diversity doesn't help aggregation, only routing) |
| Cost model parameters disputed by reviewers | Contribution 1 questioned | Sensitivity analysis shows conclusions robust across 2-5yr life, $0.05-0.20/kWh |
| MCP-Atlas pass rates too low | Statistical significance | Use coverage_score (continuous) instead of binary Pass@1 |

---

## Key Implementation Requirements

### Cost Model (new code needed)
- `agent_cap/cost/model.py` — CostModel class with CapEx, OpEx, scenario parameters
- `agent_cap/cost/calculator.py` — Per-task cost calculation from GPU-seconds + CPU-seconds
- Output: $/task for each strategy under each deployment scenario

### >2 Model Strategies (new code needed)
- `combinations.py` — New functions: `diversity_gated_cascade()`, `cross_family_vote()`, `three_tier_cascade()`
- Config: Support `models: [list]` instead of just `small` + `large`

### System Performance Measurement (new code needed)
- `agent_cap/perf/throughput.py` — Concurrent request driver with configurable batch size
- `agent_cap/perf/monitor.py` — CPU utilization tracking during inference
- Output: Throughput curves, CPU util time series, cost decomposition breakdown

### Existing Code Changes
- `store.py` — Add columns: `cpu_seconds`, `tool_exec_seconds`, `gpu_idle_seconds`
- `analyze_results.py` — Add cost model integration, new figure types
