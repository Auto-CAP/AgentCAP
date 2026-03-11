# AgentCAP Experiment Plan

**Paper**: "When Two Heads Are Cheaper Than One: Cost-Optimal Multi-Agent Combinations on Local GPU Clusters"
**Target**: NeurIPS 2026

---

## 1. Models

| Model | Type | Total Params | Active Params | BF16 Size | GPU Layout | Status |
|---|---|---|---|---|---|---|
| Qwen3-4B | Dense | 4B | 4B | 7.6 GB | 1 GPU (TP=1) | Cached |
| Qwen3-30B-A3B | MoE (128 experts) | 30B | 3B | 57 GB | 1 GPU (TP=1) | Cached |
| Qwen3-32B | Dense | 32B | 32B | 62 GB | 1 GPU (TP=1) | Cached |
| Qwen3-235B-A22B | MoE | 235B | 22B | ~470 GB BF16 / ~235 GB FP8 | 4 GPU (TP=4, FP8) | Not cached |

### Model Pairs

| Pair | Small | Large | Gap Description | Purpose |
|---|---|---|---|---|
| **A** | Qwen3-4B (4B dense) | Qwen3-30B-A3B (3B active MoE) | Huge gap: 4B vs 3B active, but MoE has 30B total | Dense vs MoE, similar active params |
| **B** | Qwen3-30B-A3B (3B active) | Qwen3-32B (32B dense) | Moderate gap: 3B active vs 32B | MoE vs Dense, ~10x active param difference |
| **C** | Qwen3-4B (4B dense) | Qwen3-32B (32B dense) | Large gap: 4B vs 32B, both dense | Pure scale comparison |
| **D** | Qwen3-32B (32B dense) | Qwen3-235B-A22B (22B active MoE, FP8) | Premium: 32B vs 22B active | Diminishing returns at scale? |

**Priority**: A (done) > B (next) > C (if time) > D (stretch, needs FP8 download)

---

## 2. Benchmarks

| Benchmark | Type | Task Count | Eval Method | Tool Calls? | Status |
|---|---|---|---|---|---|
| **BigCodeBench** (ICML 2024) | Code generation | 1140 total, use 50-100 | Execution-based (unittest) | No | Available |
| **MCP-Atlas** (Scale AI 2025-2026) | Multi-tool orchestration | 691 total, use 50-100 | Claims-based (GPT-4o judge) | Yes (36 MCP servers) | Available |

### Why These Two Benchmarks

- **BigCodeBench**: Non-agentic, single-turn code generation. Tests raw model capability. Well-established.
- **MCP-Atlas**: Agentic, multi-turn tool-use. Tests agent orchestration across 36 real MCP servers. Modern (2025-2026). 
- **The contrast**: Shows how combination strategies behave differently on agentic vs non-agentic tasks.

---

## 3. Combination Strategies

### Multi-Agent Combinations (paper focus)

| Strategy | Flow | Models Used | Key Question |
|---|---|---|---|
| **Cascade** | Small -> eval -> if fail, Large | Both | Does smart routing beat brute force? |
| **Adaptive Cascade** | Small -> self-assess -> if not confident, Large | Both | Can self-assessment replace oracle eval? |
| **Vote** | Both generate in parallel -> pick best | Both | Does model diversity help? |
| **Generate-Verify** | Small generates -> Large verifies -> accept or Large regenerates | Both | Is verification cheaper than generation? |

### Single-Agent Baselines (controls)

| Strategy | Flow | Models Used | Purpose |
|---|---|---|---|
| **Single-Pass** | One model, one shot | Each model separately | Lower bound |
| **Best-of-N (N=3)** | Same model, 3 samples, pick best | Each model separately | Sampling diversity baseline |
| **Self-Critique** | Generate -> critique -> revise | Each model separately | Multi-turn single-agent baseline |

### Strategy Applicability by Benchmark

| Strategy | BigCodeBench | MCP-Atlas |
|---|---|---|
| Single-Pass | Yes | Yes |
| Cascade | Yes | Yes |
| Adaptive Cascade | Yes | Yes |
| Vote | Yes | Yes (pick higher coverage) |
| Generate-Verify | Yes | No (can't verify agent trajectories easily) |
| Best-of-N | Yes | Yes |
| Self-Critique | Yes | No (multi-turn agent state can't be critiqued) |

---

## 4. Metrics

### Paper Metrics (3 primary)

| Metric | Definition | Measures |
|---|---|---|
| **Pass@1** | Fraction of tasks solved correctly | Accuracy |
| **GPU-s/task** | Average GPU-seconds per task | Cost |
| **MCV** | Marginal Cost of Quality = delta_GPU-s / delta_accuracy | Cost-efficiency |

### Agentic-Specific Metrics

| Metric | Definition | Applies To |
|---|---|---|
| **tool_call_count** | Number of tool calls per task | MCP-Atlas only |
| **coverage_score** | Fraction of claims covered (0-1) | MCP-Atlas only |
| **escalation_rate** | Fraction of tasks escalated to large model | Cascade/Adaptive only |

### Pareto Analysis

For each model pair x benchmark, compute:
- Pareto frontier: {(accuracy, GPU-s)} across all strategies
- Identify Pareto-optimal strategies
- Compare frontiers across model pairs

---

## 5. Experiment Phases

### Phase 1: BigCodeBench x Pair A [DONE]

- Models: Qwen3-4B (small) vs Qwen3-30B-A3B (large)
- Strategies: All 8 (cascade, adaptive-cascade, vote, generate-verify, best-of-n-small/large, self-critique-small/large)
- Tasks: 50
- Results: 400 runs in `results/qwen_combo.db`
- Key finding: Cascade is Pareto-optimal (44% @ 23.9 GPU-s), Best-of-N-Small highest accuracy (50% @ 104.4 GPU-s)

### Phase 2: Framework + BigCodeBench x Pair B [NEXT]

**2a. Add tool_call_count to framework**
- Add `tool_call_count` field to `CombinationResult`, `RunResult`, DB schema
- Add `tool_call_count` to StepRecord
- One-line tracking: count `finish_reason == "tool_calls"` in ChatClient responses

**2b. BigCodeBench x Pair B**
- Models: Qwen3-30B-A3B (small) vs Qwen3-32B (large)
- GPU layout: 30B-A3B on GPU 0 (TP=1, 57GB), 32B on GPU 1 (TP=1, 62GB)
- Strategies: All 8
- Tasks: 50
- Expected: Closer accuracy gap -> cascade might not help as much

**2c. Create Pair B config**
```yaml
name: qwen-combo-bigcodebench-pairB
small_model:
  id: "Qwen/Qwen3-30B-A3B"
  arch: MoE
  params_b: 30
  active_b: 3
  tp: 1
  port: 30000
  cuda_visible_devices: "0"
large_model:
  id: "Qwen/Qwen3-32B"
  arch: dense
  params_b: 32
  active_b: 32
  tp: 1
  port: 30001
  cuda_visible_devices: "1"
```

### Phase 3: MCP-Atlas Live Combos [HIGH PRIORITY]

**3a. Integrate MCP-Atlas into AgentCAP**
- Add MCP-Atlas benchmark loader to `benchmarks.py`
- Add MCP-Atlas evaluator to `evaluator.py` (calls GPT-4o judge)
- Or: wrap `mcp_completion_script.py` as external agent runner

**3b. Agentic combination strategies**
For MCP-Atlas, the "agent" is the completion service (port 3000) that does:
  LLM call -> tool execution -> LLM call -> tool execution -> ... -> final answer

Combination strategies for agentic:
- **Single-Pass**: Run agent once per model
- **Cascade**: Run small agent -> evaluate coverage -> if < threshold, run large agent
- **Vote**: Run both agents -> pick higher coverage
- **Best-of-N**: Run same agent N times -> pick highest coverage
- **Adaptive-Cascade**: Run small agent -> check coverage heuristic -> escalate if needed

**3c. Run MCP-Atlas x Pair B**
- Models: Qwen3-30B-A3B (small) vs Qwen3-32B (large)
- Requires: Docker MCP env (port 1984) + SGLang servers + completion service
- Tasks: 50 (same as baselines)
- Track: pass_rate, coverage_score, tool_call_count, GPU-s

### Phase 4: Cross-Benchmark Analysis [PAPER FIGURES]

- Compare Pareto frontiers: BigCodeBench vs MCP-Atlas for same model pair
- Show: "Cascade dominates on code (non-agentic), Vote/Best-of-N better on tool-use (agentic)"
- Generate paper figures:
  - Pareto frontier plot (accuracy vs GPU-s) per benchmark
  - Strategy comparison bar chart (grouped by benchmark)
  - Escalation rate vs accuracy scatter
  - Tool call count distribution (MCP-Atlas only)

### Phase 5: Scale & Robustness [IF TIME]

- Increase to 100 tasks per benchmark (statistical significance)
- Add Pair C (4B vs 32B) for BigCodeBench
- Try Pair D with Qwen3-235B-A22B FP8 (need to download, ~235GB)
- Sensitivity analysis: vary N in Best-of-N (N=2,3,5), vary threshold in Adaptive-Cascade (5,6,7,8)

---

## 6. GPU Resource Plan

4x NVIDIA RTX PRO 6000 (98GB each). User processes on GPU 2-3 (~10GB).

| Phase | GPU 0 | GPU 1 | GPU 2 | GPU 3 | Duration Est |
|---|---|---|---|---|---|
| 2b: BCB Pair B | 30B-A3B (57GB) | 32B (62GB) | User | User | ~3-4 hours |
| 3c: MCP-Atlas Pair B | 30B-A3B (57GB) | 32B (62GB) | User | User | ~4-6 hours |
| 5: Scale to 100 | Same | Same | User | User | ~8 hours |
| 5: 235B FP8 | TP=4 across all 4 GPUs (~59GB each) | - | - | - | Needs user GPUs freed |

---

## 7. Execution Order

```
[DONE] Phase 1: BigCodeBench x Pair A (400 runs)
  |
  v
[NOW]  Phase 2a: Add tool_call_count to framework
  |            (parallel)
  +---> Phase 2b: BigCodeBench x Pair B (while 2a is being coded)
  |
  v
Phase 3a: Integrate MCP-Atlas agent runner
  |
  v
Phase 3b+3c: MCP-Atlas live combos x Pair B
  |
  v
Phase 4: Cross-benchmark analysis & paper figures
  |
  v
Phase 5: Scale & robustness (if time)
```

---

## 8. Expected Paper Story

1. **Cascade is universally Pareto-optimal for non-agentic tasks**: Smart routing (small->eval->large) beats brute-force sampling at same cost
2. **Sampling diversity > model diversity for accuracy ceiling**: Best-of-N-Small achieves highest accuracy, but at disproportionate cost
3. **Self-assessment is unreliable as cascade gate**: Adaptive-cascade wastes cost on self-assessment that doesn't add value
4. **Self-critique hurts**: Confirmed across models — critique-revise cycle degrades quality
5. **Agentic tasks change the calculus**: [Expected from Phase 3] Tool-use tasks may favor different strategies because maintaining agent state matters
6. **Model gap matters**: [Expected from Phase 2b] When models are closer in capability, combination strategies provide diminishing returns
