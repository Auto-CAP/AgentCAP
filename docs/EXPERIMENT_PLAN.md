# AgentCAP Experiment Plan (v2)

**Paper**: "When Two Heads Are Cheaper Than One: Cost-Optimal Multi-Agent Combinations on Local GPU Clusters"
**Target**: NeurIPS 2026
**Hardware**: 4× NVIDIA RTX PRO 6000 (98GB VRAM each)
**Serving**: SGLang, all local, cost = GPU-seconds

---

## Current Progress (Phase 1-3)

| Experiment | Models | Dataset | Tasks | Status |
|---|---|---|---|---|
| Pair A × BigCodeBench | Qwen3-4B vs Qwen3-30B-A3B | BigCodeBench | 50×8 = 400 | ✅ Done |
| Pair A analysis | — | — | 7 figures | ✅ Done |
| MCP-Atlas baselines | 30B-A3B, 32B single-pass | MCP-Atlas | 50×2 = 100 | ✅ Scored |
| Pair B × BigCodeBench | Qwen3-30B-A3B vs Qwen3-32B | BigCodeBench | 50×6 = 300 | 🔄 53/300 |
| Pair B × MCP-Atlas combos | Qwen3-30B-A3B vs Qwen3-32B | MCP-Atlas | 50×5 = 250 | 🔄 28/250 |

### Pair A Key Findings (已确认)

1. **Cascade 是 Pareto 最优**：44% accuracy @ 23.9 GPU-s/task，cost/correct = 54.3（最低）
2. **Cheap×3 > Expensive×1**：Best-of-N-Small 50% > Single-Large 38%
3. **Self-critique 有害**：从 30% 降到 24%（small），从 38% 降到 14%（large）
4. **小模型自信 = 正确**：Cascade 不 escalate 时 100% 正确，escalate 后仅 17.6%
5. **44% 任务不可解**：Oracle ceiling = 56%

---

## Phase 4: Cross-Family Model Pairs (核心创新)

### 为什么要换模型

只用 Qwen3 会被 reviewer 质疑："这只是 Qwen 的特性，换 DeepSeek/GPT-OSS 还成立吗？"
跨模型家族测试 = **泛化性证明** = 论文更强。

### 可用模型（2025-2026，开源，SGLang 支持）

| Model | Family | Type | Total | Active | BF16 Size | 1 GPU? | Release |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | Qwen3 | Dense | 4B | 4B | 7.6 GB | ✅ | 2025 |
| Qwen3-30B-A3B | Qwen3 | MoE | 30B | 3B | 57 GB | ✅ | 2025 |
| Qwen3-32B | Qwen3 | Dense | 32B | 32B | 62 GB | ✅ | 2025 |
| **Qwen3.5-9B** | Qwen3.5 | Dense (DeltaNet) | 9B | 9B | ~18 GB | ✅ | Feb 2026 |
| **Qwen3.5-35B-A3B** | Qwen3.5 | MoE (DeltaNet) | 35B | 3B | ~70 GB | ✅ | Feb 2026 |
| **GPT-OSS-20b** | OpenAI | MoE | 21B | 3.6B | ~42 GB | ✅ | Aug 2025 |
| **GPT-OSS-120b** | OpenAI | MoE | 117B | 5.1B | ~234 GB | TP=4 or FP8 | Aug 2025 |
| **Nemotron-3-Nano** | NVIDIA | MoE+Mamba hybrid | 30B | 3B | ~57 GB | ✅ | Dec 2025 |
| **Nemotron-3-Super** | NVIDIA | MoE+Mamba hybrid | 120B | 12B | ~240 GB | ❌ 8×H100 | Mar 2026 |
| **Kimi-K2.5** | Moonshot | MoE | 1T | 32B | 太大 | ❌ | Jan 2026 |
| **DeepSeek-V3.2** | DeepSeek | MoE | 671B | 37B | 太大 | ❌ | Dec 2025 |

### 选定的 Model Pairs（能跑 + 有 insight）

| Pair | Small (1 GPU) | Large (1 GPU) | 核心问题 | Priority |
|---|---|---|---|---|
| **A** | Qwen3-4B (4B dense) | Qwen3-30B-A3B (3B MoE) | Baseline: dense vs MoE, 同家族 | ✅ Done |
| **B** | Qwen3-30B-A3B (3B MoE) | Qwen3-32B (32B dense) | 能力接近时 cascade 还有用吗？ | 🔄 Running |
| **C** | GPT-OSS-20b (3.6B MoE) | Qwen3.5-9B (9B dense) | **跨家族**：OpenAI vs Qwen | 🔥 Must |
| **D** | Nemotron-3-Nano (3B MoE+Mamba) | Qwen3.5-9B (9B dense) | **新架构**：Mamba hybrid 做 small 行不行 | 🔥 Must |
| **E** | Qwen3.5-9B (9B dense) | Qwen3.5-35B-A3B (3B MoE) | **最新模型**：dense 做 small vs MoE 做 large | Should |

### 为什么选这些 Pair

- **Pair C** (GPT-OSS vs Qwen3.5): 不同训练方法、不同 tokenizer、不同 MoE 路由。如果 cascade 在这也 work，说明策略是通用的。
- **Pair D** (Nemotron Mamba vs Qwen3.5): Mamba 的线性 attention 推理更快但可能更弱。作为 small model 是完美的 cost-accuracy tradeoff。
- **Pair E** (Qwen3.5 内部): 展示同一架构迭代内，dense-small vs MoE-large 的对比。
- **Pair C+D** 共用同一个 large model (Qwen3.5-9B)，减少实验量。

---

## Phase 5: 完整实验矩阵

### BigCodeBench（非 agentic，代码生成）

| | Pair A (Qwen3) | Pair B (Qwen3) | Pair C (GPT-OSS↔Qwen3.5) | Pair D (Nemotron↔Qwen3.5) | Pair E (Qwen3.5) |
|---|---|---|---|---|---|
| Cascade | ✅ | 🔄 | ⏳ | ⏳ | ⏳ |
| Adaptive-Cascade | ✅ | 🔄 | ⏳ | ⏳ | ⏳ |
| Vote | ✅ | 🔄 | ⏳ | ⏳ | ⏳ |
| Generate-Verify | ✅ | 🔄 | ⏳ | ⏳ | ⏳ |
| Best-of-N | ✅ | 🔄 | ⏳ | ⏳ | ⏳ |
| Self-Critique | ✅ | 🔄 | ⏳ | ⏳ | — |

### MCP-Atlas（agentic，multi-tool）

| | Pair B (Qwen3) | Pair C (GPT-OSS↔Qwen3.5) | Pair D (Nemotron↔Qwen3.5) |
|---|---|---|---|
| Cascade | 🔄 | ⏳ | ⏳ |
| Adaptive-Cascade | 🔄 | ⏳ | ⏳ |
| Vote | 🔄 | ⏳ | ⏳ |
| Best-of-N | 🔄 | ⏳ | ⏳ |

---

## Phase 6: Expected Insights（每个实验该告诉我们什么）

### Insight 1: Cascade 是否跨家族通用？
- **数据来源**: Pair A vs Pair C vs Pair D 的 cascade accuracy 和 GPU-s/correct
- **如果是**: "Cascade 是 architecture-agnostic 的通用最优策略" → 强 contribution
- **如果不是**: 哪些模型特性让 cascade 失效？（MoE routing noise? Mamba 的 recurrence?）
- **图表**: 3 个 pair 的 Pareto frontier 叠在一起

### Insight 2: MoE-small vs Dense-small 谁更适合做 cascade router？
- **数据来源**: Pair C (GPT-OSS-20b MoE small) vs Pair A (Qwen3-4B dense small) 的 escalation rate 和 not-escalated accuracy
- **核心指标**: "小模型自信时的正确率" —— Pair A 是 100%，GPT-OSS 呢？
- **Insight**: MoE 的 sparse activation 是否导致更不可靠的 self-routing？
- **图表**: Escalation analysis 对比图

### Insight 3: Mamba hybrid 在 agentic 任务中的表现
- **数据来源**: Pair D × MCP-Atlas 的 tool_call_count 和 coverage_score
- **核心问题**: Mamba 的线性 attention 在多轮 tool-use 中是否丢失 context？
- **Insight**: 如果 Nemotron 在 BigCodeBench 还行但 MCP-Atlas 很差 → "Mamba 不适合 agentic"
- **图表**: BigCodeBench vs MCP-Atlas accuracy 的 paired comparison

### Insight 4: 能力差距 vs 策略收益
- **数据来源**: 5 个 pair 按 "accuracy gap between small and large single-pass" 排序
- **核心指标**: cascade improvement over single-large（Δaccuracy / GPU-s saved）
- **Insight**: 画出 "能力差距" vs "cascade 收益" 的曲线。是线性的还是有拐点？
- **图表**: X = accuracy gap, Y = cascade cost-efficiency gain. 5 个数据点 + 趋势线

### Insight 5: Agentic vs Non-agentic 策略差异
- **数据来源**: 同一个 pair 在 BigCodeBench vs MCP-Atlas 的最优策略
- **核心问题**: 哪些策略在 agentic 变好？哪些变差？
- **Insight**: 如果 Vote 在 MCP-Atlas 比 Cascade 好 → "agentic 任务需要 diversity 而不是 routing"
- **图表**: Strategy ranking shift 图（两列，连线表示排名变化）

### Insight 6: Self-critique 为什么有害
- **数据来源**: 所有 pair 的 self-critique vs single-pass
- **核心问题**: 是所有模型都有害还是只有弱模型？
- **Insight**: 如果只有弱模型受害 → "self-critique 需要最低能力门槛"
- **图表**: X = single-pass accuracy, Y = self-critique Δaccuracy. 标注哪些正哪些负

---

## Phase 7: Execution Order（按时间排）

```
[DONE]  Phase 1-2a: Pair A + tool_call_count
[DONE]  Figure redesign (strategy-centric)
[RUN]   Phase 2b: Pair B × BigCodeBench (53/300, GPU 0+1)
[RUN]   Phase 3:  Pair B × MCP-Atlas combos (28/250, GPU 2+3)
  |
  v (Phase 2b+3 完成后)
Phase 4a: Pair B analysis + MCP-Atlas GPT-4o scoring
Phase 4b: Cross-benchmark analysis (BigCodeBench vs MCP-Atlas for Pair B)
  |
  v
Phase 5a: Download & cache new models
  - Qwen3.5-9B (~18GB)
  - GPT-OSS-20b (~42GB)
  - Nemotron-3-Nano (~57GB)
  - Qwen3.5-35B-A3B (~70GB)
  |
  v
Phase 5b: Pair C (GPT-OSS-20b vs Qwen3.5-9B) × both benchmarks
  - GPU 0: GPT-OSS-20b (42GB)
  - GPU 1: Qwen3.5-9B (18GB)
  - GPU 2+3: 并行跑第二个 pair
  |
  v (parallel on GPU 2+3)
Phase 5c: Pair D (Nemotron-3-Nano vs Qwen3.5-9B) × both benchmarks
  - GPU 2: Nemotron-3-Nano (57GB)
  - GPU 3: Qwen3.5-9B (18GB, shared with Pair C)
  |
  v
Phase 5d: Pair E (Qwen3.5-9B vs Qwen3.5-35B-A3B) × BigCodeBench only
  |
  v
Phase 6: Final cross-pair analysis (5 pairs × 2 benchmarks)
  - Pareto frontiers overlay
  - Capability gap vs cascade gain curve
  - Strategy ranking shift (agentic vs non-agentic)
  - Self-critique harm analysis
```

---

## GPU 并行计划

每个 pair 需要 2 GPU（small + large）。我们有 4 GPU，可以同时跑 2 个 pair。

| Time Slot | GPU 0 | GPU 1 | GPU 2 | GPU 3 | Duration |
|---|---|---|---|---|---|
| NOW | 30B-A3B (PairB BCB) | 32B (PairB BCB) | 30B-A3B (PairB MCP) | 32B (PairB MCP) | ~12-20h |
| After PairB | GPT-OSS-20b (PairC) | Qwen3.5-9B (PairC) | Nemotron-Nano (PairD) | Qwen3.5-9B (PairD) | ~8-12h |
| After C+D | Qwen3.5-9B (PairE) | Qwen3.5-35B-A3B (PairE) | — | — | ~6h |

**Total estimated GPU time**: ~30-40h after current experiments finish.

---

## Paper Figures Checklist（最终需要的图）

| # | Figure | Data Source | 核心 Message |
|---|---|---|---|
| 1 | **Pareto Frontier Overlay** (all pairs) | 5 pairs × BigCodeBench | Cascade 在所有 pair 都是 Pareto 最优 |
| 2 | **Strategy Ranking Shift** | Pair B+C+D × 2 benchmarks | Agentic 改变了最优策略 |
| 3 | **Capability Gap vs Cascade Gain** | 5 pairs 的 gap 和 gain | 存在 sweet spot |
| 4 | **Escalation Analysis** (cross-pair) | Cascade 的 escalation rate | 不同模型的 routing 可靠性 |
| 5 | **Cost per Correct** (grouped by pair) | 5 pairs | Cascade 一致最低 |
| 6 | **Tool Call Distribution** | MCP-Atlas all pairs | Agentic 的 cost 构成不同 |
| 7 | **Self-Critique Harm** | All pairs | Δaccuracy vs base accuracy |
| 8 | **Task Difficulty Heatmap** | Pair A (best data) | 策略互补性 |

---

## 风险和备选方案

| Risk | Mitigation |
|---|---|
| GPT-OSS-20b 不支持 SGLang tool calling | 用 vLLM 或只跑 BigCodeBench |
| Nemotron Mamba 不支持 SGLang | 已确认 SGLang Day-0 支持 Nano |
| Qwen3.5-35B-A3B BF16 70GB 超单 GPU | 用 `--mem-fraction-static 0.9` 或 FP8 量化 |
| MCP-Atlas 对新模型 pass rate 太低 | 用 coverage_score 代替 pass@1 做连续指标 |
| 跨家族 tokenizer 导致 prompt 长度不一致 | 标准化为 token count 而不是 character count |
