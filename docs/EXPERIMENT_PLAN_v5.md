# AgentCAP 实验计划 (v5)

**论文题目**: "Plan Smart, Execute Cheap: Cost-Optimal Hybrid Delegation in Agentic AI" (聪明计划，廉价执行：智能体 AI 中的成本最优混合授权研究)
**目标会议**: NeurIPS 2026
**核心设置**: 基于 **Two-Phase Cline** 架构，利用 Cline CLI 的 `--config` 原生功能，将任务分为 API 辅助的“计划阶段” (Planning) 与本地模型驱动的“执行阶段” (Execution)。

---

## 1. 核心论点 (Thesis)

在智能体 (agentic) 任务中，推理开销主要集中在工具调用与多轮执行中。本研究提出：通过使用高成本 API（如 Claude 4.6 Opus）生成高质量的分步计划，并将其交给低成本的本地模型（如 Qwen3-32B）执行，可以在保持高准确度的同时显著降低总体成本。我们旨在探索 **Planner 质量 vs Executor 能力** 在混合部署中的最优平衡点。

这一架构解决了“大脑”与“肢体”的不对称性：计划需要极高的逻辑推理能力（API 强项），而执行则需要频繁的工具交互与长文本上下文（本地部署强项）。

---

## 2. 主要贡献 (Contributions)

1. **混合成本模型 (Hybrid Cost Model)**: 建立 API 计费 ($/token) 与本地 GPU CapEx+OpEx 统一的成本衡量框架。
2. **Plan-Execute 分离评估**: 首次系统性评估“强 API 计划 + 弱本地执行”在多工具编排任务上的效能。
3. **Executor 能力-成本边界**: 发现本地执行器模型的“最低可用能力阈值”，为企业级 agent 部署提供实践指南。

---

## 3. 核心架构：Two-Phase Cline

废弃之前的 Python Wrapper 方案，改用 Cline CLI 原生的两阶段模式。该方案不需修改 Cline 源码，完全通过 CLI 命令编排实现。

### 3.1 阶段一：计划 (Planning)
使用昂贵的 API 模型配置（如 Claude 4.6 Opus）。Cline 读取 MCP-Atlas 任务描述，分析所需工具和步骤，并输出一个结构化的分步执行计划。
- **配置**: `--config ~/.cline-opus`
- **操作**: 仅生成文档/计划，不执行破坏性工具调用。

### 3.2 阶段二：执行 (Execution)
使用廉价的本地模型配置（如 Qwen3-32B）。Cline 接收阶段一生成的计划作为“指令”，负责具体的工具调用、环境交互及任务最终交付。
- **配置**: `--config ~/.cline-qwen32b`
- **操作**: 全力执行，包含多轮工具调用。

**CLI 调用编排示例**:
```bash
# 步骤 1: 创建独立的配置环境
mkdir -p ~/.cline-opus ~/.cline-qwen32b
cline --config ~/.cline-opus auth anthropic --modelid claude-3-5-sonnet-20241022
cline --config ~/.cline-qwen32b auth openai-compatible --base-url http://localhost:30000/v1 --modelid Qwen/Qwen3-32B

# 步骤 2: 运行 Phase 1 (生成 JSON 格式计划)
PLAN=$(cline --config ~/.cline-opus --json -y "Analyze this task: {task}. Output a specific step-by-step plan.")

# 步骤 3: 运行 Phase 2 (输入计划并执行)
cline --config ~/.cline-qwen32b -y "Execute the following plan: $PLAN"
```

---

## 4. 实验矩阵 (Experiment Matrix)

我们将测试 2 种 Planner × 4 种 Executor，以及对应的全本地对照组，共 11 个配置点。

### 4.1 混合配置 (Plan → Execute)
| # | 配置名称 | Planner | Executor | 核心观察点 |
|---|---|---|---|---|
| 1 | **opus-self** | Claude 4.6 Opus | Claude 4.6 Opus | **Baseline**: 闭源 API 的性能上限 |
| 2 | **opus→gptoss** | Claude 4.6 Opus | GPT-OSS-120B | 强 Plan 对本地超大模型的加持 |
| 3 | **opus→qwen32b** | Claude 4.6 Opus | Qwen3-32B | 典型“强脑弱手”性价比组合 |
| 4 | **opus→qwen4b** | Claude 4.6 Opus | Qwen3-4B | 极限测试：极小执行器是否可用 |
| 5 | **gpt54-self** | GPT-5.4 | GPT-5.4 | **Baseline**: 另一个闭源家族基准 |
| 6 | **gpt54→gptoss** | GPT-5.4 | GPT-OSS-120B | 不同家族 Planner 兼容性 |
| 7 | **gpt54→qwen32b** | GPT-5.4 | Qwen3-32B | 跨家族 Plan → Execute 性能 |
| 8 | **gpt54→qwen4b** | GPT-5.4 | Qwen3-4B | 成本最低的混合路径 |

### 4.2 本地独立基线 (No Planner)
| # | 配置名称 | 模型 | 实验意义 |
|---|---|---|---|
| 9 | **local-gptoss** | GPT-OSS-120B | 本地模型在无 Plan 下的编排能力 |
| 10 | **local-qwen32b** | Qwen3-32B | 中型模型在无 Plan 下的编排能力 |
| 11 | **local-qwen4b** | Qwen3-4B | 小型模型在无 Plan 下的编排能力 |

---

## 5. 研究问题 (Research Questions)

- **RQ1: 分离架构的经济价值**: 相比全 API 模式，Plan-Execute 分离在 MCP-Atlas 任务上平均能节省多少 $/Task？
- **RQ2: Executor 的能力下限**: Executor 模型规模缩小到何种程度（如 27B vs 9B）会导致任务成功率发生“断崖式”下跌？
- **RQ3: Planner 的补偿效应**: 对于同一个本地执行器，来自 Claude 的计划是否显著优于来自 GPT 的计划？优秀的计划能弥补执行器多少比例的逻辑短板？

---

## 6. 指标定义 (Metrics)

| 分类 | 指标 | 定义 | 记录阶段 |
|---|---|---|---|
| **效能** | **accuracy** | 任务最终成功率 (Pass@1, GPT-4o 评分) | Total |
| | **coverage** | claims 覆盖率 (0-1) | Total |
| **成本** | **plan_cost** | 计划阶段产生的 API 实付金额 (USD) | Phase 1 |
| | **exec_cost** | 执行阶段产生的 API 或 本地等效成本 (USD) | Phase 2 |
| | **total_cost** | 任务总成本 (Plan + Exec) | Total |
| **性能** | **plan_latency** | 计划生成耗时 (ms) | Phase 1 |
| | **exec_latency** | 计划执行全过程耗时 (ms) | Phase 2 |
| | **tool_calls** | 执行阶段发起的实际工具调用次数 | Phase 2 |
| **Token** | **plan_tokens** | 计划阶段输入与输出的 Token 总量 | Phase 1 |
| | **exec_tokens** | 执行阶段输入与输出的 Token 总量 | Phase 2 |

---

## 7. 实施计划 (Timeline)

### 阶段 0: 环境准备 (~2小时)
- 安装 Cline CLI 2.0 并验证 headless 运行模式 (`-y`)。
- 配置 5 个模型 config 目录，确保各自的 API Key 和 Base URL 正确。
- 部署 MCP-Atlas Docker 环境并确认 Port 1984 可访问。

### 阶段 1: 本地模型部署与基准测速 (~3小时)
- 在 4× GPUs 上并行部署 Qwen3-32B, Qwen3-4B 和 GPT-OSS-120B。
- **关键**: 实测每个模型的平均生成吞吐量 (tokens/s)，用于计算本地成本。

### 阶段 2: 驱动脚本与数据库设计 (~4小时)
- 编写 `scripts/run_hybrid_experiment.py`：负责读取任务、调用 Cline CLI、捕获 JSON 输出并驱动第二阶段执行。
- 更新 `agent_cap/db/store.py`：新增分阶段统计字段。
- 编写 `agent_cap/cost/hybrid.py`：实现混合成本计算器。

### 阶段 3: 实验运行 (~18小时)
- 运行 API 基线 (Config 1, 5) 与本地基线 (Config 9-11)。
- 运行核心混合实验 (Config 2-4, 6-8)。

### 阶段 4: 结果分析与论文绘图 (~4小时)
- 生成 Pareto 前沿图、成本占比堆叠图、模型规模 vs 准确度曲线图。

---

## 8. 配置文件示例 (YAML)

```yaml
name: opus-qwen32b
experiment_type: "plan-execute"

planner:
  config_path: "~/.cline-opus"
  provider: "anthropic"
  model: "claude-4.6-opus"
  input_usd_1m: 15.0
  output_usd_1m: 75.0

executor:
  config_path: "~/.cline-qwen32b"
  provider: "sglang"
  model: "Qwen3-32B"
  port: 30000

benchmark:
  name: "mcp-atlas"
  tasks: 50
  mcp_server: "http://localhost:1984"

# 本地成本参数
hardware:
  gpu_price: 30000
  life_years: 3
  electricity_kwh: 0.1778
  pue: 1.3
```

---

## 9. 数据库 Schema 变更

`RunResult` 数据类应扩展以下字段以支持分阶段分析：

```python
# Planning 阶段
plan_model: str
plan_in_tokens: int
plan_out_tokens: int
plan_cost: float
plan_latency: float

# Execution 阶段
exec_model: str
exec_in_tokens: int
exec_out_tokens: int
exec_cost: float
exec_latency: float

# 汇总
total_cost: float
tool_call_count: int
success_bool: bool
coverage_score: float
```

---

## 10. 风险与应对 (Risks)

1. **计划解析失败**: 若 Phase 2 的 Executor 无法正确理解 Phase 1 的长计划，需在 Prompt 中强制要求 JSON 或特定的分步列表格式。
2. **API 费用超支**: 每次任务设置硬性 `max_tokens` 限制，并先进行 5 个任务的小规模冒烟测试。
3. **本地模型 OOM**: 根据 VRAM 分布合理配置 Tensor Parallel，必要时切换量化版本。
4. **Pass Rate 极低**: 若所有配置的 Pass@1 均为 0，应侧重分析 Coverage Score 的提升百分比，这在 agentic 论文中同样具备说服力。

---

**备注**: "要好用！不要很复杂"。采用 Cline CLI 原生的两阶段模式，可以最大限度利用现有工具的成熟度，将精力集中在论文核心的 delegation 逻辑验证上。
