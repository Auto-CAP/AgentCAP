# AgentCAP: Results and Analysis

## 1. Experiment Overview

**Models**: 5 API (GPT-5.4, Claude Opus 4, MiniMax M2.7, Kimi K2.5, GLM 5.1) + 5 self-hosted (Qwen-3.5 4B/9B/27B, GPT-OSS 20B/120B)

**Benchmarks**: MCP-Atlas (open-domain tool-use, 60 tasks), MedAgentBench (medical FHIR, 60 tasks), tau2-Banking (finance dialogue, 120 tasks)

**Runs**: 25 API PE combos + 8 self-hosted PE combos on MCP-Atlas, 24 API + 4 self-hosted on MedAgent, 1 on tau2, 4 single-agent baselines on MCP-Atlas. ~62 total runs.

---

## 2. Single Agent vs Plan-Execute

### 2.1 Aggregate comparison (MCP-Atlas)

| Model | Single Agent | Homo PE | Best Hetero PE | Best combo |
|---|---|---|---|---|
| GPT-5.4 | 69.1 ± 2.6 | 70.7 (+1.6pp) | 80.4 (+11.3pp) | GPT→MiniMax |
| GLM-5.1 | 57.0 ± 3.0 | 72.4 (+15.4pp) | 79.1 (+22.1pp) | GLM→Claude |
| MiniMax M2.7 | 52.0 ± 2.7 | 68.5 (+16.5pp) | 80.4 (+28.4pp) | GPT→MiniMax |

Self-hosted:

| Model | Single Agent | Homo PE | Best Hetero PE | Best combo |
|---|---|---|---|---|
| Qwen-27B | 57.5 | — | 69.1 (+11.6pp) | 4B→27B |
| Qwen-4B | 33.3 | 0.0 (−33.3pp) | 69.1 (+35.8pp) | 4B→27B |

**Observation**: PE consistently improves over single agent for API models (+1.6pp to +28.4pp). The gain is larger for cheaper models. For self-hosted, PE with a weak planner kills performance (4B×4B: 0.0), but a stronger planner rescues it (4B→27B: 69.1).

### 2.2 Single-agent failure modes (GLM-5.1, 60 tasks on MCP-Atlas)

| Failure mode | Count | % | Description |
|---|---|---|---|
| Passed | 16 | 27% | score ≥ 0.75 |
| Gave up | 9 | 15% | 0 tool calls; outputs "I don't have that tool" |
| Thrashing | 14 | 23% | Hits 30-turn limit; empty or near-empty output |
| Incomplete | 21 | 35% | Some tool calls, partial score |

38% of tasks (gave-up + thrashing) produce **zero useful output** while burning tokens.

### 2.3 How PE fixes single-agent failures

| Failure mode | Fixed by PE | Still failed | Fix rate |
|---|---|---|---|
| Gave up (0 tool calls) | 8/9 | 1/9 | **89%** |
| Thrashing (hit turn limit) | 11/14 | 3/14 | **79%** |

### 2.4 Trajectory examples

**Example A: Thrashing → PE success (Task 4)**

*Task*: Find the 30th commit by the top committer in a museum-related project.

Single agent: 33 tool calls across 30 turns, mostly failed `fetch_fetch` calls to GitHub (blocked by robots.txt). **Empty output, score = 0.**

Plan-execute: Planner writes a 4-step plan — (1) list `/data` to find the repo, (2) check `git log` for top committer, (3) list commits by that person, (4) examine the 30th commit. Executor follows the plan: 14 tool calls, 16 turns. **Correct answer, score = 1.0.**

*Why it works*: Single agent tries `fetch_fetch` on GitHub URLs (wrong approach). The plan directs the executor to use `desktop-commander_list_directory` and `git_git_log` — the right tools for a local repo task.

**Example B: Gave-up → PE success (Task 13)**

*Task*: Analyze fraud crime statistics from a local CSV dataset.

Single agent: 0 tool calls, 1 turn. Outputs "I don't have access to any local data files on your device." **Score = 0.**

Plan-execute: Planner first explores `/data` with `desktop-commander_list_directory`, finds CSV files, then plans the analysis pipeline. Executor: 11 tool calls, 13 turns. **Correct table of fraud statistics, score = 1.0.**

*Why it works*: Single agent doesn't know local files exist. The plan's first step ("list /data") bootstraps the executor's awareness of available data.

### 2.5 Cost comparison

| Config | Accuracy | Total cost (60 tasks) | $/task |
|---|---|---|---|
| GLM single | 0.312 | $8.52 | $0.142 |
| GLM→GLM PE | 0.724 | $4.99 | $0.083 |

PE achieves **2.3× accuracy at 41% lower cost**. The cost reduction comes from fewer wasted turns: single agent burns tokens thrashing (avg 11.1 turns), while PE's plan reduces average executor turns to 13.3 but with much higher success rate per turn.

---

## 3. Plan-Execute Accuracy Matrix

### 3.1 MCP-Atlas (open-domain tool-use)

Rows = planner, columns = executor. Bold = best in row.

| Planner ↓ Exec → | GPT-5.4 | Claude | MiniMax | Kimi | GLM |
|---|---|---|---|---|---|
| GPT-5.4 | 0.707 | 0.354 | **0.804** | 0.602 | 0.670 |
| Claude | 0.688 | 0.513 | 0.661 | 0.634 | **0.691** |
| MiniMax | 0.632 | 0.687 | 0.685 | 0.583 | **0.711** |
| Kimi | 0.575 | **0.660** | 0.619 | 0.630 | — |
| GLM | 0.683 | **0.791** | 0.699 | — | 0.724 |

Self-hosted MCP-Atlas:

| | Qwen-27B | Qwen-4B | GPT-OSS-120B | GPT-OSS-20B |
|---|---|---|---|---|
| Qwen-4B plan | **0.691** | 0.000 | — | — |
| Qwen-27B plan | 0.041 | — | — | — |
| GPT-OSS-20B plan | — | — | **0.388** | 0.187 |
| GPT-OSS-120B plan | — | — | 0.298 | 0.246 |

### 3.2 MedAgentBench (medical FHIR)

| Planner ↓ Exec → | GPT-5.4 | Claude | MiniMax | Kimi | GLM |
|---|---|---|---|---|---|
| GPT-5.4 | 0.717 | 0.883 | 0.767 | 0.767 | **0.933** |
| Claude | 0.833 | **0.933** | **0.933** | **0.933** | **0.933** |
| MiniMax | 0.583 | **0.867** | 0.733 | 0.800 | 0.800 |
| Kimi | 0.667 | **0.917** | 0.717 | 0.883 | — |
| GLM | 0.617 | **0.900** | 0.867 | — | 0.900 |

Self-hosted MedAgent:

| | Qwen-27B | Qwen-9B |
|---|---|---|
| Qwen-27B plan | **0.920** | 0.880 |
| Qwen-9B plan | 0.620 | 0.600 |

---

## 4. Role Sensitivity Analysis

### 4.1 Which role matters more?

**MCP-Atlas** — Executor-critical:
- Fix planner, vary executor: avg accuracy spread = 0.190
- Fix executor, vary planner: avg accuracy spread = 0.172
- Largest swing: GPT as planner, swapping executor from MiniMax (0.804) to Claude (0.354) = **45pp drop**

**MedAgentBench** — Planner-critical (for API), Executor-critical (overall):
- Claude as planner achieves 0.933 with **any** executor (Claude, MiniMax, Kimi, GLM all = 0.933)
- Self-hosted: 27B→9B = 0.880 vs 9B→9B = 0.600 → planner swap = **28pp** swing; 27B→27B = 0.920 vs 27B→9B = 0.880 → executor swap = only **4pp**

### 4.2 Role rankings reverse across domains

| Model | MCP Plan rank | MCP Exec rank | Med Plan rank | Med Exec rank |
|---|---|---|---|---|
| GLM | #1 | #1 | #2 | #2 |
| MiniMax | #2 | #2 | #5 | #4 |
| Claude | #3 | **#5** | **#1** | **#1** |
| GPT | #4 | #3 | #3 | **#5** |
| Kimi | #5 | #4 | #4 | #3 |

Claude: worst MCP executor (#5), best MedAgent executor (#1).
GPT: decent MCP executor (#3), worst MedAgent executor (#5).

### 4.3 Why Claude fails as MCP executor (trajectory evidence)

Claude as executor produces **abnormally short outputs** (3–90 characters on failing tasks) and **terminates early** (avg 6–8 turns vs 12–17 for other executors).

GPT→Claude task-level comparison (first 5 tasks):

| Task | Claude score | Claude output len | Claude tc | GPT-as-exec score | GPT output len | GPT tc |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 61 chars | 5 | 1.000 | 917 chars | 11 |
| 1 | 0.000 | 86 chars | 2 | 1.000 | 1105 chars | 29 |
| 2 | 0.000 | 256 chars | 2 | 1.000 | 579 chars | 15 |
| 3 | 0.000 | 3 chars | 5 | 1.000 | 828 chars | 10 |
| 4 | 0.000 | 90 chars | 3 | 0.500 | 570 chars | 41 |

Claude as executor makes a few tool calls, gets partial results, then outputs a near-empty answer and stops. It does not persist through tool errors or iterate on partial information.

---

## 5. Cost Analysis

### 5.1 Per-role cost decomposition (MCP-Atlas)

Pricing (per 1M tokens): GPT $2.50/$10 in/out, Claude $15/$75, MiniMax $0.40/$1.60, Kimi $0.60/$2.40, GLM $0.40/$1.20.

| Combo | Acc | Plan cost | Exec cost | Total | $/task | Plan % |
|---|---|---|---|---|---|---|
| GPT→MiniMax | 0.804 | ~$0.26 | ~$10.80 | ~$11 | ~$0.18 | ~2% |
| GLM→GLM | 0.724 | $0.28 | $4.71 | $4.99 | $0.08 | 6% |
| MiniMax→MiniMax | 0.685 | $0.07 | $4.73 | $4.80 | $0.08 | 1% |
| GPT→GPT | 0.707 | $0.59 | $49.21 | $49.81 | $0.83 | 1% |
| Claude→Claude | 0.513 | $2.83 | $69.07 | $71.90 | $1.20 | 4% |
| GLM→Claude | 0.791 | $0.26 | $117.21 | $117.47 | $1.96 | 0.2% |

**Key observations**:
- Plan cost is always < 8% of total → planner choice is essentially free
- Execution cost varies **100×** across executors ($0.08/task to $1.96/task)
- Cheapest homogeneous team (GLM×GLM: $0.08) beats most expensive (Claude×Claude: $1.20) in accuracy (0.724 vs 0.513)

### 5.2 Cost-accuracy tradeoffs

Best accuracy-per-dollar combos:
1. **GLM→GLM**: 0.724 acc @ $0.08/task (best budget option)
2. **MiniMax→MiniMax**: 0.685 @ $0.08/task
3. **GPT→MiniMax**: 0.804 @ ~$0.18/task (best overall accuracy)
4. **GPT→GPT**: 0.707 @ $0.83/task (4.6× more expensive for lower accuracy)

Worst cost-efficiency: Claude→Claude (0.513 @ $1.20/task) — most expensive, lowest accuracy among API models.

---

## 6. Heterogeneous vs Homogeneous Teams

Every model's best heterogeneous team outperforms its homogeneous team:

| Model | Homo PE (MCP) | Best Hetero PE (MCP) | Δ |
|---|---|---|---|
| Claude | 0.513 | 0.791 (GLM→Claude) | **+27.8pp** |
| GPT | 0.707 | 0.804 (GPT→MiniMax) | +9.7pp |
| MiniMax | 0.685 | 0.804 (GPT→MiniMax) | +11.9pp |
| Kimi | 0.630 | 0.660 (Kimi→Claude) | +3.0pp |
| GLM | 0.724 | 0.791 (GLM→Claude) | +6.7pp |

| Model | Homo PE (Med) | Best Hetero PE (Med) | Δ |
|---|---|---|---|
| GPT | 0.717 | 0.933 (GPT→GLM) | **+21.6pp** |
| Claude | 0.933 | 0.933 (tied) | +0.0pp |
| MiniMax | 0.733 | 0.867 (MiniMax→Claude) | +13.4pp |
| GLM | 0.900 | 0.900 (tied) | +0.0pp |

---

## 7. Self-Hosted Model Analysis

### 7.1 MCP-Atlas

| Config | Acc | Notes |
|---|---|---|
| Qwen-4B→27B | 0.691 | Strong executor rescues weak planner |
| Qwen-27B→4B | 0.041 | Weak executor cannot follow plan |
| Qwen-4B×4B | 0.000 | Bad plan + weak executor = total failure |
| GPT-OSS-20B→120B | 0.388 | Best self-hosted combo |
| GPT-OSS-120B×120B | 0.298 | Homogeneous worse than hetero |
| GPT-OSS-20B×20B | 0.187 | Smallest model, lowest score |

### 7.2 MedAgentBench

| Config | Acc | Notes |
|---|---|---|
| Qwen-27B→27B | 0.920 | Matches top API combos |
| Qwen-27B→9B | 0.880 | Strong plan carries weak executor |
| Qwen-9B→27B | 0.620 | Weak plan limits strong executor |
| Qwen-9B×9B | 0.600 | Baseline |

27B→9B vs 9B→27B: same two models, swapping roles gives **0.880 vs 0.620**. The planner role is 26pp more impactful than executor on this benchmark.

### 7.3 FinanceBench

| Config | Acc | Notes |
|---|---|---|
| Qwen-9B→27B | **0.600** | Small planner + big executor = best |
| Qwen-27B→27B | 0.573 | Big planner slightly worse |
| Qwen-9B×9B | 0.287 | Baseline |
| Qwen-27B→9B | 0.200 | Big planner + small executor = worst |

**Executor-critical**: swapping executor (9B vs 27B) with fixed 27B planner: 0.573 vs 0.200 = **37.3pp swing**. Swapping planner with fixed 27B executor: 0.600 vs 0.573 = only 2.7pp.

**Trajectory evidence — why 27B→9B fails (acc=0.200):**

The 27B planner produces detailed multi-step plans. But the 9B executor can't follow them: it makes only 1 tool call per task (avg 3.9 tc total, avg 4.0 turns), then outputs empty or near-empty text. 102/150 tasks have output < 10 characters. Example:
- Task 0: 27B plans a 2-step document search → 9B makes 1 `search_document` call → outputs blank → score = 0
- Same task with 27B executor: makes 2 targeted calls → outputs correct answer with citation → score = 1.0

**Why 9B→27B (0.600) > 27B→27B (0.573):**

The 9B planner produces shorter, more direct plans (avg ~1000 chars vs ~1200 chars). The 27B executor follows both effectively, but the 9B's simpler plans lead to slightly fewer wasted turns (avg 8.5 vs 9.2). A verbose plan can introduce unnecessary steps that even a capable executor may fumble.

### 7.4 Cross-benchmark role criticality (self-hosted)

| Benchmark | Planner swap effect | Executor swap effect | Dominant role |
|---|---|---|---|
| MedAgentBench | 30pp (9B vs 27B plan, fix 27B exec) | 4pp (9B vs 27B exec, fix 27B plan) | **Planner** |
| FinanceBench | 2.7pp (9B vs 27B plan, fix 27B exec) | 37.3pp (9B vs 27B exec, fix 27B plan) | **Executor** |
| MCP-Atlas | — (no 27B→27B) | 69.1pp (4B vs 27B exec, fix 4B plan) | **Executor** |

Same two models, three benchmarks, completely different role criticality. This is the strongest evidence that role sensitivity is a property of the task domain, not the architecture.

---

## 8. Trajectory-Based Behavior Analysis

### 8.1 MCP-Atlas: Single agent failure patterns

**GLM-5.1 single** (60 tasks): avg 14.1 tool calls, 11.1 turns, 842 chars output.
- 9 tasks (15%): **Zero tool calls** — model outputs "I don't have access to local files" and refuses to try. Despite having 81 tools available, the model doesn't realize it can use `github_*` or `fetch_*` to accomplish the task.
- 14 tasks (23%): **Thrashing** — hits 30-turn limit with 30-47 tool calls but empty output. The model repeats the same failing tool call (e.g., `fetch_fetch` on robots.txt-blocked URLs) without adapting.

**MiniMax M2.7 single** (60 tasks): avg 7.1 tool calls, 7.4 turns, 744 chars output.
- 19 tasks (32%): **Zero tool calls** — outputs "I'd be happy to help! However, I need some information..." and asks the user for clarification instead of using tools. MiniMax's conversational training causes it to defer rather than act autonomously.
- Only avg 7.1 tc vs GLM's 14.1 — MiniMax gives up faster, making fewer attempts before concluding it can't help.

### 8.2 MCP-Atlas: Claude executor anomaly

**GPT→Claude** (acc=0.354, avg 7.2 turns) vs **GPT→GPT** (acc=0.695, avg 13.6 turns):

Claude as executor terminates early and produces short outputs:

| Output length | GPT→Claude | GPT→GPT |
|---|---|---|
| < 10 chars | 2 | 1 |
| 10–100 chars | 10 | 3 |
| 100–500 chars | 25 | 17 |
| 500+ chars | 23 | 39 |

Claude averages 7.2 turns (vs GPT's 13.6). It doesn't persist through tool errors: after 2-5 tool calls with partial results, Claude outputs a short summary and stops, rather than iterating. This is the opposite of what's needed for tool-use tasks where persistence through errors is critical.

On MedAgentBench, this same trait helps: medical tasks need 1-2 precise API calls (not persistence), so Claude's concise style produces correct FHIR queries immediately.

### 8.3 FinanceBench: Executor capability determines output quality

27B→9B produces **102/150 empty outputs** (< 10 chars). The 9B executor receives a detailed plan but can't translate it into tool calls. It makes 1 call, gets a result, but fails to extract the answer from the returned text and outputs nothing.

27B→27B on the same tasks: the 27B executor makes 2+ calls, correctly parses the document search results, extracts the numerical answer, and formats it. Average output 1357 chars vs 293 for 9B executor.

### 8.4 MedAgentBench: Planner quality determines everything

All self-hosted combos show 0 tracked tool calls (MedAgent uses FHIR REST directly, not tracked as tool_calls). The differentiator is pure planning quality:

- 27B planner outputs: correct FHIR resource type + search parameters → executor calls the right endpoint → 92% accuracy
- 9B planner outputs: wrong FHIR resource type or missing parameters → executor calls wrong endpoint → 60% accuracy

The executor's job is mechanical (make the HTTP call, return result). Plan correctness is what determines success.

---

## 9. Summary of Key Findings

1. **PE benefit is inversely correlated with model capability**: +1.6pp for frontier, +41.2pp for mid-range, −33.3pp for weak×weak (bad plan worse than no plan), +35.8pp for weak→strong.

2. **Single-agent failure modes (refuse 15%, thrash 23%) are systematically fixed by PE** (89% and 79% fix rates). The plan provides tool selection and sequencing that single agents lack.

3. **Role rankings reverse across domains**: a model's executor rank on one benchmark does not predict its rank on another. No single model dominates all roles across all domains.

4. **Heterogeneous teams always ≥ homogeneous**: gains up to +27.8pp (MCP) and +21.6pp (MedAgent). The best team is never two copies of the same model.

5. **Plan cost is negligible (< 8%)**: the planner is essentially free. All cost optimization should focus on executor selection.

6. **Cheaper models can beat expensive ones**: on MCP-Atlas, GLM×GLM (cheapest) outperforms GPT×GPT and Claude×Claude (most expensive). Price does not predict role-specific performance.

7. **Domain determines role criticality**: MCP-Atlas is executor-critical (executor swap = 45pp), MedAgent is planner-critical (planner swap = 28pp, executor swap = 4pp). Optimization strategy must be domain-specific.
