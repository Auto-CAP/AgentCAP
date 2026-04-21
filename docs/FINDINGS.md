# AgentCAP: Principal Findings

We evaluate plan-execute agent teams across 4 API models and 4 self-hosted models on three benchmarks (MCP-Atlas: open-domain tool-use, MedAgentBench: medical reasoning, FinanceBench: financial document QA). Below we present five principal findings, each supported by quantitative evidence and mechanistic analysis from execution trajectories.

---

## Finding 1: Plan-execute consistently improves over single-agent baselines, and the magnitude of improvement is inversely related to single-agent capability.

**Quantitative evidence.** On MCP-Atlas, plan-execute improves all tested API models: GPT-5.4 from 65.4% to 69.5% (+4.1%), GLM-5.1 from 57.0% to 72.4% (+15.4%), and MiniMax from 52.0% to 68.5% (+16.5%). On MedAgentBench with self-hosted models, the 9B model improves from 60.0% (homogeneous) to 88.0% when a 27B model provides the plan.

**Mechanism.** Trajectory analysis reveals three systematic failure modes in single-agent execution that plan-execute addresses.

(1) *Decision paralysis*: confronted with 80+ available tools and an open-ended task, single agents default to asking the user for input rather than acting autonomously. GPT-5.4 single agent produces zero tool calls on 35% of tasks, instead outputting requests like "Please upload your project files" — despite having tools such as `desktop-commander_list_directory` and `filesystem_read_file` available. This mirrors the *choice overload* effect studied in human decision-making (Iyengar & Lepper, 2000) and echoes dual-process theory (Kahneman, 2011): without a plan activating deliberate (System 2) reasoning, the model falls back on a safe, conversational (System 1) default.

(2) *Refusal*: the agent outputs "I don't have access to that" and makes zero tool calls, despite having relevant tools available (15% of GLM tasks, 32% of MiniMax tasks).

(3) *Unproductive iteration*: the agent repeatedly invokes the same failing tool without adapting its strategy, exhausting the turn budget with no output (23% of GLM tasks).

Under plan-execute, the plan eliminates decision paralysis by specifying which tools to invoke and in what order. The executor no longer needs to decide *whether* to act — only *how* to execute the given step. PE resolves 89% of refusal failures and 79% of unproductive iteration failures. Critically, the plan's value is not just task decomposition but *commitment to action*: it converts a hesitant single agent into a directed executor.

---

## Finding 2: A model's role-specific effectiveness is not predictable from its single-agent ranking, and role rankings reverse across task domains.

**Quantitative evidence.** On MCP-Atlas, we rank each API model by its average accuracy when serving as planner (averaging over all executors) and as executor (averaging over all planners). Claude Opus ranks #3 as planner but #5 (last) as executor. On MedAgentBench, Claude ranks #1 in both roles. GPT-5.4 shows the opposite reversal: #3 executor on MCP-Atlas, #5 (last) on MedAgentBench.

**Mechanism.** The reversal arises from a mismatch between model behavioral tendencies and domain-specific demands.

*Claude as MCP-Atlas executor* terminates after 4–8 turns (vs. 12–17 for other models) and produces outputs of median length 180 characters (vs. 900+ for GPT). Trajectory inspection shows Claude does not recover from tool errors: when the first search returns "No results found," Claude tries one alternative query, then outputs a brief apology and stops. GPT, by contrast, switches tool categories entirely — from `ddg-search` to `fetch_fetch` (direct URL access) to `github_search_repositories` — and persists until it obtains the information.

*Claude as MedAgentBench executor* achieves 93.3% because medical tasks require 1–2 precise FHIR API calls, not iterative exploration. Claude's tendency toward concise, targeted action is well-matched to tasks where the first tool call is likely correct.

This finding implies that single-agent leaderboard rankings are insufficient for configuring agent teams; role-specific, domain-specific evaluation is necessary.

---

## Finding 3: Role criticality — whether upgrading the planner or executor yields greater returns — is determined by the task domain, not by the architecture or the models involved.

**Quantitative evidence.** We measure role criticality via a controlled swap: fix one role's model, swap the other between a 9B and 27B self-hosted model, and measure accuracy change. Across three benchmarks with the same model pair:

| Benchmark | Planner swap | Executor swap | Critical role |
|---|---|---|---|
| MedAgentBench | 9B→27B planner: 60.0%→92.0% (+32.0%) | 9B→27B executor: 88.0%→92.0% (+4.0%) | Planner |
| FinanceBench | 9B→27B planner: 60.0%→57.3% (−2.7%) | 9B→27B executor: 20.0%→57.3% (+37.3%) | Executor |

For API models on MCP-Atlas, the executor swap effect is similarly large: fixing the GPT planner, swapping executor from MiniMax (80.4%) to Claude (35.4%) yields a 45.0% accuracy swing.

The same two models, with the same plan-execute architecture, yield opposite role criticality depending solely on the task domain.

**Mechanism.** The divergence traces to where each domain's difficulty concentrates.

*MedAgentBench (planner-critical)*: success requires selecting the correct FHIR resource type and search parameters — a reasoning problem. The 27B planner correctly identifies that a medication lookup requires `MedicationRequest?patient=X&code=Y`; the 9B planner outputs `Patient?name=X`, querying the wrong endpoint entirely. Once the correct endpoint is specified, even the 9B executor can issue the HTTP call and return the result. Plan quality determines task success; execution is mechanical.

*FinanceBench (executor-critical)*: the plan specifies `search_document(doc_name="3M_2018_10K", query="capital expenditure")` — both 9B and 27B planners produce comparable plans. The difference lies in what happens after the document search returns a multi-page excerpt. The 27B executor parses the financial table, extracts the relevant figure, and outputs "The capital expenditure for FY2018 was $1,577 million." The 9B executor receives the same search result but fails to extract the answer, producing blank output in 68% of tasks (102/150). The bottleneck is information extraction from retrieved documents, a capability that varies with executor size.

---

## Finding 4: Heterogeneous teams outperform homogeneous teams across all models and benchmarks tested, and the optimal team composition pairs a strong planner with a cost-efficient executor.

**Quantitative evidence.** Every API model's best heterogeneous configuration exceeds its homogeneous baseline: Claude from 51.3% to 79.1% (+27.8%), GPT from 70.7% to 80.4% (+9.7%), MiniMax from 68.5% to 80.4% (+11.9%). The pattern holds across benchmarks: on MedAgentBench, GPT improves from 71.7% (homogeneous) to 93.3% (GPT planner + GLM executor, +21.6%). On FinanceBench, the 9B planner with 27B executor (60.0%) outperforms the 27B homogeneous team (57.3%).

The best overall configuration on MCP-Atlas — GPT planner with MiniMax executor (80.4%) — outperforms the best homogeneous team (GLM×GLM: 72.4%) by 8.0% while costing approximately $0.10/task vs. $0.83/task for GPT×GPT. This is possible because planning requires a single LLM call (< 8% of total cost), while execution requires 8–20 calls (> 92% of total cost). Using an expensive model for the inexpensive role (planning) and a cheap model for the expensive role (execution) optimizes both accuracy and cost simultaneously.

**Mechanism.** Homogeneous teams force a single model to serve both roles, but planning and execution demand different capabilities. Planning is a reasoning task (decompose a problem into tool-call sequences); execution is a grounding task (translate abstract steps into concrete API calls and handle errors). Models that reason well (GPT, Claude) may lack the persistence for multi-turn tool use; models with strong tool-use patterns (GLM, MiniMax) may lack the reasoning depth for planning. Heterogeneous assignment allows each role to be filled by its best-suited model.

A subtler effect appears on FinanceBench: the 9B planner with 27B executor (60.0%) slightly outperforms the 27B planner with 27B executor (57.3%). The 9B planner produces shorter, more direct plans (~1000 chars vs. ~1200 chars). On the task of computing a quick ratio, the 9B plan instructs the executor to search and calculate directly; the 27B plan adds intermediate verification steps that introduce additional failure points. This suggests that plan verbosity can be counterproductive — a more capable planner does not always produce a more effective plan.

---

## Finding 5: The cost structure of plan-execute creates an asymmetric optimization landscape: planner upgrades are nearly free, while executor cost varies by two orders of magnitude.

**Quantitative evidence.** Across all MCP-Atlas configurations, plan cost accounts for 0.1–8% of total per-run cost. The planner makes one call per task; the executor makes 8–20 calls with growing context windows. Concrete comparison at matched accuracy (~70%):

| Configuration | Accuracy | $/task | Plan cost % |
|---|---|---|---|
| GLM→GLM | 72.4% | $0.08 | 5.6% |
| GPT→GPT | 70.7% | $0.83 | 1.2% |
| Claude→Claude | 51.3% | $1.20 | 3.9% |

Replacing GLM's planner with Claude (the most expensive model) increases plan cost from $0.28 to $2.83 but does not change executor cost ($4.71). Total cost rises from $4.99 to $7.54 (+51%), while accuracy rises from 72.4% to 79.1%. In contrast, replacing GLM's executor with Claude changes executor cost from $4.71 to $117.21 while accuracy rises by the same amount — at 23× the cost.

**Implication.** This asymmetry yields a clear practitioner guideline: always allocate the strongest available model to the planner role regardless of its per-token price, then select the cheapest executor that meets the accuracy target. This strategy simultaneously maximizes accuracy and minimizes cost because the planner's contribution to total cost is negligible.
