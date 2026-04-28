# Raw data for AgentCAP cost-accuracy plots

Data is split by **deployment mode** so each plot can be assembled directly:

- `data_api.csv` &mdash; API planner with API executor (all-API teams)
- `data_hybrid.csv` &mdash; mixed teams (API + self-hosted)
- `data_self.csv` &mdash; self-hosted planner with self-hosted executor (all-local teams)

## Schema (same in all three files)

| Column | Meaning |
|---|---|
| `benchmark` | MCP-Atlas, MedAgentBench, FinanceBench, IMO-AnswerBench, or SWEBench |
| `planner` | Model assigned to the planning role |
| `executor` | Model assigned to the execution role |
| `accuracy_pct` | Task success rate (%) |
| `cost_per_task_usd` | Average dollar cost per task (USD); blank if not yet computed |
| `source` | Provenance: `metrics_json` (raw run logs), `imo_db` (sqlite), `paper_matrix` (paper Section 4 / Section 6 tables) |

## Coverage

`data_api.csv`: full 4x4 API matrix on the four core benchmarks; SWEBench currently has the 3x3 GPT-5.4 / Claude-Opus-4.6 / MiniMax-M2.7 block (GLM-5.1 not yet in collaborator's SWEBench runs).

| Benchmark | API rows | Notes |
|---|---|---|
| MCP-Atlas | 16 | full 4x4 |
| MedAgentBench | 16 | full 4x4 |
| FinanceBench | 16 | full 4x4 |
| IMO-AnswerBench | 16 | accuracy from paper Section 4 (LLM-judge), cost from sqlite DB |
| SWEBench | 9 | 3x3, no GLM-5.1 row/column yet |

`data_hybrid.csv`: every API+local team we have logs for (FinanceBench has the densest coverage; SWEBench hybrid rows come from the paper Section 4 matrix).

`data_self.csv`: self-hosted planner and executor (SWEBench includes the 4x4 local-local block from the paper Section 4 matrix; MCP-Atlas and IMO-AnswerBench come from raw runs).

## Pricing assumptions

API token prices (input / output, $ per million tokens):

| Model | Input | Output |
|---|---|---|
| GPT-5.4 | 2.50 | 10.00 |
| Claude-Opus-4.6 | 5.00 | 25.00 |
| MiniMax-M2.7 | 0.40 | 1.60 |
| GLM-5.1 | 0.95 | 3.15 |

API costs apply a 65% prompt-cache discount on the executor input path (cached tokens charged at 35% of the input price).

Self-hosted prices at 4 x H100 SXM, $1.89/hr/GPU, c=1000, TP=4 (paper Appendix throughput table); prefill and decode within ~5% on long-exec workloads, both treated as one rate per model:

| Model | $ per million tokens |
|---|---|
| Qwen3.5-27B | 0.46 |
| GPT-OSS-120B | 0.29 |
| GPT-OSS-20B | 0.20 (estimated by size scaling) |
| Qwen3.5-4B | 0.10 (estimated) |
| Qwen3.5-4B | 0.06 (estimated) |
| Qwen3-32B | 0.55 (estimated; only present in FinanceBench hybrid runs) |

## Suggested plots

- **Plot 1 (homo vs heterogeneous, all-API)**: scatter `cost_per_task_usd` (log x) vs `accuracy_pct` from `data_api.csv`, color by whether `planner == executor`. Per benchmark, draw the Pareto frontier.

- **Plot 2 (deployment modes)**: combine all three files and color by which file each row comes from (API / hybrid / self). Per benchmark, draw the Pareto frontier.

## Other files in this directory

- `cost_accuracy_team_configs.csv` &mdash; original aggregate table from paper Section 6 cost-accuracy summary
- `plot1_homo_vs_hetero_allapi.csv` &mdash; subset for plot 1 from the original master table
- `plot2_deployment_modes.csv` &mdash; subset for plot 2 from the original master table
- `plot1_full_matrix_allapi.csv` &mdash; older single-file version of `data_api.csv` (kept for compatibility)
- `mcp_atlas_selfplay_per_task.csv` &mdash; per-task DB results for MCP-Atlas API self-play (60 tasks each)
- `financebench_selfplay_aggregate.csv` &mdash; per-role token totals for FinanceBench self-play
