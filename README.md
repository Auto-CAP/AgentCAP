# Raw data for AgentCAP cost-accuracy plots

This folder packages the data behind the two plots requested in the review:
1. Homogeneous vs heterogeneous cost--accuracy Pareto frontiers.
2. Local vs hybrid vs API team configurations.

## Files

| File | Purpose | Source |
|---|---|---|
| `cost_accuracy_team_configs.csv` | Master table: every team configuration shown in the paper, tagged with team type and deployment mode | Paper Section 6 cost--accuracy table |
| `plot1_homo_vs_hetero_allapi.csv` | Subset for plot 1: all-API teams only, distinguished as homogeneous (planner=executor) vs heterogeneous (planner!=executor) | Filtered from master |
| `plot2_deployment_modes.csv` | Subset for plot 2: every team grouped by deployment mode (`all-API`, `hybrid`, `all-local`) | Filtered from master |
| `mcp_atlas_selfplay_per_task.csv` | Per-task DB results for the four API self-play runs on MCP-Atlas (60 tasks each), if recomputation from raw is needed | `/home/sicheng/AgentCAP/results/hybrid_*_self.db` |
| `financebench_selfplay_aggregate.csv` | Aggregate per-role token totals for FinanceBench self-play, used to derive cost from provider prices | `/data/sicheng/agent-team-data/*_*_financebench/team_plan_execute/metrics_*190823.json` |

## Schema

`cost_accuracy_team_configs.csv` columns:

- `benchmark` (MCP-Atlas, MedAgentBench, FinanceBench, IMO-AnswerBench, SWEBench)
- `planner`, `executor` model identifiers
- `team_type` (homogeneous | heterogeneous)
- `deployment_mode` (all-API | hybrid | all-local)
- `cost_per_task_usd` (USD)
- `accuracy_pct` (%)

## Pricing assumptions (for derived costs)

Used to convert token totals into dollar cost when a benchmark only has token-level data:

- API per-million tokens (input / output): GPT-5.4 \$2.50/\$10.00; Claude-Opus-4.6 \$5.00/\$25.00; MiniMax-M2.7 \$0.40/\$1.60; GLM-5.1 \$0.95/\$3.15.
- Self-hosted: 4 x H100 SXM at \$1.89/hr/GPU, amortized via measured vLLM throughput at the target concurrency. Production-concurrency points use c=1000.
- API prompt-cache discount: 65% reuse on the executor input path (cached input charged at 35% of the input price).

## Suggested plots

- **Plot 1 (homo vs hetero, all-API)**: scatter `cost_per_task_usd` vs `accuracy_pct`, color by `team_type`. Per benchmark connect Pareto frontier. The story: heterogeneous teams reach or approach homogeneous frontiers at much lower cost on most benchmarks.
- **Plot 2 (deployment modes)**: scatter all teams, color by `deployment_mode`. Per benchmark connect Pareto frontier. The story: which deployment mode owns the frontier flips by benchmark (hybrid for MCP-Atlas, all-API for MedAgentBench, all-local for FinanceBench, hybrid/all-local at top tie for IMO-AnswerBench, all-API for SWEBench).
