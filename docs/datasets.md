# Datasets

Pass via `--dataset <name>` to `python -m agent_cap.agents`. The loader
lives in `agent_cap/runner/unified_runner.py:_load_dataset_tasks`.
SWE-Bench Lite / Pro are covered separately in
[RUN_NO_DOCKER.md §3](RUN_NO_DOCKER.md).

| Name | Source | Eval | Extra setup |
|---|---|---|---|
| `imo-answerbench` | HF `Hwilner/imo-answerbench` | `imo` (math_verify + llm-judge fallback) | none |
| `mcp-atlas` | HF `ScaleAI/mcp-atlas` (filtered to free-tier servers) | `gtfa` (Gemini judge) | MCP server running on :1984 |
| `tau2-banking` | local clone | `tau2` (built-in scorer) | clone tau2-bench |
| `medagentbench` | local clone | `exact` or `llm-judge` | clone MedAgentBench |
| `financebench` | local clone | `llm-judge` | clone financebench |

All HuggingFace datasets are downloaded to `~/.cache/huggingface/` on
first run. Local clones go under `third_party/`.

---

## imo-answerbench

Math problems with short numerical answers; tools optional.

```bash
python -m agent_cap.agents \
  --strategy single \
  --model openai/gpt-oss-120b \
  --base-url http://localhost:30000/v1 \
  --dataset imo-answerbench \
  --evaluator imo \
  --tool-backend none \
  --num-tasks 20
```

For LLM-judge fallback when `math_verify` fails, set
`OPENROUTER_API_KEY`. Set `--tool-backend python` to let the agent run
Python during reasoning.

---

## mcp-atlas

Tool-use over 20+ MCP servers. **Requires the MCP server running** —
see [mcp-server.md](mcp-server.md).

```bash
# Terminal 1
bash mcp-server/start.sh

# Terminal 2
python -m agent_cap.agents \
  --strategy single \
  --model openai/gpt-oss-120b \
  --base-url http://localhost:30000/v1 \
  --dataset mcp-atlas \
  --evaluator gtfa \
  --tool-backend mcp \
  --mcp-server-url http://localhost:1984 \
  --num-tasks 60
```

The loader filters the public dataset to tasks whose `ENABLED_TOOLS`
fit a 22-server free subset (deterministic, no shuffle). `gtfa`
evaluator uses Gemini via OpenRouter — set `OPENROUTER_API_KEY`.

---

## tau2-banking

Multi-turn customer-service simulation. Built-in tau2 scorer.

```bash
git clone https://github.com/sierra-research/tau2-bench third_party/tau2-bench

python -m agent_cap.agents \
  --strategy single \
  --model openai/gpt-oss-120b \
  --base-url http://localhost:30000/v1 \
  --dataset tau2-banking \
  --evaluator tau2 \
  --num-tasks 25
```

Reads `third_party/tau2-bench/data/tau2/domains/banking_knowledge/tasks/task_*.json`.
The evaluator uses tau2's own retrieval + reward-basis scoring.

---

## medagentbench

EHR / clinical reasoning tasks with structured function calls.

```bash
git clone https://github.com/stanfordnlp/MedAgentBench third_party/MedAgentBench

python -m agent_cap.agents \
  --strategy single \
  --model openai/gpt-oss-120b \
  --base-url http://localhost:30000/v1 \
  --dataset medagentbench \
  --evaluator llm-judge \
  --judge model=openai/gpt-4o-mini,api_key=$OPENROUTER_API_KEY,base_url=https://openrouter.ai/api/v1 \
  --num-tasks 30
```

Reads `third_party/MedAgentBench/data/medagentbench/{test_data_v1,funcs_v1}.json`.

---

## financebench

SEC-filing QA. Agent gets `search_document` + `calculate` tools.

```bash
git clone https://github.com/patronus-ai/financebench third_party/financebench

python -m agent_cap.agents \
  --strategy single \
  --model openai/gpt-oss-120b \
  --base-url http://localhost:30000/v1 \
  --dataset financebench \
  --evaluator llm-judge \
  --judge model=openai/gpt-4o-mini,api_key=$OPENROUTER_API_KEY,base_url=https://openrouter.ai/api/v1 \
  --num-tasks 50
```

Reads `third_party/financebench/data/financebench_open_source.jsonl`.

---

## Common flags

| Flag | Default | Notes |
|---|---|---|
| `--num-tasks N` | 0 (all) | Cap at first N tasks (deterministic, no shuffle) |
| `--output-dir DIR` | none | Persist per-task results + `results.jsonl` |
| `--resume` | off | Skip tasks already in `<output-dir>/results.jsonl` |
| `--max-turns N` | 10 | Agent turn cap per task |
| `--strategy` | `single` | Also: `plan-execute`, `supervisor`, `sequential` |
| `--mock` | off | Use `MockLLMClient` for offline smoke tests |

See [cli.md](agents/cli.md) for the full flag matrix and
[strategies.md](agents/strategies.md) for when to pick which strategy.
