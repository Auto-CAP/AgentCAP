# agent_cap.agents

Unified agent runtime for AgentCAP. One CLI, one Python API, four built-in
strategies, three built-in LLM protocols, five built-in evaluators. Every
piece is replaceable via a `@register_*` decorator.

## Install

This module ships with AgentCAP. From the repo root:

```bash
pip install -e .
```

Optional, installed only if you actually use them:

| You want | Install |
|---|---|
| harmony protocol (`gpt-oss-*` models) | `pip install openai-harmony` |
| `--evaluator imo` symbolic check | `pip install math-verify` |
| `--evaluator gtfa` | (already shipped) |
| `--tool-backend mcp` | local MCP server reachable on the URL you pass |
| `--tool-backend math-python` | `pip install jupyter-client ipykernel` |

## Quick examples

### 1. Smoke test, no API key required

```bash
python -m agent_cap.agents --mock --strategy plan-execute \
  --task "What is (12 + 8) / 4?"
```

### 2. One model, OpenAI-compatible endpoint (self-hosted or hosted)

```bash
python -m agent_cap.agents --strategy single \
  --agent agent=name=Qwen/Qwen2.5-72B-Instruct,base_url=http://localhost:30000/v1,api_key=EMPTY \
  --dataset imo-answerbench --num-tasks 5 \
  --tool-backend math-python \
  --evaluator imo \
  --output-dir results/qwen_imo --resume
```

### 3. Two models, plan-execute, mixed protocols

```bash
python -m agent_cap.agents --strategy plan-execute \
  --agent planner=name=Qwen/Qwen2.5-72B-Instruct,base_url=http://localhost:30000/v1,api_key=EMPTY \
  --agent executor=name=gpt-oss-120b,base_url=http://localhost:8000/v1,api_key=EMPTY \
  --dataset imo-answerbench --tool-backend math-python --evaluator imo \
  --output-dir results/mixed
```

`Qwen` auto-routes to the openai protocol, `gpt-oss-120b` auto-routes to
harmony. Same CLI for both.

### 4. LLM as a judge

```bash
python -m agent_cap.agents --strategy single \
  --agent agent=name=Qwen/Qwen2.5-7B-Instruct,base_url=http://localhost:30000/v1,api_key=EMPTY \
  --dataset imo-answerbench --num-tasks 5 \
  --tool-backend math-python \
  --evaluator llm-judge \
  --judge "name=gpt-4o,base_url=https://api.openai.com/v1,api_key=$OPENAI_API_KEY"
```

### 5. gpt-oss on sglang (harmony protocol, native /generate)

```bash
python -m agent_cap.agents --strategy single \
  --agent agent=name=gpt-oss-120b,base_url=http://localhost:30000,api_key=EMPTY,engine=sglang \
  --dataset imo-answerbench --num-tasks 5 \
  --tool-backend math-python --evaluator imo \
  --output-dir results/imo_gptoss_sglang
```

Drop `engine=sglang` (or set `engine=vllm`) to hit vLLM's `/v1/completions`
instead. Same CLI either way.

### 6. 100-worker swarm from a YAML

```yaml
strategy: supervisor
agents:
  supervisor: {name: Qwen/Qwen2.5-72B-Instruct, base_url: http://localhost:30000/v1, api_key: EMPTY}
  worker:    {name: Qwen/Qwen2.5-7B-Instruct,  base_url: "http://localhost:3010{i}/v1", api_key: EMPTY, replicas: 100}
task: "..."
```

```bash
python -m agent_cap.agents --config configs/agents_swarm.yaml
```

## Documentation

See `agent_cap/agents/docs/` for the full reference:

| File | Topic |
|---|---|
| [docs/cli.md](docs/cli.md) | All CLI flags and how they combine with YAML |
| [docs/yaml.md](docs/yaml.md) | YAML config: agents pool, roles, replicas, share_state, includes |
| [docs/concepts.md](docs/concepts.md) | Agent, Strategy, Protocol, Tool, Evaluator: what each is and how they interact |
| [docs/protocols.md](docs/protocols.md) | LLM protocols: openai, harmony, mock; auto-routing rules; adding new |
| [docs/extending.md](docs/extending.md) | Recipes: add a strategy, protocol, tool, evaluator, or whole new agent topology |
