# AgentCAP

Benchmarking framework for AI agent systems — measures **Cost**, **Accuracy**,
and **Performance** of single-agent and multi-agent workloads across multiple
benchmarks (NL2Repo, IMO AnswerBench, MCP-ATLAS, MedAgentBench, FinanceBench,
SWE-Bench). Runs AgentCAP's own team strategies and other agent harnesses
(Codex, Claude Code, DeepSeek Harness, ...) on
[Harbor](https://github.com/harbor-framework/harbor), with the same metering
for all of them.

## Install

```bash
git clone --recurse-submodules https://github.com/Auto-CAP/AgentCAP.git
cd AgentCAP
pip install -e ".[harbor]"
```

The Harbor quick start below needs Python ≥ 3.12 and Docker. For the
[native runner](#native-runner-without-harbor) alone, `pip install -e .` is enough.

## Quickstart — NL2Repo on Harbor

Harbor builds a container for each task, runs the agent against it, and grades
the result in a separate container. Every run below has the same shape: a
benchmark, a harness (DeepSeek Harness, AgentCAP's own teams, Codex, ...), and a model.

### 1. Serve a model

```bash
vllm serve Qwen/Qwen3.5-4B \
  --port 30000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Any OpenAI-compatible endpoint works. For an API model, skip this step and pass
e.g. `--model openai/gpt-5.4` (key in `OPENAI_API_KEY`) instead of `--model`/`--base-url` below.

### 2. Build the NL2Repo tasks (once)

```bash
git clone https://github.com/multimodal-art-projection/NL2RepoBench
python -m agent_cap.harbor adapt nl2repo --source NL2RepoBench --out datasets/nl2repo
```

This writes 104 Harbor tasks, one per directory in `datasets/nl2repo/`. Each
task is named after the open-source Python project the agent has to rebuild
from a written specification (`start.md`); the original code is not in the container.
A separate verifier container, built from the task's official evaluation image
(pulled from ghcr.io on first use), runs the project's own tests and scores the
fraction that pass.

The examples use one task, **`math-verify`**: rebuild Hugging Face's
[Math-Verify](https://github.com/huggingface/Math-Verify), a library that
parses math answers (LaTeX or plain text) and checks whether two of them are
equivalent. It has 192 tests. `--task-names` picks tasks by name
(`ls datasets/nl2repo` lists them); without it, all 104 run, and
`--concurrency 4` runs four at a time.

### 3. DeepSeek Harness

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (DSH) runs inside the
task container; setup installs Node and DSH there. `--meter` sends its model
requests through AgentCAP's metering proxy on this machine, so the container
reaches `localhost` and every request's tokens, TTFT and decode time are
recorded.

One DSH agent:

```bash
python -m agent_cap.harbor run --harness dsh --meter \
  --benchmark nl2repo --dataset datasets/nl2repo --task-names math-verify \
  --model Qwen/Qwen3.5-4B --base-url http://localhost:30000/v1
```

An Agent Team (a lead plus 2 teammates that share the workspace):

```bash
python -m agent_cap.harbor run --harness dsh --meter --ak teammates=2 \
  --benchmark nl2repo --dataset datasets/nl2repo --task-names math-verify \
  --model Qwen/Qwen3.5-4B --base-url http://localhost:30000/v1
```

With the DeepSeek API instead of a local server, the key is read from `DEEPSEEK_API_KEY`:
`--model deepseek/deepseek-chat` (no `--base-url`). DSH options (`--dsh-version`,
`--effort`, `--ak context_window=...`) are in [docs/harbor.md](docs/harbor.md#deepseek-harness---harness-dsh).

### 4. AgentCAP single agent

`--harness capagent` (the default) runs AgentCAP's own strategies. The agent
loop and every model request run on this machine, and its `bash` and
`str_replace_editor` calls run in the task container:

```bash
python -m agent_cap.harbor run --benchmark nl2repo --dataset datasets/nl2repo --task-names math-verify \
  --model Qwen/Qwen3.5-4B --base-url http://localhost:30000/v1
```

### 5. Plan-execute

Both roles on the same endpoint:

```bash
python -m agent_cap.harbor run --benchmark nl2repo --dataset datasets/nl2repo --task-names math-verify \
  --strategy plan-execute \
  --model Qwen/Qwen3.5-4B --base-url http://localhost:30000/v1
```

A different model per role: one `--role` per role, with the same fields as
`--agent` in `python -m agent_cap.agents`. Quote the specs so that
`${OPENAI_API_KEY}` is resolved at run time and never written to the job file;
`use_streaming=true` lets each request's TTFT be measured.

```bash
python -m agent_cap.harbor run --benchmark nl2repo --dataset datasets/nl2repo --task-names math-verify \
  --strategy plan-execute \
  --role 'planner=name=gpt-5.4,base_url=https://api.openai.com/v1,api_key=${OPENAI_API_KEY},use_streaming=true' \
  --role 'executor=name=Qwen/Qwen3.5-4B,base_url=http://localhost:30000/v1,api_key=EMPTY,use_streaming=true'
```

### 6. Supervisor with workers

The supervisor delegates each round to a worker by name (`DELEGATE coder: ...`)
and stops with `DONE: ...`; workers have the tools:

```bash
python -m agent_cap.harbor run --benchmark nl2repo --dataset datasets/nl2repo --task-names math-verify \
  --strategy supervisor --ak 'role_names=[supervisor,coder,tester]' \
  --model Qwen/Qwen3.5-4B --base-url http://localhost:30000/v1
```

`--strategy sequential` and custom strategies work the same way; see
[docs/agents/strategies.md](docs/agents/strategies.md).

### 7. Other harnesses, same flags

Any agent Harbor ships runs the same way; like DSH, it runs in the task
container, and the metering proxy (on by default for these) records each model
request.

```bash
# Codex
python -m agent_cap.harbor run --harness codex \
  --benchmark nl2repo --dataset datasets/nl2repo --task-names math-verify --model openai/gpt-5.4

# Claude Code
python -m agent_cap.harbor run --harness claude-code \
  --benchmark nl2repo --dataset datasets/nl2repo --task-names math-verify --model anthropic/claude-sonnet-4-5
```

Also `openhands`, `mini-swe-agent`, `gemini-cli`, `opencode`, ...
(`harbor run --help` lists them).

### 8. Results

Each run writes `results/harbor/<job>/`:

| File | Contents |
|---|---|
| `results.jsonl` | one row per task: reward (fraction of hidden tests passed), pass/fail, tokens per role, per-request TTFT/decode time |
| `metrics_*.json`, `metadata_*.json`, `detailed-results_*.jsonl`, `output-data_*.jsonl` | the four TEAS files, as read by TEASBench's cost scripts |

Harbor's own logs (trajectories, verifier output) are in `jobs/<job>/`; browse
them with `harbor view jobs`. To re-export a finished job:
`python -m agent_cap.harbor export --job-dir jobs/<job> --dataset nl2repo --out results/<job>`.

An agent can download a reference implementation from PyPI instead of writing
one; scan trajectories before reporting NL2Repo results
([docs/harbor.md](docs/harbor.md#nl2repo)).

### IMO-AnswerBench on Harbor

```bash
python -m agent_cap.harbor adapt imo-answerbench --out datasets/imo-answerbench --limit 25
python -m agent_cap.harbor run --benchmark imo-answerbench --dataset datasets/imo-answerbench \
  --model Qwen/Qwen3.5-4B --base-url http://localhost:30000/v1 --concurrency 4
```

The agent gets a `python` tool on a Jupyter kernel in the task container and
AgentCAP's IMO prompts; the grader checks the last `\boxed{}` with math-verify,
then an LLM judge if `IMO_JUDGE_BASE_URL`, `IMO_JUDGE_API_KEY` and
`IMO_JUDGE_MODEL` (or `OPENROUTER_API_KEY`) are set. A TEASBench agent config
can replace the model flags: `--teas-agent-yaml pipeline/configs/agents/imo-answerbench_gptoss_sglang.yaml`.
For Harbor's own agents, adapt with `--answer-file-instruction` so they write
their answer to `/workspace/answer.txt`.

### Flags

| Flag | Meaning |
|---|---|
| `--benchmark`, `--dataset` | `nl2repo` or `imo-answerbench`, and the directory written by `adapt` |
| `--strategy` | AgentCAP strategy: `single` (default), `plan-execute`, `supervisor`, `sequential` |
| `--model` | `Qwen/Qwen3.5-4B` (with `--base-url`), `openai/gpt-5.4`, `anthropic/...`, `deepseek/deepseek-chat`, ... |
| `--base-url` | inference endpoint (e.g. a local vLLM server) |
| `--api-key-name` | name of the env var holding the API key (default: the provider's, e.g. `OPENAI_API_KEY`); the value is never written to files |
| `--role ROLE=k=v,...` | one model per role instead of `--model` (repeatable) |
| `--teas-agent-yaml` | a TEASBench agent config instead of `--model` / `--role` |
| `--harness` | `capagent` (default: AgentCAP's strategies), `dsh`, `codex`, `claude-code`, any Harbor agent, or `module:Class` |
| `--api-format`, `--effort` | inference API format, reasoning effort |
| `--ak KEY=VALUE`, `--ae KEY=VALUE` | harness option (e.g. `role_names=[...]`, `teammates=2`), env var for the harness |
| `--meter` / `--no-meter` | metering proxy for harnesses that run in the container (default on for Harbor's agents) |
| `--task-names`, `--n-tasks`, `--concurrency`, `--attempts`, `--env` | task filter, task count, concurrent trials, attempts per task, Harbor environment |

These are Harbor's own `harbor run` flags, so plain Harbor runs the same agents:
`harbor run -p datasets/nl2repo --agent agent_cap.harbor:CAPAgent --model ... --ak strategy=plan-execute --ak tool_backend=shell`
(also `agent_cap.harbor.metered:codex`, `agent_cap.harbor:DSHAgent`).
Details, metering, grading rules and caveats: [docs/harbor.md](docs/harbor.md).

## Native runner (without Harbor)

`python -m agent_cap.agents` runs the same strategies directly on this machine,
without task containers: tool-use benchmarks against an MCP server, IMO via
Harmony, and the other datasets in [docs/datasets.md](docs/datasets.md).

### Tool-use benchmark — MCP-ATLAS on Qwen3.5

Serve the model with tool-call parser:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --port 30000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Start the MCP tool server (in another terminal):

```bash
bash mcp-server/start.sh
```

Run the benchmark:

```bash
python -m agent_cap.agents \
  --strategy single \
  --model Qwen/Qwen3.5-4B \
  --base-url http://localhost:30000/v1 \
  --api-key dummy \
  --dataset mcp-atlas \
  --evaluator gtfa \
  --tool-backend mcp --mcp-server-url http://localhost:1984 \
  --num-tasks 30 \
  --output-dir results/mcp
```

### Multi-agent — plan-execute on one engine

Two distinct roles (`planner` writes a plan, `executor` follows it with tools),
both hitting the same vLLM endpoint. Each role keeps its own conversation
history and system prompt — only the engine connection is shared.

```bash
python -m agent_cap.agents \
  --strategy plan-execute \
  --agent planner=name=Qwen/Qwen3.5-4B,base_url=http://localhost:30000/v1,api_key=dummy \
  --agent executor=name=Qwen/Qwen3.5-4B,base_url=http://localhost:30000/v1,api_key=dummy \
  --task "Find the largest prime under 100. Use tools if helpful." \
  --tool-backend mcp --mcp-server-url http://localhost:1984
```

Supervisor with two workers — the supervisor delegates each turn to a worker
of its choice and stops when it emits `DONE: <answer>`:

```bash
python -m agent_cap.agents \
  --strategy supervisor \
  --agent supervisor=name=Qwen/Qwen3.5-4B,base_url=http://localhost:30000/v1,api_key=dummy \
  --agent researcher=name=Qwen/Qwen3.5-4B,base_url=http://localhost:30000/v1,api_key=dummy \
  --agent writer=name=Qwen/Qwen3.5-4B,base_url=http://localhost:30000/v1,api_key=dummy \
  --task "Write a one-paragraph summary of the 2024 EU AI Act, then compute its character count." \
  --tool-backend mcp --mcp-server-url http://localhost:1984
```

For a YAML version (a 1-line `roles:` mapping fans out one endpoint to N
roles), see [configs/agents_pool.yaml](configs/agents_pool.yaml). Other
strategies — `sequential`, etc. — work the same way; see
[docs/agents/strategies.md](docs/agents/strategies.md).

### Reasoning benchmark — IMO AnswerBench via Harmony (gpt-oss)

For models that need client-side Harmony token decoding (gpt-oss family +
fine-grained `reasoning_effort`), use the bundled config:

```bash
python -m agent_cap.agents --config configs/agents_imo_gptoss_sglang.yaml
```

That config sets `protocol: harmony`, `engine: sglang`, and points at a local
sglang server. See [docs/agents/protocols.md](docs/agents/protocols.md) for
when to opt into Harmony vs. the default OpenAI-compatible path.

### Mock mode

No LLM, no GPU — sanity check the framework:

```bash
python -m agent_cap.agents --mock --strategy plan-execute --task "compute 1+1"
```

## Documentation

| File | Topic |
|---|---|
| [docs/datasets.md](docs/datasets.md) | All 5 datasets (imo, mcp-atlas, medagent, finance, swe-bench) — command per dataset |
| [docs/mcp-server.md](docs/mcp-server.md) | Self-host MCP server without Docker (`mcp-server/start.sh`) |
| [docs/RUN_NO_DOCKER.md](docs/RUN_NO_DOCKER.md) | SWE-Bench Lite via Modal (no Docker needed locally) |
| [docs/agents/concepts.md](docs/agents/concepts.md) | Agent / Strategy / Protocol / Tool / Evaluator |
| [docs/agents/cli.md](docs/agents/cli.md) | All CLI flags and precedence |
| [docs/agents/yaml.md](docs/agents/yaml.md) | YAML config: pool, roles, replicas, includes |
| [docs/agents/strategies.md](docs/agents/strategies.md) | Built-in strategies: single, plan-execute, supervisor, sequential |
| [docs/agents/protocols.md](docs/agents/protocols.md) | LLM protocols (openai default, harmony opt-in, mock) |
| [docs/agents/serving.md](docs/agents/serving.md) | vLLM / sglang launch flags per model family |
| [docs/agents/extending.md](docs/agents/extending.md) | Recipes for custom strategies, protocols, tools, evaluators |
| [docs/harbor.md](docs/harbor.md) | Any harness on Harbor (CAPAgent, DeepSeek Harness, Codex, Claude Code, ...): one set of flags, metering proxy, IMO-AnswerBench and NL2Repo, TEAS export |

## Citation

If you use AgentCARD, please cite:

```bibtex
@misc{jiang2026agentcard,
  title         = {Specialize Roles, Mix Deployments: Pushing the Cost-Accuracy Frontier of {LLM} Agent Teams},
  author        = {Jiang, Yinsicheng and Cheng, Liang and Huang, Yeqi and Zhao, Yufan and Lu, Zhan and Dong, Li and Li, Wenda and Ponti, Edoardo and Mai, Luo},
  year          = {2026},
  eprint        = {2606.20629},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MA},
  url           = {https://arxiv.org/abs/2606.20629}
}
```

## Acknowledgement

This project is supported by the Advanced Research and Invention Agency (ARIA)'s
grant *"Scaling Compute: AI at 1/1000th the cost. Technical Area 4 Benchmarking"*.
