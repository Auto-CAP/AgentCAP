# CLI reference

Invocation:

```bash
python -m agent_cap.agents [flags]
```

## Discovery flags

| Flag | Effect |
|---|---|
| `--list-strategies` | Print built-in + plugin strategies, exit. |
| `--list-protocols`  | Print registered LLM protocols, exit. |

## Strategy

| Flag | Default | Notes |
|---|---|---|
| `--strategy NAME` | `plan-execute` | One of `--list-strategies`. |
| `--max-turns N` | `8` | Cap on tool-use turns per agent. |
| `--sequence r1,r2,...` | (config order) | Only for `--strategy sequential`. |

## Agents and roles

Two YAML layouts are supported. See [yaml.md](yaml.md) for details. From the
command line:

| Flag | Repeatable | Notes |
|---|---|---|
| `--agent ROLE=k=v,k=v` | yes | Inline. ROLE is the agent pool key AND the role name unless you also use `--role`. |
| `--agents-file PATH`   | yes | YAML file with only `agents:` (+ optional `defaults:`/`include:`). Later files override earlier. |
| `--agents-glob PATTERN` | yes | Shell glob, expanded and sorted. |
| `--role ROLE=AGENT`    | yes | Bind strategy role to a pool key. Lets two roles share one endpoint. |
| `--share-state ROLE1,ROLE2` | yes | Roles in the group share one Agent instance (conversation history). |

Supported keys inside `--agent` and `--role` values:

```
name           model id / served-model-name
base_url       OpenAI-compatible base URL (must include /v1 for hosted)
api_key        bearer; any non-empty string is fine for self-hosted
max_tokens     int, default 4096
temperature    float, default 0.0
use_streaming  bool, default false
protocol       explicit override: openai | harmony | mock | ...
engine         serving-engine variant for the protocol (e.g. vllm | sglang for harmony)
system_prompt  optional system message
```

## Tasks

| Flag | Notes |
|---|---|
| `--task "..."` | Single prompt. |
| `--task-file PATH.jsonl` | Each line: `{"task_id": ..., "user_prompt": ..., ...}`. |
| `--dataset NAME` | Uses `agent_cap.runner.unified_runner._load_dataset_tasks`. Known names: `imo-answerbench`, `mcp-atlas`, `swe-bench-pro`, `medagentbench`, `financebench`. |
| `--num-tasks N` | Cap dataset to N tasks (0 = all). |

## Tools

| Flag | Notes |
|---|---|
| `--demo-tools` | Built-in `calc` + `echo` for smoke tests. |
| `--tool-backend NAME` | Real backends: `mcp`, `math-python`, or `demo`. |
| `--mcp-server-url URL` | Required when `--tool-backend mcp`. |
| `--no-tools` | Force-disable tools even if config provides them. |

## Evaluation

| Flag | Notes |
|---|---|
| `--evaluator NAME` | One of `--list-strategies`-equivalent for evaluators: `exact`, `gtfa`, `imo`, `imo-answerbench`, `llm-judge` / `judge`, `none`. |
| `--judge "k=v,k=v"` | Config for `llm-judge` (or as fallback for other evaluators). Keys: `name`, `base_url`, `api_key`, `temperature`, `max_tokens`, `timeout_s`, `system_prompt`, `user_template`, `decision_field`, `score_field`, `extract`. |

## Output and resume

| Flag | Notes |
|---|---|
| `--output PATH.json` | Single-file dump when not using `--output-dir`. |
| `--output-dir DIR`   | Writes `results.jsonl` (one row per task) and `output-data.jsonl`. |
| `--resume` | Skip `task_id`s already present in `<output-dir>/results.jsonl`. |
| `-v / -vv` | -v prints per-task summary; -vv adds per-turn trace. |

## Runtime

| Flag | Notes |
|---|---|
| `--mock` | Deterministic offline LLM. No API key, no network. |
| `--load-module DOTTED.PATH` | Import a module so its `@register_*` decorators fire. Repeatable. |
| `--config PATH.yaml` | Main YAML config. CLI flags override fields in it. |

## Precedence

When a setting can be supplied in multiple places:

```
CLI flag  >  --config YAML  >  --agents-file / --agents-glob  >  YAML `include:`
```

For per-agent fields specifically:

```
CLI --agent (inline)  >  CLI --role (mapping)  >  --config agents:  >  --agents-file agents:  >  defaults:
```
