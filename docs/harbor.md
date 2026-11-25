# Running agent harnesses on Harbor

`agent_cap.harbor` runs any agent harness on
[Harbor](https://github.com/harbor-framework/harbor): AgentCAP's own strategies
(`CAPAgent`), DeepSeek Harness (`DSHAgent`), and every agent Harbor ships
(Codex, Claude Code, OpenHands, mini-swe-agent, Gemini CLI, OpenCode, ...).
Harbor owns the task format, containers, trial scheduling, retries and grading.
AgentCAP adds benchmark adapters, per-request metering and the TEAS output
files, the same for every harness.

```
TEASBench / user ─> python -m agent_cap.harbor run --harness H --model M ─> harbor run -c job.yaml
                    (or plain: harbor run --agent <H> --model M ...)            │
                                                                                 ▼
  capagent: strategy loop on the host ──────── tool calls via exec ──> task container
  dsh / codex / claude-code / ...:  harness runs in the task container
        └─ model requests ─> AgentCAP meter (host proxy) ─> model endpoint
                                                                                 │
  task container ──(/workspace artifact)──> verifier container ──> reward.json   │
  TEAS files + results.jsonl <── agent_cap.harbor.export_teas <── jobs/<job>/ <──┘
```

## Install

Harbor needs Python 3.12 or newer.

```bash
pip install -e ".[harbor]"
```

## One interface for every harness

Every harness is a Harbor agent and takes Harbor's model connection fields. The
AgentCAP CLI, plain `harbor run` and the job YAML spell them as follows:

| `python -m agent_cap.harbor run` | `harbor run` | job YAML (`agents:` entry) | Meaning |
|---|---|---|---|
| `--harness H` | `--agent H` | `name` / `import_path` | which harness |
| `--model` | `--model` | `model_name` | `openai/gpt-5.4`, `anthropic/claude-sonnet-4-5`, `deepseek/deepseek-chat`, or any id served at `--base-url` |
| `--base-url` | `--base-url` | `base_url` | inference endpoint |
| `--api-key-name` (alias `--api-key-env`) | `--api-key-name` | `model_api_key_name` | *name* of the env var holding the key; the value never reaches a file |
| `--api-format` | `--api-format` | `model_api_format` | `openai_chat_completions`, `openai_responses`, `anthropic_messages`, ... |
| `--effort` | `--effort` | `kwargs.reasoning_effort` | reasoning effort, where the harness supports it |
| `--ak KEY=VALUE` | `--ak KEY=VALUE` | `kwargs` | harness option (e.g. `teammates=2`, `version=0.172.0`) |
| `--ae KEY=VALUE` | `--ae KEY=VALUE` | `env` | env var for the harness |

Harnesses:

| `--harness` | Harbor agent | Runs | Per-request metering (`agentcard.metering`) |
|---|---|---|---|
| `capagent` | `agent_cap.harbor:CAPAgent` | strategy loop on the host, tools in the container | `agentcap-host`: AgentCAP's clients, per role |
| `dsh` | `agent_cap.harbor:DSHAgent` | in the container | `dsh-session`: DSH's session logs, per role (lead, teammates); `--meter` adds the proxy view |
| `codex`, `claude-code`, `openhands`, `mini-swe-agent`, ... (all Harbor agent names) | `agent_cap.harbor.metered:<name>` (`--no-meter`: Harbor's own) | in the container | `meter-proxy`: AgentCAP's host proxy |
| `module:Class` | that class, metered unless `--no-meter` | — | `meter-proxy` if the class takes `--base-url` |

Without metering (`--no-meter`, or a harness that cannot take `--base-url`), the
exporter falls back to `harbor-context`: Harbor's token totals and the per-step
tokens of the agent's ATIF trajectory, without TTFT or decode time.

## Run

```bash
# Codex, metered
python -m agent_cap.harbor run --harness codex \
  --benchmark nl2repo --dataset datasets/nl2repo --model openai/gpt-5.4

# Claude Code
python -m agent_cap.harbor run --harness claude-code \
  --benchmark nl2repo --dataset datasets/nl2repo --model anthropic/claude-sonnet-4-5

# DeepSeek Harness, Agent Team of a lead + 2 teammates
python -m agent_cap.harbor run --harness dsh \
  --benchmark nl2repo --dataset datasets/nl2repo --model deepseek/deepseek-chat --ak teammates=2

# AgentCAP plan-execute; every role on one self-hosted endpoint
python -m agent_cap.harbor run --harness capagent --strategy plan-execute \
  --benchmark nl2repo --dataset datasets/nl2repo \
  --model Qwen/Qwen3.5-27B --base-url http://localhost:30000/v1

# any OpenAI-compatible endpoint for a container harness
python -m agent_cap.harbor run --harness mini-swe-agent \
  --benchmark nl2repo --dataset datasets/nl2repo \
  --model openai/Qwen/Qwen3.5-27B --base-url http://localhost:30000/v1 --api-key-name VLLM_API_KEY
```

`run` writes `jobs/<job>.yaml`, calls `harbor run -c <job>.yaml --yes`, and
exports to `results/harbor/<job>/`. `job` only writes the YAML. The same agents
work from plain Harbor, followed by `export`:

```bash
harbor run -p datasets/nl2repo --agent agent_cap.harbor.metered:codex --model openai/gpt-5.4
harbor run -p datasets/nl2repo --agent agent_cap.harbor:DSHAgent --model deepseek/deepseek-chat --ak teammates=2
harbor run -p datasets/nl2repo --agent agent_cap.harbor:CAPAgent --model Qwen/Qwen3.5-27B \
  --base-url http://localhost:30000/v1 --ak strategy=plan-execute --ak tool_backend=shell
python -m agent_cap.harbor export --job-dir jobs/<job> --dataset nl2repo --out results/<job>
```

Plain Harbor does not apply the benchmark presets (tool backend, step limit,
system prompt, answer file) that `python -m agent_cap.harbor run` adds for
CAPAgent and DSH; pass them with `--ak`, as above.

Where the endpoint must be reachable from: CAPAgent and the metering proxy call
it from this host, so `localhost` works for them. An unmetered container
harness calls it from the task container.

## Metering proxy

`agent_cap.harbor.metered:<agent>` is the Harbor agent `<agent>` with one
change: before it starts, its `--base-url` points at `agent_cap.harbor.meter`,
a reverse proxy on this host that forwards each request unchanged to the
endpoint the agent would have used and records:

* `ttft_s`: request received to first content, reasoning or tool-call token;
* `decode_s`: first to last token; `latency_s`: request to end of response;
* tokens from the response's usage block (input includes cached tokens), tool
  calls, model, HTTP status.

It understands OpenAI chat completions, the Responses API (SSE and WebSocket,
Codex's default transport) and Anthropic Messages. Each trial gets its own route
(`http://<host>:<port>/<token>/...`), so concurrent trials stay apart. The proxy
runs in its own thread and event loop, so Harbor's work does not add latency to
the measurements. Each trial writes `agent/meter_result.json`; the exporter turns
it into the same row CAPAgent writes. Responses-API warm-ups (`generate: false`,
which Codex sends once per connection) are not model turns: they are left out of
`turn_stats` and reported under `agentcard.warmup_requests`.

The task container must reach the proxy:

* Harbor's `docker` environment: the container's default gateway, found from
  `/proc/net/route`, needs no setup.
* Other environments (Modal, Daytona, ...): set `AGENTCAP_METER_HOST` (or
  `--meter-host`) to an address of this host the sandbox can reach;
  `AGENTCAP_METER_PORT` fixes the port (default: a free one).
* With a restricted network policy, allow that host with
  `--allow-agent-host` (Harbor).

If the proxy cannot be used (no address, or the agent cannot take
`--base-url`), the agent runs unmetered and `meter_result.json` says why.

## CAPAgent

`CAPAgent` runs any registered AgentCAP strategy (`single`, `plan-execute`,
`supervisor`, `sequential`) with the existing LLM clients. The loop and every
model request stay on the host, so TTFT, decode time and tokens are measured per
request exactly as with `python -m agent_cap.agents`. Only tool calls run in the
task container:

* `shell`: `bash` and `submit` through AgentCAP's `ToolExecutor`;
  `str_replace_editor` through SWE-agent's official editor script
  (`agent_cap/backends/swe_agent_tools`), uploaded into the container.
* `math-python`: the `python` tool on a persistent Jupyter kernel inside the
  container (`pykernel_server.py`). Outputs match `MathPythonBackend`
  (preload, last-expression print, error format, timeout and restart).

The team comes from Harbor's connection (every role of the strategy uses
`--model`; `--ak role_names=[a,b]` and `--ak 'role_options={max_tokens: 4096}'`
adjust it) or from explicit roles: `--role` / `--teas-agent-yaml` on the CLI,
`roles:` in the job YAML. Each trial writes `agent/capagent_result.json`, the
row the AgentCAP CLI writes to `results.jsonl` (per-role usage, `turn_stats`).

## DeepSeek Harness (`--harness dsh`)

`DSHAgent` runs [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
in the task container:

1. **setup**: installs Node 24 with nvm if the container has no Node >= 22
   (and curl first if needed), then `npm install @deepseek-ai/dsh@<version>`
   under `/opt/agentcard-dsh`. Skipped when that version is already installed.
2. **run**: writes a profile to `/logs/agent/dsh-home/profiles/agentcard/` and
   runs `dsh --profile agentcard --json "<instruction>"` in `/workspace`, with
   `DSH_PERMISSION_MODE=danger-full-access` and telemetry off.
   * the model is Harbor's connection: `--api-format` selects DSH's route
     (`openai_chat_completions` -> `openai-completions`, `openai_responses`,
     `anthropic_messages`); `--ak protocol=deepseek-messages` selects DSH's own
     DeepSeek adapter;
   * bundles: `dsh-base` + `dsh-headless`; with `--ak teammates=K`, also
     `dsh-experimental-agent-team-profile` and `maxMembers = maxActiveSubagents = K`;
   * session titles, the DeepSeek account route, web search and fetch, workflow
     delegation and package inventory are disabled;
   * the API key reaches `dsh` as an environment variable of the exec call;
     the profile names only the variable.
3. In team mode the prompt asks for exactly K teammates, a shared task board,
   file ownership in the shared workspace, and `wait_agent` on every teammate
   before finishing (DSH stops teammates when the lead finishes). Override it
   with `--ak team_prompt=...` (`{teammates}` and `{workdir}` are filled in).
4. **after the run**: every agent's `session.v4.jsonl` is parsed into the same
   row CAPAgent writes. Usage is split by role (`lead`, teammates by name), and
   each request's TTFT and decode time come from DSH's stream timestamps,
   measured in the container. `agentcard.team` records teammates expected,
   spawned and completed. With `--meter`, `agentcard.meter` adds the host
   proxy's view of the same requests.

Other DSH options (`--ak`): `context_window` (131072), `max_tokens` (16384),
`dsh_version` / `version` (0.2.0-rc.2), `compat`, `reasoning_efforts`. In DSH
0.2.0-rc.2, `spawn_teammate` has no model argument, so a team is homogeneous.

## Benchmarks

### IMO-AnswerBench

```bash
python -m agent_cap.harbor adapt imo-answerbench --out datasets/imo-answerbench --limit 100
```

* `instruction.md` is AgentCAP's user prompt, unchanged; the dataset system
  prompt is passed to `CAPAgent` by the `imo-answerbench` preset.
* `--limit N` keeps the first N problems in dataset order, as AgentCAP does.
* `CAPAgent` and `DSHAgent` write their final message to `/workspace/answer.txt`;
  `tests/grade_imo.py` applies AgentCAP's `imo` evaluator to it: last
  `\boxed{}`, then math-verify, then an LLM judge.
* Judge: set `IMO_JUDGE_BASE_URL`, `IMO_JUDGE_API_KEY`, `IMO_JUDGE_MODEL`
  (or `OPENROUTER_API_KEY`) in the environment of `harbor run`. Harbor passes
  them to the verifier through `[verifier.env]`; they are never written into
  task or job files. A TEASBench agent YAML's `judge` block maps onto these.
* Harbor's own agents do not know to write the answer file. Generate with
  `--answer-file-instruction` for them (the CLI warns otherwise); it changes the
  prompt, so compare such runs only with each other.

### NL2Repo

```bash
git clone https://github.com/multimodal-art-projection/NL2RepoBench
python -m agent_cap.harbor adapt nl2repo --source NL2RepoBench --out datasets/nl2repo --require-commit 781a1da
```

* Agent container: Python 3.12 with only `start.md` in `/workspace`;
  instruction is upstream's OpenHands prompt. Grading is file-based, so every
  harness runs it unchanged.
* Verifier: a separate container built from the official evaluation image
  `ghcr.io/multimodal-art-projection/nl2repobench/<task>:1.0` (amd64). Its
  `/workspace` holds the hidden tests and the original `pyproject.toml`, so the
  image is used only for grading: `tests/Dockerfile` moves that content to
  `/opt/nl2repo/reference`, and Harbor hands the agent's `/workspace` over as an
  artifact. Hidden tests never exist in the agent's container.
* `tests/nl2repo_eval.py` reproduces upstream `post_processor.py`: delete
  packaging files and the task's test paths from the submission, overlay it on
  the reference, run `test_commands.json` (shlex-split, no shell,
  `PYTHONPATH=/workspace`), and score `passed / test_case_count` (capped at 1)
  from the pytest output with upstream's regexes.
* `reward.json` holds `reward` (upstream score), `passed`, `failed`, `errors`,
  `total`, `pass_all`, `junit_passed` (JUnit cross-check) and
  `reward_hardened`, the score after removing `conftest.py`,
  `sitecustomize.py`, `usercustomize.py` and `.pth` files outside the hidden
  test paths. Upstream keeps those files, so an agent could otherwise pass tests
  with a pytest hook; compare the two scores.
* The agent container has network access, as in upstream. An agent can download
  the reference package from PyPI; the end-to-end tests do exactly that and
  score 192/192 on `math-verify`. Scan trajectories for this before reporting
  results.

## Results

`run` and `export` write `results/harbor/<job>/`:

| File | Contents |
|---|---|
| `results.jsonl` | one row per trial: reward, pass/fail, tokens per role, per-request TTFT/decode time, `agentcard.metering` |
| `metrics_*.json`, `metadata_*.json`, `detailed-results_*.jsonl`, `output-data_*.jsonl` | the four TEAS files, as read by TEASBench's cost scripts |

`metadata_*.json` gains a `harbor` section (Harbor version, job, environment,
agent and import path, model, metering sources, strategy, roles). TEAS hardware
and engine fields still come from the `TEAS_*` environment variables, as for the
existing CLI. `--teas-model` overrides the model name in the TEAS files.

## Original flags

These still work and write the same jobs as before:

* CAPAgent: `--role 'ROLE=name=...,base_url=...,api_key=${VAR}'` (repeatable;
  quote it so `${VAR}` is expanded at run time, not written to the job file) or
  `--teas-agent-yaml <TEASBench agent YAML>`; `--strategy`, `--max-turns`,
  `--task-timeout-s`.
* DSH: `--dsh-model`, `--dsh-base-url`, `--dsh-api-key-env`, `--dsh-protocol`,
  `--dsh-teammates`, `--dsh-context-window`, `--dsh-max-tokens`,
  `--dsh-reasoning-effort`, `--dsh-version`.

## Tests

```bash
pytest tests/test_harbor_integration.py tests/test_harbor_dsh.py tests/test_harbor_harness.py
AGENTCAP_DOCKER_TESTS=1 AGENTCAP_IMO_TEST_IMAGE=<built IMO task image> pytest tests/test_harbor_integration.py -k container
```

## Known gaps

* MCP-Atlas and SWE-bench Lite adapters are not written yet.
* EIDF (Kubernetes without pod RBAC) needs a custom Harbor environment;
  Harbor's `gke` environment requires gcloud.
* The proxy meters what it can parse: harnesses on other protocols (Gemini's
  `generateContent`, Bedrock) are forwarded and timed but have no token fields;
  their totals come from Harbor's context.
* Roles inside a third-party harness (e.g. Claude Code subagents) are not
  visible to the proxy; requests are split by model only.
* `agent_cap/backends/tool_executor.py`'s own `str_replace` builds
  `python3 -c "$'...'"`, which bash does not expand inside double quotes, so it
  fails with a SyntaxError. `CAPAgent` does not use it (it calls the official
  editor script); the native runners that do are unchanged.
