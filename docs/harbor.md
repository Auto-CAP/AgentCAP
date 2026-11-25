# Running AgentCAP on Harbor

`agent_cap.harbor` runs AgentCAP configurations on
[Harbor](https://github.com/harbor-framework/harbor). Harbor owns the task
format, containers, trial scheduling, retries and grading. AgentCAP keeps the
team strategies, per-request metering and the TEAS output files.

```
TEASBench / user ──> python -m agent_cap.harbor run ──> harbor run -c job.yaml
                                                           │
                       CAPAgent (host) ── chat/completions ──> engine (vLLM / SGLang / API)
                           │  tool calls via environment.exec
                           ▼
                     task container ──(/workspace artifact)──> verifier container ──> reward.json
                                                           │
  TEAS files + results.jsonl <── agent_cap.harbor.export_teas <── jobs/<job>/
```

## Install

Harbor needs Python 3.12 or newer.

```bash
pip install -e ".[harbor]"
```

## Components

| Harbor concept | AgentCAP implementation |
|---|---|
| Task directory | `agent_cap/harbor/adapters/` (`imo_answerbench.py`, `nl2repo.py`, templates) |
| Agent (`import_path`) | `agent_cap.harbor.agent:CAPAgent`; `agent_cap.harbor.dsh:DSHAgent` (DeepSeek Harness) |
| Tools inside the environment | `agent_cap/harbor/tools.py` (`shell`, `math-python`) |
| Job config | `agent_cap/harbor/jobspec.py`, `python -m agent_cap.harbor job` |
| Trial outputs | `agent_cap/harbor/export_teas.py` |

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

Each trial writes `agent/capagent_result.json`, the same row the AgentCAP CLI
writes to `results.jsonl` (per-role usage, `turn_stats`). The exporter joins it
with Harbor's verifier reward and calls `write_teas_outputs`, so `metadata_*`,
`metrics_*`, `detailed-results_*` and `output-data_*` have the fields
TEASBench's cost scripts read. `metadata_*.json` gains a `harbor` section
(Harbor version, job, environment, agent, strategy, roles).

## DeepSeek Harness (`--harness dsh`)

`agent_cap.harbor.dsh:DSHAgent` runs
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) in the task
container instead of an AgentCAP strategy:

1. **setup**: installs Node 24 with nvm if the container has no Node >= 22
   (and curl first if needed), then `npm install @deepseek-ai/dsh@<version>`
   under `/opt/agentcard-dsh`. Skipped when that version is already installed,
   so an image with DSH baked in starts faster.
2. **run**: writes a profile to `/logs/agent/dsh-home/profiles/agentcard/` and
   runs `dsh --profile agentcard --json "<instruction>"` in `/workspace`, with
   `DSH_PERMISSION_MODE=danger-full-access` and telemetry off.
   * bundles: `dsh-base` + `dsh-headless`; with `--dsh-teammates K`, also
     `dsh-experimental-agent-team-profile` and `maxMembers = maxActiveSubagents = K`;
   * the model route uses the pi-ai adapter for `openai-*` and
     `anthropic-messages`, and DSH's own adapter for `deepseek-messages`;
   * session titles, the DeepSeek account route, web search and fetch, workflow
     delegation and package inventory are disabled;
   * the API key reaches `dsh` as an environment variable of the exec call;
     the profile names only the variable.
3. In team mode the prompt asks for exactly K teammates, a shared task board,
   file ownership in the shared workspace, and `wait_agent` on every teammate
   before finishing (DSH stops teammates when the lead finishes). Override it
   with the `team_prompt` agent kwarg (`{teammates}` and `{workdir}` are filled in).
4. **after the run**: every agent's `session.v4.jsonl` is parsed into the same
   row CAPAgent writes. Usage is split by role (`lead`, teammates by name), and
   each request's TTFT and decode time come from DSH's stream timestamps.
   Because DSH runs in the container, these are measured there, not on the host.
   `agentcard.team` records teammates expected, spawned and completed.

For IMO-AnswerBench, DSH's final message is written to `/workspace/answer.txt`
and graded as for CAPAgent. The model endpoint must be reachable from the
container. In DSH 0.2.0-rc.2, `spawn_teammate` has no model argument, so a team
is homogeneous.

## Benchmarks

### IMO-AnswerBench

```bash
python -m agent_cap.harbor adapt imo-answerbench --out datasets/imo-answerbench --limit 100
```

* `instruction.md` is AgentCAP's user prompt, unchanged; the dataset system
  prompt is passed to `CAPAgent` by the `imo-answerbench` preset.
* `--limit N` keeps the first N problems in dataset order, as AgentCAP does.
* `CAPAgent` writes its final message to `/workspace/answer.txt`;
  `tests/grade_imo.py` applies AgentCAP's `imo` evaluator to it: last
  `\boxed{}`, then math-verify, then an LLM judge.
* Judge: set `IMO_JUDGE_BASE_URL`, `IMO_JUDGE_API_KEY`, `IMO_JUDGE_MODEL`
  (or `OPENROUTER_API_KEY`) in the environment of `harbor run`. Harbor passes
  them to the verifier through `[verifier.env]`; they are never written into
  task or job files. A TEASBench agent YAML's `judge` block maps onto these.
* Other Harbor agents do not know to write the answer file. Generate with
  `--answer-file-instruction` for them; it changes the prompt, so leave it off
  for TEAS parity runs.

### NL2Repo

```bash
git clone https://github.com/multimodal-art-projection/NL2RepoBench
python -m agent_cap.harbor adapt nl2repo --source NL2RepoBench --out datasets/nl2repo --require-commit 781a1da
```

* Agent container: Python 3.12 with only `start.md` in `/workspace`;
  instruction is upstream's OpenHands prompt.
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
  the reference package from PyPI; the end-to-end test does exactly that and
  scores 192/192 on `math-verify`. Scan trajectories for this before reporting
  results.

## Run

With a TEASBench agent YAML:

```bash
python -m agent_cap.harbor run \
  --benchmark imo-answerbench --dataset datasets/imo-answerbench \
  --teas-agent-yaml ../TEASBench/pipeline/configs/agents/imo-answerbench_gptoss_sglang.yaml \
  --concurrency 4 --env docker
```

With explicit roles (any AgentCAP strategy):

```bash
python -m agent_cap.harbor run \
  --benchmark nl2repo --dataset datasets/nl2repo --task-names math-verify \
  --strategy plan-execute \
  --role 'planner=name=gpt-5.4,base_url=https://api.openai.com/v1,api_key=${OPENAI_API_KEY}' \
  --role 'executor=name=Qwen/Qwen3.5-27B,base_url=http://127.0.0.1:8000/v1,api_key=EMPTY'
```

Quote role specs so that `${VAR}` reaches the job file unexpanded; `CAPAgent`
expands it at run time. `run` writes `jobs/<job>.yaml`, calls
`harbor run -c <job>.yaml --yes`, and exports to `results/harbor/<job>/`. Use
`job` to only write the YAML, and `export` to re-export a finished job:

```bash
python -m agent_cap.harbor export --job-dir jobs/<job> --dataset imo-answerbench --out results/<job>
```

TEAS hardware and engine fields still come from the `TEAS_*` environment
variables, as for the existing CLI.

## Tests

```bash
pytest tests/test_harbor_integration.py
AGENTCAP_DOCKER_TESTS=1 AGENTCAP_IMO_TEST_IMAGE=<built IMO task image> pytest tests/test_harbor_integration.py -k container
```

## Known gaps

* MCP-Atlas and SWE-bench Lite adapters are not written yet.
* EIDF (Kubernetes without pod RBAC) needs a custom Harbor environment;
  Harbor's `gke` environment requires gcloud.
* `agent_cap/backends/tool_executor.py`'s own `str_replace` builds
  `python3 -c "$'...'"`, which bash does not expand inside double quotes, so it
  fails with a SyntaxError. `CAPAgent` does not use it (it calls the official
  editor script); the native runners that do are unchanged.
