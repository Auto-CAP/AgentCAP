# Reproducing the gpt-oss-120b benchmarks without Docker

This guide reproduces the same setup that previously ran on Kubernetes:

- **MCP-ATLAS**: 60 tasks, agentic tool-use over 22 MCP servers
- **SWE-bench Lite**: 100 tasks, code-edit + test on real Python repos

The original pipeline relied on Kubernetes pods + Docker images. This guide
covers a host that **does not have Docker** but does have:

- Python 3.11+
- Node.js 18+ and `npm`
- `uv` (Astral) for Python project mgmt
- Internet access (to reach a remote vLLM/SGLang model server, Hugging Face,
  external MCP-tool APIs, and optionally Modal cloud for SWE-bench sandboxes)
- A separate machine (or remote endpoint) actually serving `gpt-oss-120b`
  via vLLM/SGLang. **You are not expected to run the model on this host.**

The agent code lives in **AgentCAP**, branch `sicheng/k8s-benchmark-pipeline`,
which has the gpt-oss tool-use fixes that any reproduction depends on.

> ⚠️ Critical: do NOT run on a branch that lacks commit `787114f`
> ("Fix gpt-oss tool-use on vLLM/SGLang without server-side parser").
> Without that fix the model rambles for thousands of tokens per turn and
> tool-call recovery silently fails.

---

## 0. Required env vars and endpoints

```bash
# Already-running model server (any OpenAI-compatible /v1 endpoint)
export VLLM_URL="http://<your-gpu-host>:30002/v1"
export MODEL_NAME="openai/gpt-oss-120b"   # whatever --served-model-name is

# MCP tool credentials (only needed for MCP-ATLAS)
export GITHUB_TOKEN="ghp_..."
export BRAVE_API_KEY="BSAxk..."

# Modal (only needed for SWE-bench Lite without Docker)
# After `pip install modal`, run `modal token new` once to authenticate.
```

Sanity check the model server first:

```bash
curl -s "$VLLM_URL/models" | head
```

You should see `"id": "openai/gpt-oss-120b"` (or your `--served-model-name`).

---

## 1. Clone and install AgentCAP

```bash
git clone https://github.com/Auto-CAP/AgentCAP.git
cd AgentCAP
git checkout sicheng/k8s-benchmark-pipeline
git submodule update --init --recursive   # pulls third_party/mcp-atlas
pip install -e .
pip install 'swe-rex>=1.4.0'              # SWE-bench harness
```

The `mcp-atlas` submodule is the agent-environment used by the MCP-ATLAS
benchmark. It runs as a separate process; we set it up in §2.

---

## 2. MCP-ATLAS (60 tasks)

### 2a. Start the MCP server natively (no Docker)

The Docker image `ghcr.io/scaleapi/mcp-atlas:latest` boils down to:

- a Python `agent-environment` web service on port 1984 (uvicorn)
- which lazily spawns 22 MCP servers over stdio (mostly Node `npx` and
  Python `uvx` packages)

Start it directly from the submodule:

```bash
cd /path/to/AgentCAP/third_party/mcp-atlas/agent-environment
uv sync                                   # one-time

# httpx>=0.28 dropped TimeoutError; agent-environment still imports it.
# Patch via sitecustomize so all subprocesses get the fix:
python3 -c '
import site, os
sc = os.path.join(site.getsitepackages()[0], "sitecustomize.py")
os.makedirs(os.path.dirname(sc), exist_ok=True)
open(sc, "w").write(
    "import httpx\n"
    "if not hasattr(httpx, \"TimeoutError\"):\n"
    "    httpx.TimeoutError = httpx.TimeoutException\n"
)
print("httpx.TimeoutError patched")
'

ENABLED_SERVERS="arxiv,brave-search,calculator,cli-mcp-server,clinicaltrialsgov-mcp-server,context7,ddg-search,desktop-commander,fetch,filesystem,git,github,mcp-code-executor,mcp-server-code-runner,memory,met-museum,open-library,osm-mcp-server,pubmed,weather,whois,wikipedia" \
GITHUB_TOKEN="$GITHUB_TOKEN" \
BRAVE_API_KEY="$BRAVE_API_KEY" \
./entrypoint.sh uv run python -m uvicorn agent_environment.main:app \
    --host 0.0.0.0 --port 1984
```

The first launch installs all `npx` / `uvx` MCP servers; this can take
10–20 minutes and a few hundred MB of disk. Subsequent runs are instant.

Leave this running. In another shell, sanity check:

```bash
curl -s http://localhost:1984/health
```

> If `npm` / `uv` cannot be installed on this host, this benchmark cannot
> be reproduced exactly. The set of 22 servers is what defines the
> benchmark and dropping any of them changes results.

### 2b. Run the benchmark

```bash
cd /path/to/AgentCAP
mkdir -p results/mcpatlas_local

python -m agent_cap.runner.unified_runner \
    --model-name "$MODEL_NAME" \
    --dataset mcp-atlas \
    --backend mcp \
    --serving-engine vllm \
    --base-url "$VLLM_URL" \
    --mcp-server-url http://localhost:1984 \
    --max-turns 15 \
    --num-tasks 60 \
    --output-dir results/mcpatlas_local
```

Expected wall-clock: ~30–60 min depending on tool latency and TP/GPU.

### 2c. Score with Gemini judge

`unified_runner` writes `output-data_*.jsonl` with raw model outputs.
Acc must be computed by the GTFA judge (OpenRouter Gemini-flash-lite-preview).
There is a helper at `scripts/reeval_mcpatlas.py` (or use the inline script
shown in `tmp/eval_mcpatlas.py` in earlier conversations); it adds an
`eval` block + `quality.acc` to `metrics_*.json`.

Required env: `OPENROUTER_API_KEY=sk-or-...`

```bash
python scripts/reeval_mcpatlas.py results/mcpatlas_local
```

Reference numbers (vLLM A100 TP=2): acc ≈ 0.49–0.55 (run-to-run noise is
real — same config, two runs, was 0.555 vs 0.493).

---

## 3. SWE-bench Lite (100 tasks) — via Modal

Without Docker, the per-task sandbox runs on **Modal** cloud. You only
need a Python process locally; Modal pulls the swebench eval image
(`swebench/sweb.eval.x86_64.<instance_id>`) into its own container and
sweagent talks to it over HTTP. Cost ≈ $5–20 for 100 tasks.

### 3a. One-time Modal setup

```bash
pip install 'swe-rex[modal]' modal
modal token new          # opens browser, writes ~/.modal.toml
```

### 3b. Get sweagent + a config

```bash
git clone https://github.com/SWE-agent/SWE-agent.git /tmp/swe_agent
pip install -e /tmp/swe_agent
# bash_only.yaml is the config used in our k8s runs:
ls /tmp/swe_agent/config/bash_only.yaml
```

### 3c. Run the batch

The unified script `k8s/run_sweagent.py` (commit `bf3edc5` on this
branch) supports `--deployment {k8s,docker,local,modal}`.

```bash
cd /path/to/AgentCAP

python k8s/run_sweagent.py \
    --deployment modal \
    --dataset swe-bench-lite \
    --num-tasks 100 \
    --concurrency 20 \
    --vllm-url "$VLLM_URL" \
    --sweagent-dir /tmp/swe_agent \
    --output-dir results/swebench_lite_modal \
    --image-repo swebench/sweb.eval.x86_64
```

Notes:
- `--concurrency 20`: Modal happily parallelizes; each container is
  isolated. Limit by your model server's max-concurrent-requests, not
  Modal's.
- `--vllm-url`: must be reachable from the Modal containers if you set
  `--env.deployment.host` to that URL. In practice the agent process
  runs locally and forwards HTTP to your endpoint, so as long as
  `$VLLM_URL` is reachable from your *local* machine that's fine.
- Each task writes `task_NNN/{patch.diff,trajectory.traj,problem.txt}`
  under `--output-dir`, plus a global `batch_summary.json`.

### 3d. Evaluate patches with the official harness

The official `swebench` package can also use Modal so you don't need
Docker for evaluation either:

```bash
pip install swebench
# Build a predictions file: {instance_id, model_patch, model_name_or_path}
python scripts/build_swebench_predictions.py \
    results/swebench_lite_modal \
    > results/swebench_lite_preds.json

python -m swebench.harness.run_evaluation \
    --predictions_path results/swebench_lite_preds.json \
    --modal true \
    --max_workers 50 \
    --run_id eidf-modal-eval \
    --dataset_name princeton-nlp/SWE-bench_Lite
```

The harness writes `<run_id>.eidf-modal-eval.json` with per-task pass/fail.

> If `scripts/build_swebench_predictions.py` doesn't exist in this branch,
> it's a 20-line script: walk `results/swebench_lite_modal/task_*/`,
> read `patch.diff`, build the JSON list. Look at any prior run under
> `TEAS_Development_Results_Private/.../swe-bench-lite/.../detailed-results_*.jsonl`
> for the exact field names the harness expects.

Reference numbers (vLLM A100 TP=2 streaming): pass-rate ≈ 0.32–0.40.

---

## 4. What to verify before reporting numbers

For each completed run, sanity-check:

1. **Tool-call recovery**: in the trajectory / detailed-results files,
   `tool_calls` should be non-empty for the majority of turns. If
   essentially all are `[]`, the gpt-oss `<|call|>` stop-token recovery
   is broken — verify you are on the branch with commit `787114f`.

2. **Decode token count per turn**: should be 100–500 tokens for
   gpt-oss-120b on agentic tasks. If you see 3000–6000 tokens/turn, the
   `stop_token_ids` aren't being sent — same fix as #1.

3. **MCP-ATLAS no-answer rate (gsm8k pattern)**: should be 0–5% with
   the `#### <number>` system prompt. If much higher, ensure you have
   commit `d89e567` from MoE-CAP main (or are using AgentCAP, which
   doesn't have this issue at all — it's a MoE-CAP bug).

4. **LongBench v1 deps** (irrelevant here but easy to forget): `rouge`,
   `fuzzywuzzy`, `python-Levenshtein` must be installed or qmsum/samsum
   silently score 0.

---

## 5. Quick reference — files in this repo you'll touch

| Path | Purpose |
|---|---|
| `agent_cap/runner/unified_runner.py` | mcp-atlas + IMO + medagentbench + tau2 + swebench-lite |
| `agent_cap/runner/llm_client.py` | gpt-oss stop_token + tool-arg cleaning |
| `k8s/run_sweagent.py` | SWE-bench Lite/Pro batch runner; `--deployment` flag |
| `third_party/mcp-atlas/` | submodule, the MCP server you start in §2a |
| `agent_cap/evaluators/gtfa_eval.py` | Gemini judge for mcp-atlas |

## 6. Things that will trip you up

- **Modal needs the predictions to be reachable**: if `$VLLM_URL` points
  at a private-network IP, Modal containers can't reach it. The agent
  process is local; it talks to vLLM locally. The Modal container only
  needs to talk to *the agent over HTTP via swerex*, not to vLLM.
- **mcp-atlas dataset filter**: the public `ScaleAI/mcp-atlas` dataset
  has more than 60 rows. The 60-task benchmark uses `--num-tasks 60`,
  which takes the first 60 after `dataset.shuffle(seed=42)`. Don't
  reshuffle.
- **GPU non-determinism**: even with `temperature=0`, the same run can
  shift ±5pp on accuracy run-to-run (cuBLAS reduction order). For real
  comparisons, run 3+ seeds and take the mean.
