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

The agent code lives in **AgentCAP** `main`, which has the gpt-oss
tool-use fixes and the unified `agent_cap.agents` runtime that any
reproduction depends on.

## 0. Required env vars and endpoints

Shell exports for the runner side:

```bash
# Already-running model server (any OpenAI-compatible /v1 endpoint)
export VLLM_URL="http://<your-gpu-host>:30002/v1"
export MODEL_NAME="openai/gpt-oss-120b"   # whatever --served-model-name is

# Modal (only needed for SWE-bench Lite without Docker)
# After `pip install modal`, run `modal token new` once to authenticate.
```

MCP tool credentials (only needed for MCP-ATLAS) live in
`third_party/mcp-atlas/.env` (gitignored). `mcp-server/start.sh` copies
`env.template` to `.env` on first run; fill in the keys you need
(`GITHUB_PERSONAL_ACCESS_TOKEN`, `BRAVE_API_KEY`, `ALCHEMY_API_KEY`, etc.).
Servers with empty keys still start; they return auth errors only at
tool-call time.

Sanity check the model server first:

```bash
curl -s "$VLLM_URL/models" | head
```

You should see `"id": "openai/gpt-oss-120b"` (or your `--served-model-name`).

---

## 1. Clone and install AgentCAP

```bash
git clone --recurse-submodules https://github.com/Auto-CAP/AgentCAP.git
cd AgentCAP
pip install -e .
pip install 'swe-rex>=1.4.0'              # SWE-bench harness
```

The `mcp-atlas` submodule is the agent-environment used by the MCP-ATLAS
benchmark. It runs as a separate process; we set it up in §2.

---

## 2. MCP-ATLAS (60 tasks)

### 2a. Start the MCP server natively (no Docker)

```bash
cd /path/to/AgentCAP
bash mcp-server/start.sh
```

This is the docker-free equivalent of `ghcr.io/scaleapi/mcp-atlas:latest`
(Python 3.12 venv + agent-environment + envsubst + uvicorn). The full
setup, environment variables, troubleshooting, and concurrency tuning
live in **[docs/mcp-server.md](mcp-server.md)**.

Leave the server running. In another shell:

```bash
curl -s http://localhost:1984/health
# {"status":"health_and_client_connection_ok"}
```

> If `npm` / `uv` / `node>=20` cannot be installed on this host, this
> benchmark cannot be reproduced exactly. The set of MCP servers is
> what defines the benchmark and dropping any of them changes results.

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

The unified script `scripts/run_sweagent.py` (commit `dfcf2d7` on this
branch) supports `--deployment {k8s,docker,local,modal}`.

```bash
cd /path/to/AgentCAP

python scripts/run_sweagent.py \
    --deployment modal \
    --dataset swe-bench-lite \
    --task-indices benchmarks/swe_bench_lite_curated_100.json \
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
   is broken — make sure you're on a recent `main` (the fix lives in
   `agent_cap.runner.llm_client._stream_chat_completion_harmony`).

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
| `scripts/run_sweagent.py` | SWE-bench Lite/Pro batch runner; `--deployment` flag |
| `third_party/mcp-atlas/` | submodule, the MCP server you start in §2a |
| `agent_cap/evaluators/gtfa_eval.py` | Gemini judge for mcp-atlas |

## 6. Things that will trip you up

- **Modal needs the predictions to be reachable**: if `$VLLM_URL` points
  at a private-network IP, Modal containers can't reach it. The agent
  process is local; it talks to vLLM locally. The Modal container only
  needs to talk to *the agent over HTTP via swerex*, not to vLLM.
- **mcp-atlas dataset filter**: the public `ScaleAI/mcp-atlas` dataset
  has hundreds of rows but ~85 of them require **paid** MCP tools
  (firecrawl, etc.). `unified_runner` keeps only tasks whose
  `ENABLED_TOOLS` are all inside a 22-server free subset
  (arxiv, brave-search, calculator, ..., wikipedia — see
  `_FREE_SERVERS` in `unified_runner.py:1222`). After that filter,
  `--num-tasks 60` takes the first 60 in HF dataset order. **No
  shuffle**, so the 60 tasks are deterministic across runs. Don't
  add a shuffle.

- **swe-bench-lite is a *curated* 100-task subset, not first-100**.
  The 100 tasks used in prior reports were hand-balanced across the
  12 repos in SWE-bench_Lite:
    - 6 astropy
    - 18 django (drawn from previously-passing tasks)
    - 6 django (drawn from previously-failing tasks)
    - 70 uniformly sampled across the other 10 repos (matplotlib,
      sympy, scikit-learn, sphinx-doc, pytest-dev, requests/psf,
      xarray/pydata, pylint-dev, seaborn/mwaskom, ...)
  The exact indices and instance_ids live in
  `benchmarks/swe_bench_lite_curated_100.json` (committed to this
  branch). Pass it via `--task-indices`:

  ```bash
  python scripts/run_sweagent.py --deployment modal \
      --dataset swe-bench-lite \
      --task-indices benchmarks/swe_bench_lite_curated_100.json \
      --vllm-url "$VLLM_URL" \
      --output-dir results/swebench_lite_modal
  ```

  The runner reads either `new_indices` or `indices` from that JSON
  and ignores `--num-tasks` when `--task-indices` is given. Without
  `--task-indices` you get rows `[0:N]` of HF dataset order — a
  *different* benchmark, not comparable to prior numbers.
- **GPU non-determinism**: even with `temperature=0`, the same run can
  shift ±5pp on accuracy run-to-run (cuBLAS reduction order). For real
  comparisons, run 3+ seeds and take the mean.
