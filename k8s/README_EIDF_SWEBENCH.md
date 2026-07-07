# SWE-bench Lite (docker version) on EIDF k8s — deployment guide

Runs the **same pipeline as `scripts/run_swebench_docker_100.sh`** (SWE-agent +
official swebench docker images + official harness grading) on the EIDF GPU
service, where there is **no docker daemon and no dind**. Every docker
container is replaced by a k8s pod running the same official image:

| docker version                          | EIDF k8s version                                   |
|-----------------------------------------|----------------------------------------------------|
| LLM served by local docker image        | k8s Job + ClusterIP Service, official image        |
| per-task sandbox container (swe-rex)    | per-task sidecar pod + `kubectl port-forward`, SWE-agent `remote` deployment (`--sweagent-deployment k8s`) |
| `swebench.harness.run_evaluation` (docker) | per-instance eval pod + local grading with `swebench.harness.grading` (`--evaluator swebench-k8s`) |

All images are the official ones: `lmsysorg/sglang:v0.5.9`,
`vllm/vllm-openai:v0.21.0`, `docker.io/swebench/sweb.eval.x86_64.<iid>:latest`.
Everything is orchestrated from the login node (needs only kubectl + conda).

## One-time setup (login node)

```bash
conda activate agentcap
pip install -e ~/AgentCAP
pip install swe-rex swebench
git clone https://github.com/SWE-agent/SWE-agent.git ~/swe_agent
python ~/AgentCAP/scripts/patch_sweagent_streaming.py ~/swe_agent   # TTFT/TPOT/cached-token capture
pip install -e ~/swe_agent
```

swe-rex 1.4.0 fix (client-side): `swerex/deployment/remote.py`'s
`is_alive()` must accept `timeout=` like the docker/modal deployments do
(one-line signature change + pass-through), otherwise SWE-agent's
autosubmission crashes on the remote deployment and partial patches are
lost. Applied in the `agentcap` env.

Model weights must be on `llm-cache-pvc` at
`/workspace/models/unsloth/gpt-oss-120b` (already there).

## Per-experiment workflow

```bash
cd ~/AgentCAP

# 1. Launch the LLM server (engine: sglang|vllm; gpu: a100|h100|h200)
bash k8s/launch_llm_server.sh sglang h200
kubectl logs -l app=sglang-gptoss-h200 -f        # wait for "ready"

# 2. Keep a tunnel alive (separate terminal / nohup)
nohup bash k8s/port_forward_llm.sh sglang-gptoss-h200 8000 > /tmp/pf.log 2>&1 &
curl -s http://127.0.0.1:8000/v1/models          # sanity

# 3. Run the curated-100 benchmark (agent phase + eval phase)
bash scripts/run_swebench_k8s_100.sh \
    --output-dir ~/agentcap-results/sglang_h200x1_$(date +%y%m%d-%H%M) \
    --concurrency 4
# interrupted? add --resume to skip completed tasks

# 4. Package into TEAS results layout
python scripts/package_teas_results.py \
    --run-dir ~/agentcap-results/sglang_h200x1_... \
    --engine sglang --gpu-type "NVIDIA H200" --num-gpus 1 --tp 1

# 5. Tear down
bash k8s/launch_llm_server.sh sglang h200 --stop
```

GPU configs used for the paper runs: `a100`=2×A100-SXM4-80GB (tp2),
`h100`=2×H100-80GB (tp2), `h200`=1×H200 (tp1). Project quota is 12 GPUs —
check `kubectl describe queue eidf230ns-user-queue` before launching
(a long-running 8-GPU job may force experiments to run sequentially).

## Known issue: sglang gpt-oss tool-call parsing (NOT patched — aligned with vastai runs)

Stock sglang v0.5.9 (parser identical in v0.5.12.post1, the version used for
the vastai B200/B300 runs) sometimes returns an empty message (no content,
no tool_calls) for gpt-oss-120b tool calls: when the model omits/varies the
`<|constrain|>json` marker, emits the call on the analysis channel, or
produces JSON args with `\'` escapes (bash heredocs). SWE-agent then retries
up to 3× and may abort the task ("repeated format errors" / "arguments are
not valid JSON").

We deliberately run the OFFICIAL image unmodified so results stay aligned
with the vastai reference runs — these failures are part of the measured
system behaviour. `k8s/patch_sglang_gptoss.py` exists in the repo as
documentation of the root causes (replay-verified) but is NOT applied.

## Sizing / knobs

- `--concurrency 4`: 4 sidecar pods (2 CPU / 8 Gi each) + 4 SWE-agent
  subprocesses on the login node. The login node has 8 cores / 15 GiB —
  don't go much higher.
- `SWEAGENT_TASK_TIMEOUT` (default 1800 s) per-task SWE-agent wall clock.
- `SWEBENCH_EVAL_TIMEOUT` (default 1800 s) per-instance test run.
- `SWEBENCH_K8S_NAMESPACE` (default `eidf230ns`).
- Sidecar/eval pods pip-install `swe-rex` at startup (~30 s) — no custom
  image registry needed.

## Troubleshooting

- Job stuck `Suspended` → Kueue queueing; `kubectl describe workload <name>`.
- Sidecar `ImagePullBackOff` → instance image name wrong or Docker Hub rate
  limit; check `kubectl describe pod <pod>`.
- Port-forward drops kill in-flight LLM streams — the keeper script restarts
  the tunnel, affected task fails and can be re-run with `--resume`.
- Login node OOM → lower `--concurrency`.
