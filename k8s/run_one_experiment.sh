#!/usr/bin/env bash
# Self-contained driver for ONE SWE-bench experiment on EIDF.
# Designed to run detached (setsid) so it survives terminal/session restarts:
#   setsid nohup bash k8s/run_one_experiment.sh sglang h100 > /dev/null 2>&1 &
#
# Steps: launch server -> tunnel -> wait ready -> curated-100 run (resume-safe)
#        -> package TEAS -> stop server.
# All progress goes to ~/agentcap-results/<engine>_<gpu>x<tp>_<date>/driver.log
set -uo pipefail

ENGINE="${1:?engine sglang|vllm}"
GPU="${2:?gpu a100|h100|h200}"
CONCURRENCY="${3:-4}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentcap
cd "$(dirname "$0")/.."

case "$GPU" in
  a100) NGPU=2; GPUTYPE="NVIDIA A100";;
  h100) NGPU=2; GPUTYPE="NVIDIA H100";;
  h200) NGPU=1; GPUTYPE="NVIDIA H200";;
esac
case "$ENGINE" in
  sglang) EVER="0.5.9";;
  vllm)   EVER="0.21.0";;
esac

OUT=~/agentcap-results/${ENGINE}_${GPU}x${NGPU}_$(date +%y%m%d)
mkdir -p "$OUT"
LOG="$OUT/driver.log"
exec >> "$LOG" 2>&1
echo "=== driver start $(date -Is) engine=$ENGINE gpu=$GPU out=$OUT"

# 1. server (skip if this engine+gpu's service already serves the model)
SVC="${ENGINE}-gptoss-${GPU}"
if ! kubectl get pods -l "app=$SVC" --no-headers 2>/dev/null | grep -q Running; then
    bash k8s/launch_llm_server.sh "$ENGINE" "$GPU"
fi

# 2. tunnel (kill stale keepers first)
pkill -f "port_forward_llm.sh" 2>/dev/null; sleep 2
setsid nohup bash k8s/port_forward_llm.sh "$SVC" 8000 > "$OUT/pf.log" 2>&1 < /dev/null &

# 3. wait for model (up to 40 min for image pull + load + queue)
for i in $(seq 1 160); do
    curl -s -m 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q gpt-oss && break
    sleep 15
done
if ! curl -s -m 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q gpt-oss; then
    echo "FATAL: server never became ready"; exit 1
fi
echo "server ready $(date -Is)"

# 4. benchmark (resume-safe: rerun of this driver continues where it left off)
RESUME=""
[ -s "$OUT/results.jsonl" ] && RESUME="--resume"
bash scripts/run_swebench_k8s_100.sh --output-dir "$OUT" --concurrency "$CONCURRENCY" $RESUME
echo "run finished $(date -Is)"

# 5. package (run.sh = provenance copied into the TEAS dir)
cat > "$OUT/run.sh" <<PROV
#!/usr/bin/env bash
# EIDF ${NGPU}x ${GPUTYPE} -- official image, ${ENGINE} v${EVER} serving unsloth/gpt-oss-120b TP=${NGPU}
# AgentCAP $(git rev-parse --short HEAD 2>/dev/null); SWE-bench Lite curated-100 via SWE-agent + k8s swe-rex sidecars
# eval: per-instance k8s pods running official TestSpec scripts, swebench.harness.grading
bash k8s/launch_llm_server.sh ${ENGINE} ${GPU}
bash k8s/port_forward_llm.sh ${ENGINE}-gptoss-${GPU} 8000 &
bash scripts/run_swebench_k8s_100.sh --output-dir $OUT --concurrency ${CONCURRENCY}
# TEAS outputs (metadata/metrics/detailed-results/output-data) are written
# automatically at run end by agent_cap.agents.teas_output
PROV
# TEAS-format outputs are written automatically by the run itself
# (agent_cap.agents.teas_output; TEAS_* env exported by run_swebench_k8s_100.sh)

# 6. teardown
bash k8s/launch_llm_server.sh "$ENGINE" "$GPU" --stop
pkill -f "port_forward_llm.sh" 2>/dev/null
echo "=== driver done $(date -Is)"
