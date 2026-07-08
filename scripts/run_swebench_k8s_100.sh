#!/usr/bin/env bash
# Run SWE-agent on the curated-100 SWE-bench Lite subset using K8S SIDECAR
# sandboxes (EIDF GPU service — no docker daemon / no dind needed).
#
# Sandbox + eval both run as k8s pods from the official docker.io/swebench
# instance images:
#   - agent phase:  per-task swe-rex sidecar pod, SWE-agent connects via
#                   kubectl port-forward (deployment type "remote")
#   - eval  phase:  per-instance pod, official TestSpec eval script, graded
#                   locally with swebench.harness.grading (same semantics as
#                   swebench.harness.run_evaluation)
#
# REQUIRED RUNTIME (login node):
#   - kubectl configured for the project namespace (default eidf230ns)
#   - conda env with: agent_cap (pip -e), swe-rex, swebench, sweagent (pip -e
#     of a checkout patched by scripts/patch_sweagent_streaming.py)
#   - LLM server running in-cluster (k8s/launch_llm_server.sh) and
#     port-forwarded to localhost:8000 (k8s/port_forward_llm.sh)
#
# USAGE:
#   bash scripts/run_swebench_k8s_100.sh --output-dir DIR [--llm-url URL]
#        [--concurrency N] [--model M] [--num-tasks N] [--resume]
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

LLM_URL="${LLM_URL:-http://127.0.0.1:8000/v1}"
OUTPUT_DIR=""
CONCURRENCY=4
INDICES_FILE="$REPO_ROOT/benchmarks/swe_bench_lite_curated_100.json"
SWEAGENT_DIR="${SWEAGENT_DIR:-$HOME/swe_agent}"
MODEL="${MODEL:-openai/unsloth/gpt-oss-120b}"
NUM_TASKS=0
RESUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm-url) LLM_URL="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --indices) INDICES_FILE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --sweagent-dir) SWEAGENT_DIR="$2"; shift 2 ;;
        --num-tasks) NUM_TASKS="$2"; shift 2 ;;
        --resume) RESUME="--resume"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done
[[ -n "$OUTPUT_DIR" ]] || { echo "ERROR: --output-dir required"; exit 1; }

# One run per output dir — a duplicate concurrent run corrupts results.jsonl
# and the per-task stream_stats (observed when a session flap re-executed the
# launch command).
mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.run.lock"
flock -n 9 || { echo "ERROR: another run is already active on $OUTPUT_DIR"; exit 1; }

# Clear leftovers from crashed runs: stale sidecar port-forwards would collide
# with new tunnels (SWE-agent then hits an OLD sidecar -> SessionExistsError),
# and orphaned sidecar jobs waste quota.
pkill -f "port-forward pod/swe-rex" 2>/dev/null || true
kubectl delete job -l app=sweagent-sidecar --ignore-not-found >/dev/null 2>&1 || true

echo "== Verifying kubectl =="
kubectl get queue >/dev/null || { echo "ERROR: kubectl not working"; exit 1; }

echo "== Verifying LLM endpoint =="
curl -sS -m 5 -o /dev/null -w "  $LLM_URL  ->  %{http_code}\n" "$LLM_URL/models" || {
    echo "ERROR: LLM not reachable at $LLM_URL"; exit 1
}

# The output dir name declares the engine (sglang_*/vllm_*). Refuse to run
# against a mismatched server — a tunnel pointing at the wrong engine produces
# silently mislabeled results (happened twice with concurrent sessions).
DIR_ENGINE="$(basename "$OUTPUT_DIR" | grep -oE '^(sglang|vllm)' || true)"
if [[ -n "$DIR_ENGINE" ]]; then
    SERVED_BY="$(curl -sS -m 5 "$LLM_URL/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0].get("owned_by",""))' 2>/dev/null || true)"
    if [[ -n "$SERVED_BY" && "$SERVED_BY" != "$DIR_ENGINE" ]]; then
        echo "ERROR: output dir says '$DIR_ENGINE' but $LLM_URL is served by '$SERVED_BY'"; exit 1
    fi
    echo "  engine check: $DIR_ENGINE == $SERVED_BY"
fi

echo "== Verifying SWE-agent checkout (with stream patch) =="
[[ -f "$SWEAGENT_DIR/sweagent/agent/models.py" ]] || {
    echo "ERROR: $SWEAGENT_DIR/sweagent/agent/models.py not found"; exit 1
}
grep -q "AGENTCAP_STREAMING_PATCH_APPLIED" "$SWEAGENT_DIR/sweagent/agent/models.py" || {
    echo "ERROR: $SWEAGENT_DIR not patched (run scripts/patch_sweagent_streaming.py)"; exit 1
}

echo "== Verifying curated-100 indices file =="
[[ -f "$INDICES_FILE" ]] || { echo "ERROR: $INDICES_FILE not found"; exit 1; }
N=$(python3 -c "import json;d=json.load(open('$INDICES_FILE'));print(len(d.get('indices') or d.get('new_indices') or []))")
echo "  $INDICES_FILE  ->  $N tasks"

EXTRA_ARGS=()
[[ "$NUM_TASKS" != "0" ]] && EXTRA_ARGS+=(--num-tasks "$NUM_TASKS")
[[ -n "$RESUME" ]] && EXTRA_ARGS+=("$RESUME")

echo "== Launching SWE-agent batch (k8s sidecars, concurrency=$CONCURRENCY) =="
mkdir -p "$OUTPUT_DIR"
python -m agent_cap.agents \
    --strategy sweagent \
    --model "$MODEL" \
    --base-url "$LLM_URL" \
    --api-key dummy \
    --dataset swe-bench-lite \
    --task-indices "$INDICES_FILE" \
    --sweagent-deployment k8s \
    --sweagent-dir "$SWEAGENT_DIR" \
    --sweagent-image-repo "" \
    --sweagent-call-limit 200 \
    --concurrency "$CONCURRENCY" \
    --evaluator swebench-k8s \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo "== Done.  Outputs =="
echo "  predictions.json :  $OUTPUT_DIR/predictions.json"
echo "  results.jsonl    :  $OUTPUT_DIR/results.jsonl"
echo "  eval reports     :  $OUTPUT_DIR/eval_k8s/"
tail -1 "$OUTPUT_DIR/run.log" 2>/dev/null
