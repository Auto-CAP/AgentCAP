#!/usr/bin/env bash
# Run mcp-atlas free-60 on EIDF k8s, mirroring the vastai b300 reference:
#   agent_cap.agents --strategy single --dataset mcp-atlas --use-streaming
#   --concurrency 4 --evaluator gtfa (judge google/gemini-3.1-flash-lite)
# Requires: LLM server tunneled to :8000, MCP sidecar tunneled to :1984,
#           EVAL_LLM_API_KEY in the environment (judge).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

LLM_URL="${LLM_URL:-http://127.0.0.1:8000/v1}"
MCP_URL="${MCP_URL:-http://127.0.0.1:1984}"
OUTPUT_DIR=""
CONCURRENCY=4
INDICES_FILE="$REPO_ROOT/benchmarks/mcp_atlas_free_60.json"
MODEL="${MODEL:-unsloth/gpt-oss-120b}"
RESUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --resume) RESUME="--resume"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done
[[ -n "$OUTPUT_DIR" ]] || { echo "ERROR: --output-dir required"; exit 1; }
[[ -n "${EVAL_LLM_API_KEY:-}" ]] || { echo "ERROR: EVAL_LLM_API_KEY not set (gtfa judge)"; exit 1; }

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.run.lock"
flock -n 9 || { echo "ERROR: another run active on $OUTPUT_DIR"; exit 1; }

echo "== endpoints =="
curl -sS -m 5 -o /dev/null -w "  LLM $LLM_URL -> %{http_code}\n" "$LLM_URL/models"
curl -sS -m 10 -o /dev/null -w "  MCP $MCP_URL -> %{http_code}\n" -X POST -H 'Content-Type: application/json' -d '{}' "$MCP_URL/list-tools"

# engine consistency + TEAS env (same derivation as the swe-bench script)
DIR_BASE="$(basename "$OUTPUT_DIR")"
DIR_ENGINE="$(echo "$DIR_BASE" | grep -oE '^(sglang|vllm)' || true)"
SERVED_BY="$(curl -sS -m 5 "$LLM_URL/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0].get("owned_by",""))' 2>/dev/null || true)"
if [[ -n "$DIR_ENGINE" && -n "$SERVED_BY" && "$SERVED_BY" != "$DIR_ENGINE" ]]; then
    echo "ERROR: dir says '$DIR_ENGINE' but server is '$SERVED_BY'"; exit 1
fi
GPU_KEY="$(echo "$DIR_BASE" | grep -oE '(a100|h100|h200)' | head -1 || true)"
GPU_N="$(echo "$DIR_BASE" | grep -oE 'x[0-9]+' | head -1 | tr -d x || true)"
case "$GPU_KEY" in
  a100) GPU_NAME="NVIDIA A100" ;;
  h100) GPU_NAME="NVIDIA H100" ;;
  h200) GPU_NAME="NVIDIA H200" ;;
  *)    GPU_NAME="" ;;
esac
export TEAS_ENGINE="${TEAS_ENGINE:-$DIR_ENGINE}"
case "$TEAS_ENGINE" in
  sglang) export TEAS_ENGINE_VERSION="${TEAS_ENGINE_VERSION:-0.5.12.post1}" ;;
  vllm)   export TEAS_ENGINE_VERSION="${TEAS_ENGINE_VERSION:-0.21.0}" ;;
esac
export TEAS_GPU_TYPE="${TEAS_GPU_TYPE:-$GPU_NAME}"
export TEAS_NUM_GPUS="${TEAS_NUM_GPUS:-${GPU_N:-0}}"
export TEAS_TP="${TEAS_TP:-$TEAS_NUM_GPUS}"
export TEAS_MODEL_NAME="${TEAS_MODEL_NAME:-${MODEL#openai/}}"
export TEAS_PRECISION="${TEAS_PRECISION:-mxfp4}"
export TEAS_BACKEND="mcp-atlas-localmcp"
if [[ -z "${TEAS_CPU_TYPE:-}" && -n "$GPU_KEY" ]]; then
    SRV_POD="$(kubectl get pods -l "app=$TEAS_ENGINE-gptoss-$GPU_KEY" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
    if [[ -n "$SRV_POD" ]]; then
        export TEAS_CPU_TYPE="$(kubectl exec "$SRV_POD" -- sh -c "lscpu 2>/dev/null | grep 'Model name' | head -1 | cut -d: -f2" 2>/dev/null | xargs || true)"
        export TEAS_NUM_CPUS="$(kubectl exec "$SRV_POD" -- nproc 2>/dev/null || echo 0)"
    fi
fi
echo "  TEAS env: engine=$TEAS_ENGINE v$TEAS_ENGINE_VERSION gpu='$TEAS_GPU_TYPE' x$TEAS_NUM_GPUS"

EXTRA=()
[[ -n "$RESUME" ]] && EXTRA+=("$RESUME")

echo "== mcp-atlas free-60 (concurrency=$CONCURRENCY) =="
python -m agent_cap.agents \
    --strategy single \
    --model "$MODEL" \
    --base-url "$LLM_URL" \
    --api-key dummy \
    --dataset mcp-atlas \
    --tool-backend mcp \
    --mcp-server-url "$MCP_URL" \
    --task-indices "$INDICES_FILE" \
    --use-streaming \
    --concurrency "$CONCURRENCY" \
    --evaluator gtfa \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA[@]}" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

tail -1 "$OUTPUT_DIR/run.log"
