#!/usr/bin/env bash
# DIRECT (uses the dedicated CLI): uses `scripts/run_sweagent.py` directly (NOT unified `agent_cap.agents`).
# See run_swebench_docker_100.sh for the unified version.
#
# Required runtime: docker daemon, /tmp/swe_agent checkout with streaming patches,
#   LLM server on :8000 (gpt-oss-120b, 131072 ctx).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

LLM_URL="${LLM_URL:-http://localhost:8000/v1}"
OUTPUT_DIR="/data/sicheng/agent-team-data/gptoss-120b_sweagent_swebenchlite_curated100_docker_direct"
CONCURRENCY=4
INDICES_FILE="$REPO_ROOT/benchmarks/swe_bench_lite_curated_100.json"
SWEAGENT_DIR="${SWEAGENT_DIR:-/tmp/swe_agent}"
MODEL="${MODEL:-openai/unsloth/gpt-oss-120b}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm-url) LLM_URL="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --indices) INDICES_FILE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --sweagent-dir) SWEAGENT_DIR="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

echo "== Verifying LLM endpoint =="
curl -sS -m 5 -o /dev/null -w "  $LLM_URL  ->  %{http_code}\n" "$LLM_URL/models" || {
    echo "ERROR: LLM not reachable at $LLM_URL"; exit 1
}

docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable"; exit 1; }

[[ -f "$SWEAGENT_DIR/sweagent/agent/models.py" ]] || {
    echo "ERROR: $SWEAGENT_DIR/sweagent/agent/models.py not found"; exit 1
}
grep -q "SWEAGENT_STREAM_STATS_PATH" "$SWEAGENT_DIR/sweagent/agent/models.py" || {
    echo "ERROR: $SWEAGENT_DIR not patched for streaming stats"; exit 1
}

[[ -f "$INDICES_FILE" ]] || { echo "ERROR: $INDICES_FILE not found"; exit 1; }

echo "== Launching SWE-agent batch (docker, concurrency=$CONCURRENCY, legacy run_sweagent.py) =="
mkdir -p "$OUTPUT_DIR"
python scripts/run_sweagent.py \
    --deployment docker \
    --dataset swe-bench-lite \
    --task-indices "$INDICES_FILE" \
    --vllm-url "$LLM_URL" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --sweagent-dir "$SWEAGENT_DIR" \
    --image-repo "" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "== Done.  Outputs =="
echo "  predictions.json :  $OUTPUT_DIR/predictions.json"
echo "  results.jsonl    :  $OUTPUT_DIR/results.jsonl"
tail -1 "$OUTPUT_DIR/run.log" 2>/dev/null
