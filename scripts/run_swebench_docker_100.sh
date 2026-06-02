#!/usr/bin/env bash
# Run SWE-agent on the curated-100 SWE-bench Lite subset using LOCAL DOCKER sandboxes.
#
# REQUIRED ENV:
#   (none — uses local docker; LLM endpoint can be unauthenticated)
#
# REQUIRED RUNTIME:
#   - Docker daemon (will spawn one container per task; concurrency configurable)
#   - swebench harness images, either:
#       * Pre-pulled docker.io/swebench/sweb.eval.x86_64.<iid>:latest, OR
#       * Built locally as sweb.eval.x86_64.<iid>:latest via swebench harness
#   - LLM server on :8000 (vLLM or SGLang) at gpt-oss-120b, 131072 ctx.
#     SGLang flags: --reasoning-parser gpt-oss --tool-call-parser gpt-oss
#                   --context-length 131072 --enable-cache-report
#     vLLM flags:   --reasoning-parser openai_gptoss --tool-call-parser openai
#                   --max-model-len 131072 --enable-prompt-tokens-details
#   - SWE-agent checkout at /tmp/swe_agent (with the streaming + reasoning patch).
#
# USAGE:
#   bash scripts/run_swebench_docker_100.sh [--llm-url URL] [--output-dir DIR] [--concurrency N]
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

LLM_URL="${LLM_URL:-http://localhost:8000/v1}"
OUTPUT_DIR="/data/sicheng/agent-team-data/gptoss-120b_sweagent_swebenchlite_curated100_docker"
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

echo "== Verifying docker daemon =="
docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable"; exit 1; }

echo "== Verifying SWE-agent checkout (with stream patch) =="
[[ -f "$SWEAGENT_DIR/sweagent/agent/models.py" ]] || {
    echo "ERROR: $SWEAGENT_DIR/sweagent/agent/models.py not found"; exit 1
}
grep -q "SWEAGENT_STREAM_STATS_PATH" "$SWEAGENT_DIR/sweagent/agent/models.py" || {
    echo "ERROR: $SWEAGENT_DIR not patched for streaming stats"; exit 1
}
grep -q "delta.reasoning\|_map_reasoning\|reasoning_content" "$SWEAGENT_DIR/sweagent/agent/models.py" || {
    echo "ERROR: $SWEAGENT_DIR missing vllm reasoning patch"; exit 1
}

echo "== Verifying curated-100 indices file =="
[[ -f "$INDICES_FILE" ]] || { echo "ERROR: $INDICES_FILE not found"; exit 1; }
N=$(python3 -c "import json;d=json.load(open('$INDICES_FILE'));print(len(d.get('indices') or d.get('new_indices') or []))")
echo "  $INDICES_FILE  ->  $N tasks"

echo "== Launching SWE-agent batch via unified CLI (docker, concurrency=$CONCURRENCY) =="
mkdir -p "$OUTPUT_DIR"
python -m agent_cap.agents \
    --strategy sweagent \
    --model "$MODEL" \
    --base-url "$LLM_URL" \
    --api-key dummy \
    --dataset swe-bench-lite \
    --task-indices "$INDICES_FILE" \
    --sweagent-deployment docker \
    --sweagent-dir "$SWEAGENT_DIR" \
    --sweagent-image-repo "" \
    --sweagent-call-limit 200 \
    --concurrency "$CONCURRENCY" \
    --evaluator swebench \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "== Done.  Outputs =="
echo "  predictions.json :  $OUTPUT_DIR/predictions.json"
echo "  results.jsonl    :  $OUTPUT_DIR/results.jsonl"
echo "  per-task         :  $OUTPUT_DIR/task_<instance_id>/"
tail -1 "$OUTPUT_DIR/run.log" 2>/dev/null
