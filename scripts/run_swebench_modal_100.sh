#!/usr/bin/env bash
# Run SWE-agent on the curated-100 SWE-bench Lite subset using MODAL sandboxes.
#
# REQUIRED ENV / AUTH:
#   - Modal token configured at ~/.modal.toml  (run: modal token new)
#     Workspace must allow GPU/CPU sandbox creation.
#   - (LLM endpoint is exposed via public tunnel; no API key needed by Modal side)
#
# REQUIRED RUNTIME:
#   - LLM server on local :8000 (vLLM or SGLang) at gpt-oss-120b, 131072 ctx.
#     SGLang flags: --reasoning-parser gpt-oss --tool-call-parser gpt-oss
#                   --context-length 131072 --enable-cache-report
#     vLLM flags:   --reasoning-parser openai_gptoss --tool-call-parser openai
#                   --max-model-len 131072 --enable-prompt-tokens-details
#   - cloudflared installed (https://github.com/cloudflare/cloudflared)
#     Script will (re)create a quick tunnel to localhost:8000 if --llm-url unset.
#   - SWE-agent checkout at /tmp/swe_agent (with streaming + reasoning patch).
#   - Python `boto3` package in current env (for swerex modal backend).
#
# USAGE:
#   bash scripts/run_swebench_modal_100.sh [--llm-url URL] [--output-dir DIR] [--concurrency N]
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

LLM_URL="${LLM_URL:-}"
OUTPUT_DIR="/data/sicheng/agent-team-data/gptoss-120b_sweagent_swebenchlite_curated100_modal"
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

echo "== Verifying modal auth =="
[[ -f "$HOME/.modal.toml" ]] || { echo "ERROR: run 'modal token new' first"; exit 1; }

echo "== Verifying boto3 (swerex modal needs it) =="
python3 -c "import boto3" 2>/dev/null || { echo "ERROR: pip install boto3"; exit 1; }

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

if [[ -z "$LLM_URL" ]]; then
    echo "== No --llm-url; bringing up cloudflared quick tunnel for :8000 =="
    command -v cloudflared >/dev/null || { echo "ERROR: cloudflared not installed"; exit 1; }
    curl -sS -m 5 -o /dev/null -w "  localhost:8000 -> %{http_code}\n" http://localhost:8000/v1/models || {
        echo "ERROR: local LLM not reachable on :8000"; exit 1
    }
    if ! pgrep -f "cloudflared tunnel --url http://localhost:8000" >/dev/null; then
        nohup cloudflared tunnel --url http://localhost:8000 --no-autoupdate \
            > "$HOME/cloudflared_8000.log" 2>&1 &
        echo "  waiting for trycloudflare URL..."
        for i in {1..30}; do
            URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$HOME/cloudflared_8000.log" | head -1 || true)
            [[ -n "${URL:-}" ]] && break
            sleep 2
        done
    else
        URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$HOME/cloudflared_8000.log" | head -1 || true)
    fi
    [[ -n "${URL:-}" ]] || { echo "ERROR: tunnel URL not found"; exit 1; }
    LLM_URL="$URL/v1"
fi

echo "== Verifying LLM endpoint via tunnel =="
curl -sS -m 10 -o /dev/null -w "  $LLM_URL  ->  %{http_code}\n" "$LLM_URL/models" || {
    echo "ERROR: LLM not reachable at $LLM_URL"; exit 1
}

echo "== Verifying curated-100 indices file =="
[[ -f "$INDICES_FILE" ]] || { echo "ERROR: $INDICES_FILE not found"; exit 1; }
N=$(python3 -c "import json;d=json.load(open('$INDICES_FILE'));print(len(d.get('indices') or d.get('new_indices') or []))")
echo "  $INDICES_FILE  ->  $N tasks"

echo "== Launching SWE-agent batch via unified CLI (modal, concurrency=$CONCURRENCY) =="
mkdir -p "$OUTPUT_DIR"
python -m agent_cap.agents \
    --strategy sweagent \
    --model "$MODEL" \
    --base-url "$LLM_URL" \
    --api-key dummy \
    --dataset swe-bench-lite \
    --task-indices "$INDICES_FILE" \
    --sweagent-deployment modal \
    --sweagent-dir "$SWEAGENT_DIR" \
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
