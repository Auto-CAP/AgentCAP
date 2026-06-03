#!/usr/bin/env bash
# DIRECT (uses the dedicated CLI): uses `scripts/run_sweagent.py --deployment modal` (NOT unified `agent_cap.agents`).
# See run_swebench_modal_100.sh for the unified version.
#
# Required runtime: modal auth (~/.modal.toml), boto3, /tmp/swe_agent checkout,
#   cloudflared if --llm-url not passed, LLM on :8000 (gpt-oss-120b, 131072 ctx).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

LLM_URL="${LLM_URL:-}"
OUTPUT_DIR="/data/sicheng/agent-team-data/gptoss-120b_sweagent_swebenchlite_curated100_modal_direct"
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

[[ -f "$HOME/.modal.toml" ]] || { echo "ERROR: run 'modal token new' first"; exit 1; }
python3 -c "import boto3" 2>/dev/null || { echo "ERROR: pip install boto3"; exit 1; }
[[ -f "$SWEAGENT_DIR/sweagent/agent/models.py" ]] || {
    echo "ERROR: $SWEAGENT_DIR/sweagent/agent/models.py not found"; exit 1
}
grep -q "SWEAGENT_STREAM_STATS_PATH" "$SWEAGENT_DIR/sweagent/agent/models.py" || {
    echo "ERROR: $SWEAGENT_DIR not patched for streaming stats"; exit 1
}

if [[ -z "$LLM_URL" ]]; then
    command -v cloudflared >/dev/null || { echo "ERROR: cloudflared not installed"; exit 1; }
    curl -sS -m 5 -o /dev/null http://localhost:8000/v1/models || {
        echo "ERROR: local LLM not reachable on :8000"; exit 1
    }
    if ! pgrep -f "cloudflared tunnel --url http://localhost:8000" >/dev/null; then
        nohup cloudflared tunnel --url http://localhost:8000 --no-autoupdate \
            > "$HOME/cloudflared_8000.log" 2>&1 &
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

curl -sS -m 10 -o /dev/null -w "  $LLM_URL  ->  %{http_code}\n" "$LLM_URL/models" || {
    echo "ERROR: LLM not reachable at $LLM_URL"; exit 1
}

[[ -f "$INDICES_FILE" ]] || { echo "ERROR: $INDICES_FILE not found"; exit 1; }

echo "== Launching SWE-agent batch (modal, concurrency=$CONCURRENCY, legacy run_sweagent.py) =="
mkdir -p "$OUTPUT_DIR"
python scripts/run_sweagent.py \
    --deployment modal \
    --dataset swe-bench-lite \
    --task-indices "$INDICES_FILE" \
    --vllm-url "$LLM_URL" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --sweagent-dir "$SWEAGENT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "== Done.  Outputs =="
echo "  predictions.json :  $OUTPUT_DIR/predictions.json"
echo "  results.jsonl    :  $OUTPUT_DIR/results.jsonl"
tail -1 "$OUTPUT_DIR/run.log" 2>/dev/null
