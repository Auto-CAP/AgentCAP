#!/usr/bin/env bash
# DIRECT VERSION: uses the dedicated `agent_cap.cli mcp-atlas` streaming
# runner directly (NOT the unified `agent_cap.agents` CLI).
# Kept as a reference / fallback path. See run_mcpatlas_local_60.sh for
# the unified version.
#
# Required env: BRAVE_API_KEY + GITHUB_TOKEN in $MCP_ENV_FILE.
# Required runtime: LLM server on :8000 (vLLM or SGLang) gpt-oss-120b 131072 ctx
#   SGLang: --reasoning-parser gpt-oss --tool-call-parser gpt-oss
#           --context-length 131072 --enable-cache-report
#   vLLM:   --reasoning-parser openai_gptoss --tool-call-parser openai
#           --max-model-len 131072 --enable-prompt-tokens-details
#
# USAGE: bash scripts/run_mcpatlas_local_60_direct.sh [--data-dir DIR] [--output-dir DIR]
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

DATA_DIR="${MCP_DATA_DIR:-/data}"
OUTPUT_DIR="/data/sicheng/agent-team-data/gptoss-120b_mcpatlas_60free_local_direct"
LLM_URL="http://localhost:8000"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --llm-url) LLM_URL="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

echo "== Verifying LLM endpoint =="
curl -sS -m 5 -o /dev/null -w "  $LLM_URL  ->  %{http_code}\n" "$LLM_URL/v1/models" || {
    echo "ERROR: LLM not reachable at $LLM_URL"; exit 1
}

echo "== Verifying .env =="
ENV_FILE="${MCP_ENV_FILE:-$REPO_ROOT/third_party/mcp-atlas/.env}"
[[ -f "$ENV_FILE" ]] || { echo "ERROR: $ENV_FILE not found"; exit 1; }
grep -E "^(BRAVE_API_KEY|GITHUB_TOKEN)=" "$ENV_FILE" > /dev/null || {
    echo "ERROR: BRAVE_API_KEY + GITHUB_TOKEN required in $ENV_FILE"; exit 1
}

echo "== Starting local FastMCP server (if not already up) =="
if ! curl -sS -m 3 -o /dev/null http://localhost:1984/list-tools 2>/dev/null; then
    if ! tmux has-session -t mcplocal 2>/dev/null; then
        tmux new-session -d -s mcplocal -x 200 -y 50
    fi
    tmux send-keys -t mcplocal "cd $REPO_ROOT && MCP_DATA_DIR=$DATA_DIR MCP_ENV_FILE=$ENV_FILE bash mcp-server/start.sh 2>&1 | tee \$HOME/mcp_local.log" Enter
    for i in {1..90}; do
        curl -sS -m 2 -o /dev/null http://localhost:1984/list-tools 2>/dev/null && break
        sleep 2
    done
fi
curl -sS -m 5 -o /dev/null -w "  http://localhost:1984/list-tools  ->  %{http_code}\n" http://localhost:1984/list-tools || {
    echo "ERROR: local MCP server failed to start"; exit 1
}

echo "== Running MCP-Atlas 60-task free subset (dedicated streaming runner) =="
mkdir -p "$OUTPUT_DIR"
MCP_PROMPT_DATA_ROOT="$DATA_DIR" \
python -m agent_cap.cli mcp-atlas configs/mcp_atlas_60.yaml \
    --free-only --base-url "$LLM_URL" \
    --mcp-server-url http://localhost:1984 \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "== Done.  Results: $OUTPUT_DIR/results.jsonl =="
wc -l "$OUTPUT_DIR/results.jsonl" 2>/dev/null
