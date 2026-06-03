#!/usr/bin/env bash
# Run MCP-Atlas 60-task free-tier subset against a LOCAL FastMCP server.
#
# REQUIRED ENV (in $MCP_ENV_FILE, default: third_party/mcp-atlas/.env):
#   BRAVE_API_KEY    https://api.search.brave.com/app/keys
#   GITHUB_TOKEN     https://github.com/settings/tokens (no scopes needed for public)
# Other env vars in mcp_server_template.json are unused by the 60-task subset
# (their MCP servers are paid-only and excluded).
#
# REQUIRED RUNTIME:
#   - LLM server on :8000 (vLLM or SGLang), serving gpt-oss-120b at full context
#     SGLang: --reasoning-parser gpt-oss --tool-call-parser gpt-oss
#             --context-length 131072 --enable-cache-report
#     vLLM:   --reasoning-parser openai_gptoss --tool-call-parser openai
#             --max-model-len 131072 --enable-prompt-tokens-details
#   - uv (for local MCP venv): https://github.com/astral-sh/uv
#   - npx/node 20+ (for npm-based MCP servers)
#   - Local cache dir for MCP data (default /data), writable by current user.
#
# USAGE:
#   bash scripts/run_mcpatlas_local_60.sh [--data-dir DIR] [--output-dir DIR]
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

DATA_DIR="${MCP_DATA_DIR:-/data}"
OUTPUT_DIR="/data/sicheng/agent-team-data/gptoss-120b_mcpatlas_60free_local"
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
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found.  Copy third_party/mcp-atlas/env.template and fill BRAVE_API_KEY + GITHUB_TOKEN."
    exit 1
fi
grep -E "^(BRAVE_API_KEY|GITHUB_TOKEN)=" "$ENV_FILE" > /dev/null || {
    echo "ERROR: BRAVE_API_KEY and GITHUB_TOKEN must be set in $ENV_FILE"; exit 1
}

echo "== Starting local FastMCP server (if not already up) =="
if ! curl -sS -m 3 -o /dev/null -w "" http://localhost:1984/tools/list 2>/dev/null; then
    if ! tmux has-session -t mcplocal 2>/dev/null; then
        tmux new-session -d -s mcplocal -x 200 -y 50
    fi
    tmux send-keys -t mcplocal "cd $REPO_ROOT && MCP_DATA_DIR=$DATA_DIR MCP_ENV_FILE=$ENV_FILE bash mcp-server/start.sh 2>&1 | tee \$HOME/mcp_local.log" Enter
    echo "  waiting for :1984 ..."
    for i in {1..90}; do
        curl -sS -m 2 -o /dev/null http://localhost:1984/list-tools 2>/dev/null && break
        sleep 2
    done
fi
curl -sS -m 5 -o /dev/null -w "  http://localhost:1984/list-tools  ->  %{http_code}\n" http://localhost:1984/list-tools || {
    echo "ERROR: local MCP server failed to start; see ~/mcp_local.log"; exit 1
}

echo "== Running MCP-Atlas 60-task free subset (unified CLI) =="
mkdir -p "$OUTPUT_DIR"
MCP_PROMPT_DATA_ROOT="$DATA_DIR" \
python -m agent_cap.agents \
    --strategy single \
    --model unsloth/gpt-oss-120b \
    --base-url "$LLM_URL/v1" \
    --api-key dummy \
    --use-streaming \
    --dataset mcp-atlas \
    --num-tasks 60 \
    --evaluator gtfa \
    --tool-backend mcp \
    --mcp-server-url http://localhost:1984 \
    --max-turns 20 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "== Done. Results: $OUTPUT_DIR/results.jsonl =="
wc -l "$OUTPUT_DIR/results.jsonl" 2>/dev/null
