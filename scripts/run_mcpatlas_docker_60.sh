#!/usr/bin/env bash
# Run MCP-Atlas 60-task free-tier subset against the OFFICIAL Docker MCP server.
#
# REQUIRED ENV (passed to container via --env-file):
#   BRAVE_API_KEY    https://api.search.brave.com/app/keys
#   GITHUB_TOKEN     https://github.com/settings/tokens (no scopes for public)
# Other env vars in the docker image's mcp_server_template.json are unused by
# the 60-task subset (their MCP servers are paid-only and excluded).
#
# REQUIRED RUNTIME:
#   - Docker daemon
#   - Image:  ghcr.io/scaleapi/mcp-atlas:latest  (auto-pulled)
#   - LLM server on :8000 (vLLM or SGLang) serving gpt-oss-120b at 131072 ctx.
#     SGLang flags: --reasoning-parser gpt-oss --tool-call-parser gpt-oss
#                   --context-length 131072 --enable-cache-report
#     vLLM flags:   --reasoning-parser openai_gptoss --tool-call-parser openai
#                   --max-model-len 131072 --enable-prompt-tokens-details
#
# USAGE:
#   bash scripts/run_mcpatlas_docker_60.sh [--env-file PATH] [--output-dir DIR]
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

ENV_FILE="${MCP_ENV_FILE:-$REPO_ROOT/third_party/mcp-atlas/.env}"
OUTPUT_DIR="/data/sicheng/agent-team-data/gptoss-120b_mcpatlas_60free_docker"
LLM_URL="http://localhost:8000"
IMAGE="ghcr.io/scaleapi/mcp-atlas:latest"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --llm-url) LLM_URL="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

echo "== Verifying LLM endpoint =="
curl -sS -m 5 -o /dev/null -w "  $LLM_URL  ->  %{http_code}\n" "$LLM_URL/v1/models" || {
    echo "ERROR: LLM not reachable at $LLM_URL"; exit 1
}

echo "== Verifying .env =="
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found.  Need at minimum:"
    echo "  BRAVE_API_KEY=..."
    echo "  GITHUB_TOKEN=..."
    exit 1
fi
grep -E "^(BRAVE_API_KEY|GITHUB_TOKEN)=" "$ENV_FILE" > /dev/null || {
    echo "ERROR: BRAVE_API_KEY and GITHUB_TOKEN must be set in $ENV_FILE"; exit 1
}

echo "== Verifying docker daemon =="
docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable"; exit 1; }

echo "== Starting Docker MCP-Atlas (if not already up) =="
if ! docker ps --format '{{.Names}}' | grep -q '^mcp-atlas$'; then
    docker rm -f mcp-atlas 2>/dev/null || true
    docker run --rm -d --name mcp-atlas -p 1984:1984 \
        --env-file "$ENV_FILE" \
        "$IMAGE" >/dev/null
    echo "  waiting for :1984 ..."
    for i in {1..90}; do
        curl -sS -m 2 -o /dev/null http://localhost:1984/list-tools 2>/dev/null && break
        sleep 2
    done
fi
curl -sS -m 5 -o /dev/null -w "  http://localhost:1984/list-tools  ->  %{http_code}\n" http://localhost:1984/list-tools || {
    echo "ERROR: docker mcp-atlas failed to start"
    docker logs mcp-atlas 2>&1 | tail -20
    exit 1
}

echo "== Running MCP-Atlas 60-task free subset (unified CLI) =="
mkdir -p "$OUTPUT_DIR"
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
