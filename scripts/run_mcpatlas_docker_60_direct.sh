#!/usr/bin/env bash
# DIRECT (uses the dedicated CLI): uses dedicated `agent_cap.cli mcp-atlas` runner against the docker MCP container.
# See run_mcpatlas_docker_60.sh for the unified `agent_cap.agents` version.
#
# Required env: BRAVE_API_KEY + GITHUB_TOKEN in $MCP_ENV_FILE.
# Required runtime: docker daemon, ghcr.io/scaleapi/mcp-atlas:latest,
#   LLM server on :8000 (gpt-oss-120b, 131072 ctx, cache-report enabled).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

ENV_FILE="${MCP_ENV_FILE:-$REPO_ROOT/third_party/mcp-atlas/.env}"
OUTPUT_DIR="/data/sicheng/agent-team-data/gptoss-120b_mcpatlas_60free_docker_direct"
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

[[ -f "$ENV_FILE" ]] || { echo "ERROR: $ENV_FILE not found"; exit 1; }
grep -E "^(BRAVE_API_KEY|GITHUB_TOKEN)=" "$ENV_FILE" > /dev/null || {
    echo "ERROR: BRAVE_API_KEY + GITHUB_TOKEN required in $ENV_FILE"; exit 1
}

docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not reachable"; exit 1; }

echo "== Starting Docker MCP-Atlas (if not already up) =="
if ! docker ps --format '{{.Names}}' | grep -q '^mcp-atlas$'; then
    docker rm -f mcp-atlas 2>/dev/null || true
    docker run --rm -d --name mcp-atlas -p 1984:1984 \
        --env-file "$ENV_FILE" "$IMAGE" >/dev/null
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

echo "== Running MCP-Atlas 60-task free subset (dedicated streaming runner) =="
mkdir -p "$OUTPUT_DIR"
python -m agent_cap.cli mcp-atlas configs/mcp_atlas_60.yaml \
    --free-only --base-url "$LLM_URL" \
    --mcp-server-url http://localhost:1984 \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "== Done.  Results: $OUTPUT_DIR/results.jsonl =="
wc -l "$OUTPUT_DIR/results.jsonl" 2>/dev/null
