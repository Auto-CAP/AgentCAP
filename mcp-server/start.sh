#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ATLAS_DIR="${MCP_ATLAS_DIR:-$REPO_ROOT/third_party/mcp-atlas}"
ENV_FILE="${MCP_ENV_FILE:-$ATLAS_DIR/.env}"
PORT="${MCP_PORT:-1984}"

ENABLED_SERVERS="${ENABLED_SERVERS:-calculator,fetch,whois,weather,pubmed,brave-search,exa,google-maps,github,airtable,alchemy,clinicaltrialsgov-mcp-server,context7,ddg-search,lara-translate,met-museum,open-library,osm-mcp-server,mongodb}"

if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ATLAS_DIR/env.template" ]; then
        cp "$ATLAS_DIR/env.template" "$ENV_FILE"
        echo "Created $ENV_FILE from template."
        echo "Fill in your API keys, then re-run this script."
        echo "See $ATLAS_DIR/env.template for links to obtain each key."
        exit 1
    else
        echo "Error: $ENV_FILE not found. Set MCP_ENV_FILE to your .env path."
        exit 1
    fi
fi

AGENT_ENV_DIR="$ATLAS_DIR/services/agent-environment"
if [ ! -d "$AGENT_ENV_DIR" ]; then
    echo "Error: $AGENT_ENV_DIR not found. Set MCP_ATLAS_DIR to your mcp-atlas clone."
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "First run: creating venv and installing dependencies..."
    uv venv "$SCRIPT_DIR/.venv" --python 3.13
    source "$SCRIPT_DIR/.venv/bin/activate"
    uv pip install -r "$SCRIPT_DIR/pyproject.toml"
    uv pip install -e "$AGENT_ENV_DIR" --no-deps
else
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

MCP_PORT_SAVED="$PORT"
MCP_SERVERS_SAVED="$ENABLED_SERVERS"
set -a && source "$ENV_FILE" && set +a
export ENABLED_SERVERS="$MCP_SERVERS_SAVED"
export PATH="/usr/bin:$PATH"

envsubst < "$AGENT_ENV_DIR/src/agent_environment/mcp_server_template.json" \
    > "$AGENT_ENV_DIR/src/agent_environment/mcp_server_config.json"

echo "Starting MCP server on port $MCP_PORT_SAVED with servers: $ENABLED_SERVERS"
exec uvicorn agent_environment.main:app --host 0.0.0.0 --port "$MCP_PORT_SAVED" --timeout-keep-alive 300
