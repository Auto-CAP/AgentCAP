#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ATLAS_DIR="${MCP_ATLAS_DIR:-$REPO_ROOT/third_party/mcp-atlas}"
AGENT_ENV_DIR="$ATLAS_DIR/services/agent-environment"
ENV_FILE="${MCP_ENV_FILE:-$ATLAS_DIR/.env}"
DATA_DIR="${MCP_DATA_DIR:-/data}"
PORT="${MCP_PORT:-1984}"
WORKERS="${MCP_WORKERS:-1}"
SKIP_PREINSTALL="${MCP_SKIP_PREINSTALL:-0}"

ENABLED_SERVERS="${ENABLED_SERVERS:-arxiv,brave-search,calculator,cli-mcp-server,clinicaltrialsgov-mcp-server,context7,ddg-search,desktop-commander,fetch,filesystem,git,github,mcp-code-executor,mcp-server-code-runner,memory,met-museum,open-library,osm-mcp-server,pubmed,weather,whois,wikipedia}"

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

if [ ! -d "$AGENT_ENV_DIR" ]; then
    echo "Error: $AGENT_ENV_DIR not found. Set MCP_ATLAS_DIR to your mcp-atlas clone."
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "First run: creating venv and installing dependencies..."
    # Match the docker base image (ghcr.io/astral-sh/uv:python3.12-bookworm-slim)
    uv venv "$SCRIPT_DIR/.venv" --python 3.12
    source "$SCRIPT_DIR/.venv/bin/activate"
    uv pip install -r "$SCRIPT_DIR/pyproject.toml"
    # Install agent-environment WITH its own deps (requests, etc.) to match docker `uv sync`
    uv pip install -e "$AGENT_ENV_DIR"
else
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Pre-install npm/uvx MCP server packages globally (mirrors docker
# install_mcp_packages.sh). Skipped if already done or via MCP_SKIP_PREINSTALL=1.
PREINSTALL_SCRIPT="$AGENT_ENV_DIR/dev_scripts/install_mcp_packages.sh"
PREINSTALL_MARKER="$SCRIPT_DIR/.venv/.mcp_packages_installed"
if [ "$SKIP_PREINSTALL" != "1" ] && [ ! -f "$PREINSTALL_MARKER" ] && [ -f "$PREINSTALL_SCRIPT" ]; then
    echo "Pre-installing MCP server packages (npm + uvx, ~5-10 min)..."
    bash "$PREINSTALL_SCRIPT"
    touch "$PREINSTALL_MARKER"
fi

GITHUB_MCP_DIR="$DATA_DIR/.mcp_servers/github"
if [ ! -f "$GITHUB_MCP_DIR/dist/cli.js" ]; then
    echo "Building github MCP (one-time, ~30s)..."
    mkdir -p "$GITHUB_MCP_DIR"
    git clone --quiet --depth 1 https://github.com/geobio/smitheryai-mcp-servers-github "$GITHUB_MCP_DIR"
    (cd "$GITHUB_MCP_DIR" && npm install --silent && npx --yes tsc)
fi

# Populate $DATA_DIR with the bundled assets + cloned repos (mirrors docker
# `mv /agent-environment/data/* /data/` + git submodule loop).
SUBMODULE_CSV="$AGENT_ENV_DIR/data/repos/git_submodule_info.csv"
if [ -f "$SUBMODULE_CSV" ]; then
    mkdir -p "$DATA_DIR/repos"
    if [ ! -d "$DATA_DIR/repos/mcp_code_executor_workspace" ] && [ -d "$AGENT_ENV_DIR/data/repos/mcp_code_executor_workspace" ]; then
        cp -r "$AGENT_ENV_DIR/data/repos/mcp_code_executor_workspace" "$DATA_DIR/repos/"
    fi
    while IFS=',' read -r url sha path; do
        target="$DATA_DIR/repos/$(basename "$path")"
        if [ ! -d "$target/.git" ]; then
            echo "Cloning $url -> $target"
            git clone --quiet "$url" "$target" && (cd "$target" && git checkout --quiet "$sha")
        fi
    done < "$SUBMODULE_CSV"
    # mcp-code-executor venv (docker does `uv sync` inside this dir).
    EXECUTOR_DIR="$DATA_DIR/repos/mcp_code_executor_workspace"
    if [ -d "$EXECUTOR_DIR" ] && [ ! -d "$EXECUTOR_DIR/.venv" ]; then
        echo "Setting up mcp_code_executor_workspace venv..."
        (cd "$EXECUTOR_DIR" && uv sync --quiet)
    fi
fi

MCP_PORT_SAVED="$PORT"
MCP_SERVERS_SAVED="$ENABLED_SERVERS"
set -a && source "$ENV_FILE" && set +a
export ENABLED_SERVERS="$MCP_SERVERS_SAVED"
export PATH="/usr/bin:$PATH"

export DATA_ROOT="$DATA_DIR"
envsubst < "$AGENT_ENV_DIR/src/agent_environment/mcp_server_template.json" \
    > "$AGENT_ENV_DIR/src/agent_environment/mcp_server_config.json"

echo "Starting MCP server on port $MCP_PORT_SAVED ($WORKERS worker(s)) with servers: $ENABLED_SERVERS"
exec uvicorn agent_environment.main:app --host 0.0.0.0 --port "$MCP_PORT_SAVED" --timeout-keep-alive 300 --workers "$WORKERS"
