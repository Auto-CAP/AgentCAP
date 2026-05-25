# MCP Server (self-hosted, no Docker)

`mcp-server/start.sh` runs the `mcp-atlas` agent-environment locally —
a FastAPI service on port 1984 that brokers ~21 MCP tool servers over
stdio. This is the docker-free equivalent of
`ghcr.io/scaleapi/mcp-atlas:latest`.

## Prerequisites

- `python3.12`, `uv` (`pip install uv`)
- `node >= 20`, `npm`
- `envsubst` (in `gettext-base` on Debian/Ubuntu, `gettext` on macOS)

## First run

```bash
git submodule update --init --recursive
bash mcp-server/start.sh
```

The first invocation copies `third_party/mcp-atlas/env.template` to
`.env` and exits. Fill in the API keys you need (see
[keys reference](#api-keys)), then re-run.

```bash
$EDITOR third_party/mcp-atlas/.env
bash mcp-server/start.sh
```

The script:

1. Creates `mcp-server/.venv` (Python 3.12) and installs
   `agent-environment` with its deps.
2. Sources `.env` and runs `envsubst` on
   `mcp_server_template.json` → `mcp_server_config.json`.
3. Launches `uvicorn agent_environment.main:app --port 1984`.

On startup you should see `160 tools loaded in total` and
`Uvicorn running on http://0.0.0.0:1984`.

## Verify

```bash
curl http://localhost:1984/health
# {"status":"health_and_client_connection_ok"}

curl -s http://localhost:1984/enabled-servers | python -m json.tool
# Per-server OK / ERROR_NOT_ONLINE status

curl -s -X POST http://localhost:1984/call-tool \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"calculator_calculate","tool_args":{"expression":"2+2"}}'
```

## Configuration

All knobs are environment variables — no flags.

| Var | Default | Purpose |
|---|---|---|
| `MCP_PORT` | `1984` | HTTP port |
| `MCP_WORKERS` | `1` | uvicorn worker count (raise for >100 concurrent calls) |
| `ENABLED_SERVERS` | 21 free-tier servers | Comma-separated list; empty = 20 defaults + auto-detect by key |
| `MCP_ATLAS_DIR` | `third_party/mcp-atlas` | Use an external clone |
| `MCP_ENV_FILE` | `$MCP_ATLAS_DIR/.env` | Alternate env file |

Examples:

```bash
MCP_PORT=2000 bash mcp-server/start.sh
MCP_WORKERS=4 bash mcp-server/start.sh                # higher throughput
ENABLED_SERVERS=calculator,wikipedia,fetch bash mcp-server/start.sh
```

## API keys

20 servers need no key (arxiv, calculator, fetch, filesystem, git,
wikipedia, …). The other 16 require keys; see
`third_party/mcp-atlas/env.template` for the full list and signup
links. Free-tier sufficient for benchmarks:

- `BRAVE_API_KEY` — 2000 queries/mo free
- `GITHUB_PERSONAL_ACCESS_TOKEN` — free PAT
- `EXA_API_KEY` — ~1000 searches/mo free
- `NOTION_TOKEN`, `AIRTABLE_API_KEY`, `NPS_API_KEY` — free

Paid-only: `OXYLABS_*`, `LARA_ACCESS_KEY_*`. Skip these unless you
explicitly need scraping or translation servers.

Servers with empty / missing keys still start; they return auth errors
only when called. This is harmless for benchmark scoring.

## Use with AgentCAP runner

```bash
python -m agent_cap.agents \
  --strategy single \
  --model openai/gpt-oss-120b \
  --base-url http://localhost:30000/v1 \
  --tool-backend mcp \
  --mcp-server-url http://localhost:1984 \
  --dataset mcp-atlas \
  --num-tasks 5
```

## Stop / restart

```bash
pkill -f "uvicorn agent_environment.main"
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `address already in use 1984` | `pkill -f "uvicorn agent_environment"` or use `MCP_PORT=1985` |
| `npm: command not found` | install Node ≥20 (`nvm install 20`) |
| `envsubst: command not found` | `apt install gettext-base` / `brew install gettext` |
| `ModuleNotFoundError: requests` | Stale venv. `rm -rf mcp-server/.venv && bash mcp-server/start.sh` |
| Server `ERROR_NOT_ONLINE` | Check stderr — usually missing key or unreachable git submodule |
| First call slow | npx/uvx downloading the server package on demand; 5–15 min on cold cache, instant after |

## Concurrency notes

| Load | Setting |
|---|---|
| ≤100 in-flight calls | `MCP_WORKERS=1` (default) |
| 100–500 | `MCP_WORKERS=4` |
| 500+ | `MCP_WORKERS=8` plus consider splitting heavy servers (`github`, `brave-search`) to separate ports |

Each worker spawns its own FastMCP Client and its own set of stdio
subprocesses — 4 workers ≈ 4× memory and 4× cold-start time. The
48h in-memory `tool_cache` is per-worker (not shared).
