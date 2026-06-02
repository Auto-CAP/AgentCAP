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
| `ENABLED_SERVERS` | 22 free-tier servers (matches `_FREE_SERVERS` in `unified_runner.py`) | Comma-separated list; empty = 20 defaults + auto-detect by key |
| `MCP_ATLAS_DIR` | `third_party/mcp-atlas` | Use an external clone |
| `MCP_ENV_FILE` | `$MCP_ATLAS_DIR/.env` | Alternate env file |
| `MCP_DATA_DIR` | `/data` | Where seed files (repos, CSVs, memory.json) live and where MCP servers read/write. Override when `/data` isn't writable (most non-root setups). |

Examples:

```bash
MCP_PORT=2000 bash mcp-server/start.sh
MCP_WORKERS=4 bash mcp-server/start.sh                # higher throughput
ENABLED_SERVERS=calculator,wikipedia,fetch bash mcp-server/start.sh

# Non-root setup: redirect /data to a writable location
MCP_DATA_DIR=$HOME/AgentCAP/mcp-atlas-data bash mcp-server/start.sh
```

`start.sh` substitutes `${DATA_ROOT}` in the MCP server config template with `MCP_DATA_DIR`, so `cli-mcp-server`, `filesystem`, `mcp-code-executor`, `memory`, etc. all root at the writable path. Docker default keeps `/data`.

## API keys

| Server | Key | Note |
|---|---|---|
| arxiv, calculator, cli-mcp-server, clinicaltrialsgov-mcp-server, context7, ddg-search, desktop-commander, fetch, filesystem, git, mcp-code-executor, mcp-server-code-runner, memory, met-museum, open-library, osm-mcp-server, pubmed, weather, whois, wikipedia | none | always free |
| `github` | `GITHUB_TOKEN` (or `GITHUB_PERSONAL_ACCESS_TOKEN`) | `gh auth token` works; free PAT |
| `brave-search` | `BRAVE_API_KEY` | 2000 q/mo free |
| `exa`, `notion`, `airtable`, `national-parks`, `alchemy`, `slack`, `mongodb`, `google-maps`, `google-workspace`, `twelve-data`, `e2b` | various | optional — only if you enable that server |
| `oxylabs`, `lara` | paid | don't bother |

**For the 60-task `mcp-atlas` free subset only `BRAVE_API_KEY` (2 tasks) and `GITHUB_TOKEN` (37 tasks) actually matter.** Other servers in the free set are keyless.

Missing keys → server still starts, only fails when called. Harmless.

## Use with AgentCAP runner

```bash
python -m agent_cap.agents \
  --strategy single \
  --model openai/gpt-oss-120b \
  --base-url http://localhost:30000/v1 \
  --tool-backend mcp \
  --mcp-server-url http://localhost:1984 \
  --dataset mcp-atlas \
  --num-tasks 60 \
  --evaluator gtfa \
  --use-streaming \
  --output-dir /data/results/run1
```

### Full alignment with Docker (`ghcr.io/scaleapi/mcp-atlas:latest`)

To match the Docker container's scoring you also need:

1. **Seed `/data` with files baked into the Docker image** (memory.json, root-level CSVs). One-time copy from a running container:
   ```bash
   docker cp <container>:/data/. $MCP_DATA_DIR/
   ```

2. **`MCP_PROMPT_DATA_ROOT`** — rewrites literal `/data/...` paths in dataset prompts & GTFA claims to match `MCP_DATA_DIR` (only affects ~2 prompts but the claims rewrite is needed for the judge):
   ```bash
   MCP_PROMPT_DATA_ROOT=$MCP_DATA_DIR python -m agent_cap.agents ...
   ```

3. **vllm flags for gpt-oss family** (otherwise context defaults to 16384 and ~11/60 tasks 400-out):
   ```bash
   vllm serve unsloth/gpt-oss-120b \
     --reasoning-parser openai_gptoss \
     --enable-auto-tool-choice --tool-call-parser openai \
     --max-model-len 131072 \
     --seed 0
   ```

With all of the above, self-host ≈ Docker (±2 pass on 60-task `mcp-atlas` free subset due to vllm MoE non-determinism + network jitter).

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
| `mkdir: /data/repos: permission denied` during preinstall | `/data` is root-owned. Set `MCP_DATA_DIR=$HOME/...` to a writable path. |
| `FastMCP mounted at: 'github': Client failed to connect` | First-time build of github MCP; `start.sh` auto-builds it into `$MCP_DATA_DIR/.mcp_servers/github`. If still failing, `rm -rf $MCP_DATA_DIR/.mcp_servers/github` and re-run. |
| `Input length (N) exceeds model's maximum context length (16384)` | vllm default ctx is too small. Add `--max-model-len 131072` to vllm. |
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
