# AgentCAP

Multi-agent combination cost-accuracy-performance analysis for local GPU clusters. Evaluates cascade, vote, best-of-n, adaptive-cascade, generate-verify, self-critique strategies across model pairs using BigCodeBench and MCP-Atlas.

## Installation

```bash
pip install -e ".[all]"
```
Note: Requires SGLang in a conda env. Models served locally, no API keys needed for inference. Python path for SGLang env: `/mnt/raid0nvme0/sicheng/miniconda3/envs/sglang/bin/python`.

## Quick Start: Run a BigCodeBench Combo Experiment

### Step 1: Start SGLang servers
Launch from `/tmp` to avoid import conflicts.
```bash
# GPU 0: Small Model (Qwen3-30B-A3B)
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --port 30000 --cuda-visible-devices 0

# GPU 1: Large Model (Qwen3-32B)
python -m sglang.launch_server --model-path Qwen/Qwen3-32B --port 30001 --cuda-visible-devices 1
```

### Step 2: Run combo experiment
```bash
python -m agent_cap combo configs/qwen_combo_pairB.yaml --benchmark bigcodebench:50 --db results/combo.db
```

### Step 3: Analyze results
```bash
python analyze_results.py --db results/combo.db --output-dir results/analysis/ --pair-label pairB
```

## Run MCP-Atlas (Agentic) Experiments

### Step 1: Start SGLang with tool calling
Qwen3 requires `--tool-call-parser qwen25`. Max tokens must be >= 8192 for thinking tags.
```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-32B --port 30002 --cuda-visible-devices 2 --tool-call-parser qwen25 --context-length 65536
```

### Step 2: Start Docker MCP-Atlas env
```bash
docker run --rm -d --name mcp-atlas-env -p 1984:1984 mcp-atlas-env
```

### Step 3: Run combo experiment
```bash
python scripts/mcpatlas_combo.py --config configs/mcpatlas_pairB_gpu23.yaml --num-tasks 50 --db results/mcpatlas.db
```

### Step 4: Score with GPT-4o
```bash
cd /home/sicheng/mcp-atlas/services/mcp_eval
EVAL_LLM_API_KEY=$OPENAI_API_KEY uv run python mcp_evals_scores.py --input-file="/home/sicheng/AgentCAP/results/mcpatlas/cascade.csv" --model-label="pairB-cascade" --evaluator-model="gpt-4o"
```

## Config Format
Detailed guide in `configs/README.md`. Key fields:
- `small_model`/`large_model`: model ID, port, and GPU placement
- `strategies`: List of methods (e.g., `cascade`, `vote`, `best-of-n-small`)
- `max_tokens`: Generation limit (suggest 8192+)

## Project Structure
```
agent_cap/       # Core multi-agent logic and CLI
configs/         # YAML experiment specifications
results/         # SQLite databases and analysis outputs
scripts/         # MCP-Atlas and utility scripts
analyze_results.py # result visualization and Pareto analysis
```

## Key Results (Pair A: 4B vs 30B-A3B)
| Strategy | Pass@1 | GPU-s | $/correct (H100) |
|---|---|---|---|
| Best-of-N-Small (4B×3) | 50% | 104.4 | $0.070 |
| Cascade | 44% | 23.9 | $0.018 |
| Best-of-N-Large (30B×3) | 44% | 74.7 | $0.057 |
| Adaptive-Cascade | 42% | 41.6 | $0.033 |

## Acknowledgement
This project is supported by the Advanced Research and Invention Agency (ARIA)'s grant "Scaling Compute: AI at 1/1000th the cost. Technical Area 4 Benchmarking".
