# AgentCAP

Experiment sweep engine for multi-skill agent evaluation on local GPU clusters.

AgentCAP systematically discovers trade-offs between model size, quantization, skills, and compute cost by sweeping controllable variables across agent tasks. All models run locally via SGLang/vLLM — cost is measured in GPU-seconds, not API dollars.

## Key Features

- **YAML-driven experiment configs** — define sweeps over models, quantizations, skill subsets
- **Local model serving** — auto-manages SGLang/vLLM processes per experiment
- **Novel metrics** — EPD (Effective Precision Depth), SDR, EPR for quantization × pipeline depth analysis
- **Pareto analysis** — compute quality vs. cost frontiers across configurations
- **SQLite results DB** — query and compare results across experiments
- **GPU monitoring** — nvidia-smi polling for utilization, VRAM, power

## Installation

```bash
pip install -e ".[all]"
```

## Quick Start

### 1. Define an experiment

```yaml
# configs/model_size_frontier.yaml
name: model-size-frontier
models:
  - id: "Qwen/Qwen3-32B"
    arch: dense
    params_b: 32
    tp: 2
  - id: "Qwen/Qwen3-8B"
    arch: dense
    params_b: 8
    tp: 1
quantizations: ["fp16"]
skill_subsets: ["all", "none"]
serving_engine: sglang
repetitions: 3
```

### 2. Preview configurations

```bash
agent-cap run configs/model_size_frontier.yaml --dry-run
```

### 3. Run the experiment

```bash
agent-cap run configs/model_size_frontier.yaml --tasks tasks.json
```

### 4. Analyze results

```bash
agent-cap pareto model-size-frontier
agent-cap metrics model-size-frontier
```

## Novel Metrics

| Metric | What it measures |
|--------|-----------------|
| **EPD** (Effective Precision Depth) | Max pipeline depth where quantized model stays within X% of FP16 quality |
| **SDR** (Step Degradation Rate) | How fast quality degrades across pipeline steps |
| **EPR** (Error Propagation Rate) | Whether errors cascade to subsequent steps |
| **SLR** (Skill Leverage Ratio) | Compute-equivalent value of adding skills |
| **MCV** (Marginal Compute Value) | Quality improvement per additional GPU-second |
| **GAR** (GPU Active Ratio) | Fraction of wall-clock time GPU is actually computing |

## Python API

```python
from agent_cap import (
    ExperimentConfig, ModelServerManager, ChatClient,
    compute_sdr, compute_epr, compute_epd,
    compute_pareto_frontier, ParetoPoint,
)

# Load config
config = ExperimentConfig.from_yaml("configs/my_experiment.yaml")

# Compute metrics
sdr = compute_sdr([4.5, 4.2, 3.8, 3.5, 3.0])  # quality per step
epd = compute_epd({1: 1.0, 2: 0.98, 3: 0.93, 4: 0.88}, threshold=0.95)  # safe depth = 2

# Pareto analysis
frontier = compute_pareto_frontier([
    ParetoPoint("32B-fp16", quality=4.2, gpu_seconds=45),
    ParetoPoint("8B-fp16", quality=3.8, gpu_seconds=6),
])
```

## Project Structure

```
agent_cap/
  core/           # Step-level tracing (existing)
  config/         # YAML experiment config loading
  server/         # SGLang/vLLM process management + GPU monitor
  runner/         # Experiment execution loop
  metrics/        # Novel metrics (SDR, EPR, EPD, SLR, MCV, GAR)
  analysis/       # Pareto frontiers + automated insights
  db/             # SQLite results storage
  cli.py          # CLI entrypoint
configs/          # Example YAML experiment specs
examples/         # Usage examples
```

## Acknowledgement

This project is supported by the Advanced Research and Invention Agency (ARIA)'s grant "Scaling Compute: AI at 1/1000th the cost. Technical Area 4 Benchmarking".
