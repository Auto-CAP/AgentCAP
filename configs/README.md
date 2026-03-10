# Experiment Configs

YAML files that define experiment sweeps. Each file specifies which models, quantizations, skill subsets, and other variables to sweep.

## Usage

```bash
# Preview all configurations without running
agent-cap run configs/model_size_frontier.yaml --dry-run

# Run the experiment
agent-cap run configs/model_size_frontier.yaml --tasks tasks.json
```

## Available Experiments

| File | Question | Models | Configs |
|------|----------|--------|---------|
| `model_size_frontier.yaml` | What is the smallest model that achieves acceptable quality? | 7 | 14 |
| `quantization_tradeoff.yaml` | How much quality is lost at each quantization level? | 5 | 25 |

## Config Format

```yaml
name: experiment-name          # Unique experiment identifier
description: "..."             # Human-readable description

# Variables to sweep (all combinations are tested)
models:                        # List of models to test
  - id: "Qwen/Qwen3-32B"       # HuggingFace model ID
    arch: dense                # "dense" or "MoE"
    params_b: 32               # Total parameters (billions)
    active_b: 0                # Active parameters for MoE (0 = same as params_b)
    tp: 2                      # Tensor parallel degree
    tier: M                    # Size tier: XL, L, M, S, T

quantizations: ["fp16"]        # Quantization levels to sweep
skill_subsets: ["all", "none"] # Which skills to provide
num_retries: [0]               # Retry counts to test
temperatures: [0.0]            # Temperature values
agent_modes: ["single-pass"]   # Agent modes

# Fixed settings (same for all configs)
serving_engine: sglang         # "sglang" or "vllm"
repetitions: 3                 # Runs per configuration
max_tokens: 4096               # Max generation tokens
gpu_type: H100-80G             # GPU type for cost normalization
num_gpus: 8                    # Available GPUs
```

## Writing Your Own

1. Copy an existing config
2. Modify the model list and sweep variables
3. Run with `--dry-run` to verify the configuration count
4. Total runs = models × quantizations × skill_subsets × num_retries × temperatures × agent_modes × tasks × repetitions
