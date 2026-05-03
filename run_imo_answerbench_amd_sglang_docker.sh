#!/usr/bin/env bash
set -euo pipefail

# ==========
# Host config
# ==========
IMAGE="lmsysorg/sglang:v0.5.9-rocm700-mi35x"

HOST_ROOT="/home/amd/yufan"
HOST_REPO_DIR="${HOST_ROOT}/AgentCAP"
HOST_MODELS_DIR="${HOST_ROOT}/models"
HOST_OUTPUTS_DIR="${HOST_ROOT}/outputs"
HOST_HF_CACHE_DIR="${HOST_ROOT}/hf_cache"
HOST_PIP_CACHE_DIR="${HOST_ROOT}/pip_cache"

# Use host physical AMD GPU index here.
# Example:
#   GPU_IDS=5 ./run_imo_answerbench_amd_sglang_docker.sh
GPU_IDS="${GPU_IDS:-0}"

PORT="${PORT:-8000}"

# ==========
# Required secrets
# ==========
: "${OPENROUTER_API_KEY:?Please set OPENROUTER_API_KEY in the host environment}"

mkdir -p \
  "${HOST_MODELS_DIR}" \
  "${HOST_OUTPUTS_DIR}" \
  "${HOST_HF_CACHE_DIR}/hub" \
  "${HOST_HF_CACHE_DIR}/transformers" \
  "${HOST_HF_CACHE_DIR}/datasets" \
  "${HOST_PIP_CACHE_DIR}"

echo "Pulling ${IMAGE}..."
sudo docker pull "${IMAGE}"

echo "Starting SGLang ROCm container on ROCR_VISIBLE_DEVICES=${GPU_IDS}..."

sudo docker run --rm \
  --network=host \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "${HOST_REPO_DIR}:/workspace/AgentCAP" \
  -v "${HOST_MODELS_DIR}:/workspace/models" \
  -v "${HOST_OUTPUTS_DIR}:/workspace/outputs" \
  -v "${HOST_HF_CACHE_DIR}:/workspace/hf_cache" \
  -v "${HOST_PIP_CACHE_DIR}:/workspace/pip_cache" \
  -e ROCR_VISIBLE_DEVICES="${GPU_IDS}" \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
  -e HF_HOME="/workspace/hf_cache" \
  -e HUGGINGFACE_HUB_CACHE="/workspace/hf_cache/hub" \
  -e TRANSFORMERS_CACHE="/workspace/hf_cache/transformers" \
  -e HF_DATASETS_CACHE="/workspace/hf_cache/datasets" \
  -e HF_LOCAL_MODEL_ROOT="/workspace/models" \
  -e PIP_CACHE_DIR="/workspace/pip_cache" \
  -e TRANSFORMERS_NO_TF=1 \
  -e TRANSFORMERS_NO_FLAX=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e SGLANG_LOG_LEVEL=INFO \
  -e OUTPUT_ROOT="/workspace/outputs/TEAS_Development_Results_Private/agentic_results/amd/sglang" \
  -w /workspace/AgentCAP \
  "${IMAGE}" \
  bash -lc '
set -euo pipefail

echo "===== Container environment ====="
which python
python -V

echo "===== ROCm / Torch / SGLang checks ====="
python - << "PY"
import os, sys
import torch
import sglang

print("python =", sys.executable)
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("torch file =", torch.__file__)
print("sglang =", getattr(sglang, "__version__", "unknown"))
print("sglang file =", sglang.__file__)
print("cuda available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    try:
        print(i, torch.cuda.get_device_name(i))
    except Exception as e:
        print(i, "device name unavailable:", e)
print("ROCR_VISIBLE_DEVICES =", os.getenv("ROCR_VISIBLE_DEVICES"))
print("HIP_VISIBLE_DEVICES =", os.getenv("HIP_VISIBLE_DEVICES"))
print("OPENROUTER_API_KEY set =", bool(os.getenv("OPENROUTER_API_KEY")))
print("HF_HOME =", os.getenv("HF_HOME"))
PY

echo "===== Installing AgentCAP project dependencies ====="
python -m pip install --upgrade pip setuptools wheel

python -m pip install --upgrade-strategy only-if-needed \
  requests \
  aiohttp \
  openai \
  huggingface-hub \
  jupyter-client \
  ipykernel \
  math-verify \
  openai-harmony \
  pyyaml \
  sympy

echo "===== AgentCAP import check ====="
python - << "PY"
import requests
import aiohttp
import openai
import huggingface_hub
import jupyter_client
import ipykernel
import math_verify
import openai_harmony

import agent_cap
from agent_cap.benchmarks import load_benchmark
from agent_cap.backends.math_python_backend import MathPythonBackend

print("AgentCAP imports OK")
PY

echo "===== Running benchmark ====="
python -m agent_cap.run_imo_answerbench_5 \
  --model gpt-oss \
  --num-tasks 25 \
  --max-turns 128 \
  --max-tokens 131072 \
  --temperature 1.0 \
  --startup-timeout 300 \
  --exec-timeout 5 \
  --preload minimal \
  --auto-print-last-expr \
  --port '"${PORT}"' \
  --served-model-name gpt-oss \
  --model-path /workspace/models/gpt-oss-120b \
  --dtype auto \
  --kv-cache-dtype fp8_e4m3 \
  --context-tokens 131072 \
  --mem-fraction-static 0.85 \
  --tensor-parallel-size 1 \
  --server-timeout 3600 \
  --preload-workers 8 \
  --judge-model google/gemini-3.1-flash-lite-preview
'