#!/usr/bin/env bash
set -euo pipefail

cd /home/amd/yufan/AgentCAP

unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export HIP_VISIBLE_DEVICES=5

unset VLLM_ROCM_ATTENTION_BACKEND
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0
export HSA_NO_SCRATCH_RECLAIM=1
export AMDGCN_USE_BUFFER_OPS=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

export OUTPUT_ROOT="/home/amd/yufan/outputs/TEAS_Development_Results_Private/agentic_results/amd/vllm"
mkdir -p "$OUTPUT_ROOT"

python -m agent_cap.run_imo_answerbench_4 \
  --model gpt-oss \
  --num-tasks 25 \
  --max-turns 128 \
  --max-tokens 131072 \
  --temperature 1.0 \
  --startup-timeout 30 \
  --exec-timeout 5 \
  --preload minimal \
  --auto-print-last-expr \
  --port 8000 \
  --served-model-name gpt-oss \
  --model-path /home/amd/yufan/models/gpt-oss-120b \
  --kv-cache-dtype fp8 \
  --dtype auto \
  --stream-interval 1 \
  --context-tokens 131072 \
  --batch-size 16 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --server-timeout 3600 \
  --preload-workers 16 \
  --judge-model google/gemini-3.1-flash-lite-preview
