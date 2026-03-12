#!/bin/bash

set -euo pipefail

SGLANG_PYTHON="/mnt/raid0nvme0/sicheng/miniconda3/envs/sglang/bin/python"

GPU_SMALL="0"
GPU_LARGE="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-small)
            GPU_SMALL="$2"
            shift 2
            ;;
        --gpu-large)
            GPU_LARGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "Checking Docker MCP environment..."
docker ps | grep mcp-atlas-env >/dev/null || {
    echo "Starting MCP-Atlas Docker..."
    docker run --rm -d --name mcp-atlas-env -p 1984:1984 \
        --env-file /home/sicheng/mcp-atlas/.env agent-environment:latest
}

echo "Launching SGLang servers with tool-call-parser..."

(
    cd /tmp
    CUDA_VISIBLE_DEVICES="$GPU_SMALL" "$SGLANG_PYTHON" -m sglang.launch_server \
        --model-path Qwen/Qwen3-30B-A3B --tp 1 --port 30000 \
        --trust-remote-code --mem-fraction-static 0.85 \
        --tool-call-parser qwen25
) &

(
    cd /tmp
    CUDA_VISIBLE_DEVICES="$GPU_LARGE" "$SGLANG_PYTHON" -m sglang.launch_server \
        --model-path Qwen/Qwen3-32B --tp 1 --port 30001 \
        --trust-remote-code --mem-fraction-static 0.85 \
        --tool-call-parser qwen25
) &

echo "Waiting for servers..."
for port in 30000 30001; do
    for i in $(seq 1 120); do
        curl -s "http://localhost:${port}/health" >/dev/null 2>&1 && {
            echo "Port ${port} ready"
            break
        }
        sleep 5
    done
done

echo "All services ready. Run:"
echo "  $SGLANG_PYTHON scripts/mcpatlas_combo.py --config configs/mcpatlas_pairB.yaml --num-tasks 50"
