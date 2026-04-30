#!/bin/bash
set -e
export OPENAI_API_KEY="${OPENAI_API_KEY:?please set OPENAI_API_KEY}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?please set OPENROUTER_API_KEY}"
cd /home/sicheng/AgentCAP

LOG=/data/sicheng/agentcap_logs

for cfg in gpt54_qwen35_27b minimax_qwen35_27b glm_qwen35_27b; do
    echo "=== $(date) running imo_${cfg} ==="
    python scripts/run_hybrid_experiment.py \
        --config "configs/hybrid_${cfg}_imo.yaml" \
        --dataset imo-answerbench --num-tasks 100 \
        --db "results/imo_${cfg}.db" 2>&1 | tee "$LOG/imo_${cfg}.log"
done
echo "TRACK QWEN DONE"
