#!/bin/bash
set -e
cd /home/sicheng/AgentCAP

NUM_TASKS=60
LOG_DIR=/data/sicheng/agentcap_logs
mkdir -p "$LOG_DIR"

CONFIGS=(
    hybrid_gpt54_qwen35_27b
    hybrid_claude46_qwen35_27b
    hybrid_minimax_qwen35_27b
    hybrid_glm_qwen35_27b
    hybrid_gpt54_gptoss120b
    hybrid_claude46_gptoss120b
    hybrid_minimax_gptoss120b
    hybrid_glm_gptoss120b
)

for cfg in "${CONFIGS[@]}"; do
    db="results/hybrid_${cfg#hybrid_}.db"
    log="$LOG_DIR/${cfg}.log"
    echo "================================================"
    echo "[$(date)] Running $cfg -> $db"
    echo "================================================"
    python scripts/run_hybrid_experiment.py \
        --config "configs/${cfg}.yaml" \
        --num-tasks "$NUM_TASKS" \
        --db "$db" 2>&1 | tee "$log"
    echo "[$(date)] Finished $cfg"
    echo
done

echo "ALL DONE"
