#!/bin/bash
set -e
export OPENAI_API_KEY="${OPENAI_API_KEY:?please set OPENAI_API_KEY}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?please set OPENROUTER_API_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?please set ANTHROPIC_API_KEY}"
export ANTHROPIC_BASE_URL=https://cc1.zhihuiapi.top
cd /home/sicheng/AgentCAP

NUM_TASKS=60
LOG_DIR=/data/sicheng/agentcap_logs
mkdir -p "$LOG_DIR"

CONFIGS=(
    hybrid_gpt54_self
    hybrid_claude46_self
    hybrid_minimax_self
    hybrid_glm_self
)

for cfg in "${CONFIGS[@]}"; do
    db="results/hybrid_${cfg#hybrid_}.db"
    log="$LOG_DIR/${cfg}.log"
    echo "================================================"
    echo "[$(date)] Running $cfg -> $db"
    echo "================================================"
    /bin/rm -f "$db"
    python scripts/run_hybrid_experiment.py \
        --config "configs/${cfg}.yaml" \
        --num-tasks "$NUM_TASKS" \
        --db "$db" 2>&1 | tee "$log"
    echo "[$(date)] Finished $cfg"
    echo
done

echo "ALL 4 API DONE"
