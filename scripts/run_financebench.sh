#!/bin/bash
# Run FinanceBench for all 4 Qwen combinations sequentially.
# GPU 2: Qwen3.5-9B  (port 30002)
# GPU 3: Qwen3.5-27B (port 30003)

set -e

PYTHON=/home/liang/miniconda3/envs/AgentBench/bin/python
BASE_ARGS="--strategy plan-execute --dataset financebench --backend financebench --num-tasks 150 --max-turns 30"
URL_9B="http://localhost:30002/v1"
URL_27B="http://localhost:30003/v1"
MODEL_9B="Qwen/Qwen3.5-9B"
MODEL_27B="Qwen/Qwen3.5-27B"

run_combo() {
    local name=$1
    local planner_model=$2
    local planner_url=$3
    local executor_model=$4
    local executor_url=$5

    echo "========================================"
    echo "Starting: $name"
    echo "Planner:  $planner_model"
    echo "Executor: $executor_model"
    echo "Time:     $(date)"
    echo "========================================"

    $PYTHON -m agent_cap.runner.team_runner $BASE_ARGS \
        --output-dir results/financebench_${name} \
        --planner-model  "$planner_model"  --planner-base-url  "$planner_url"  --planner-api-key  dummy \
        --executor-model "$executor_model" --executor-base-url "$executor_url" --executor-api-key dummy \
        2>&1 | tee /tmp/financebench_${name}.log

    echo "Finished: $name at $(date)"
    echo ""
}

cd /home/liang/AgentCAP

run_combo "9b_9b"   $MODEL_9B  $URL_9B  $MODEL_9B  $URL_9B
run_combo "27b_27b" $MODEL_27B $URL_27B $MODEL_27B $URL_27B
run_combo "27b_9b"  $MODEL_27B $URL_27B $MODEL_9B  $URL_9B
run_combo "9b_27b"  $MODEL_9B  $URL_9B  $MODEL_27B $URL_27B

echo "All 4 combinations done at $(date)"
