#!/usr/bin/env bash
# Full redo of sglang h100x2 (previous run hit the stale-port-forward bug:
# 13 tasks died with 0 LLM requests). Waits for redo_sglang_a100 to finish
# first so only one experiment runs at a time.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentcap
cd "$(dirname "$0")/.."
LOG=~/agentcap-results/master_queue.log
exec >> "$LOG" 2>&1

while pgrep -f "master_queue.s[h]" > /dev/null 2>&1 || pgrep -f "redo_sglang_a100.s[h]" > /dev/null 2>&1; do
    sleep 120
done
echo "=== redo_sglang_h100 start $(date -Is)"

OLD=~/agentcap-results/sglang_h100x2_260707
if [ -d "$OLD" ]; then
    mv "$OLD" "${OLD}_INVALID_stale_pf_bug"
    echo "quarantined ${OLD}_INVALID_stale_pf_bug"
fi

bash k8s/run_one_experiment.sh sglang h100 4 \
    && echo "=== redo sglang h100 OK $(date -Is)" \
    || { echo "=== redo sglang h100 FAILED, retrying $(date -Is)"; sleep 60; \
         bash k8s/run_one_experiment.sh sglang h100 4 \
             && echo "=== redo OK on retry" || echo "=== redo FAILED twice"; }
echo "=== redo_sglang_h100 done — ALL EXPERIMENTS COMPLETE $(date -Is)"
