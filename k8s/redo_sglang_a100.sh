#!/usr/bin/env bash
# Follow-up: after the master queue finishes, redo the corrupted sglang a100
# run (duplicate concurrent writers made results.jsonl + stream_stats
# unusable). Fresh output dir (date-stamped today) => no stale resume state.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentcap
cd "$(dirname "$0")/.."
LOG=~/agentcap-results/master_queue.log
exec >> "$LOG" 2>&1

# wait for the master queue to finish everything
while pgrep -f "master_queue.s[h]" > /dev/null 2>&1; do sleep 120; done
echo "=== redo_sglang_a100 start $(date -Is)"

# quarantine the corrupted run so it can't be mistaken for a result
CORRUPT=~/agentcap-results/sglang_a100x2_260707
if [ -d "$CORRUPT" ]; then
    mv "$CORRUPT" "${CORRUPT}_CORRUPTED_duplicate_writers"
    echo "quarantined ${CORRUPT}_CORRUPTED_duplicate_writers"
fi

bash k8s/run_one_experiment.sh sglang a100 4 \
    && echo "=== redo sglang a100 OK $(date -Is)" \
    || { echo "=== redo sglang a100 FAILED, retrying $(date -Is)"; sleep 60; \
         bash k8s/run_one_experiment.sh sglang a100 4 \
             && echo "=== redo OK on retry" || echo "=== redo FAILED twice"; }
echo "=== redo_sglang_a100 done $(date -Is)"
