#!/bin/bash
# Submit SWE-bench task jobs in batches
# Usage: ./submit_batch.sh /tmp/swebench_jobs [batch_size]

JOB_DIR="${1:-/tmp/swebench_jobs}"
BATCH_SIZE="${2:-10}"

echo "Submitting jobs from $JOB_DIR in batches of $BATCH_SIZE"

files=($(ls "$JOB_DIR"/task_*.yaml | sort))
total=${#files[@]}
echo "Total jobs: $total"

for ((i=0; i<total; i+=BATCH_SIZE)); do
    batch_end=$((i + BATCH_SIZE))
    if [ $batch_end -gt $total ]; then
        batch_end=$total
    fi
    echo "--- Submitting batch $((i/BATCH_SIZE + 1)): tasks $i to $((batch_end-1)) ---"
    for ((j=i; j<batch_end; j++)); do
        kubectl create -f "${files[$j]}" -n eidf230ns 2>&1 &
    done
    wait
    echo "Batch submitted. Waiting for some to complete before next batch..."

    # If not last batch, wait until running jobs drop below threshold
    if [ $batch_end -lt $total ]; then
        while true; do
            running=$(kubectl get jobs -n eidf230ns -l app=swebench-batch -o json | python3 -c "
import json, sys
data = json.load(sys.stdin)
running = sum(1 for j in data.get('items', []) if j.get('status', {}).get('active', 0) > 0)
print(running)
" 2>/dev/null)
            if [ "${running:-100}" -lt "$BATCH_SIZE" ]; then
                echo "Running jobs: $running, submitting next batch"
                break
            fi
            sleep 30
        done
    fi
done

echo "All $total jobs submitted!"
