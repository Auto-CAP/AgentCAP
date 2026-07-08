#!/usr/bin/env bash
# Master queue: finish the in-flight vllm h200 run, then run the remaining
# experiments sequentially. Fully detached-safe; every step is resume-safe,
# so if this script dies it can simply be relaunched.
#   setsid nohup bash k8s/master_queue.sh > /dev/null 2>&1 &
# Progress: ~/agentcap-results/master_queue.log
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentcap
cd "$(dirname "$0")/.."

LOG=~/agentcap-results/master_queue.log
exec >> "$LOG" 2>&1
echo "=== master_queue start $(date -Is)"

VOUT=~/agentcap-results/vllm_h200x1_260707

# ---- Phase 0: let the in-flight vllm h200 run finish ----
while pgrep -f "run_swebench_k8s_10[0]" > /dev/null 2>&1; do sleep 120; done
if grep -q "FINAL" "$VOUT/run.log" 2>/dev/null; then
    echo "vllm h200 completed: $(grep FINAL "$VOUT/run.log" | tail -1)"
else
    echo "vllm h200 incomplete — resuming via driver"
    bash k8s/run_one_experiment.sh vllm h200 4   # resumes, packages, stops server
fi

# package + teardown for the completed-in-place case
if [ ! -d "$VOUT/teas" ]; then
    cat > "$VOUT/run.sh" <<'PROV'
#!/usr/bin/env bash
# EIDF 1x NVIDIA H200 -- official vllm/vllm-openai:v0.21.0 serving unsloth/gpt-oss-120b TP=1
# AgentCAP a1e54c7; SWE-bench Lite curated-100 via SWE-agent + k8s swe-rex sidecars
# eval: per-instance k8s pods running official TestSpec scripts, swebench.harness.grading
bash k8s/launch_llm_server.sh vllm h200
bash k8s/port_forward_llm.sh vllm-gptoss-h200 8000 &
bash scripts/run_swebench_k8s_100.sh --output-dir ~/agentcap-results/vllm_h200x1_260707 --concurrency 4
python scripts/package_teas_results.py --run-dir ~/agentcap-results/vllm_h200x1_260707 \
    --engine vllm --engine-version 0.21.0 --gpu-type "NVIDIA H200" --num-gpus 1 --tp 1 --concurrency 4
PROV
    python scripts/package_teas_results.py --run-dir "$VOUT" \
        --engine vllm --engine-version 0.21.0 --gpu-type "NVIDIA H200" \
        --num-gpus 1 --tp 1 --concurrency 4
fi
bash k8s/launch_llm_server.sh vllm h200 --stop || true

# ---- Phases 1-4: remaining experiments, one at a time ----
for exp in "sglang h100" "vllm h100" "sglang a100" "vllm a100"; do
    set -- $exp
    echo "=== starting $1 $2 $(date -Is)"
    if bash k8s/run_one_experiment.sh "$1" "$2" 4; then
        echo "=== $1 $2 OK $(date -Is)"
    else
        echo "=== $1 $2 FAILED, retrying once $(date -Is)"
        sleep 60
        bash k8s/run_one_experiment.sh "$1" "$2" 4 \
            && echo "=== $1 $2 OK on retry" \
            || echo "=== $1 $2 FAILED twice — skipping"
    fi
done

echo "=== master_queue ALL DONE $(date -Is)"
