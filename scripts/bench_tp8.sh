#!/bin/bash
set -e

MODELS=(
    "openai/gpt-oss-120b:gpt-oss-120b"
    "Qwen/Qwen3.5-27B:qwen35-27b"
)

TP=8
CASES=(
    "long-exec:50000:170"
    "balanced-plan:6000:300"
    "short-burst:800:700"
)
CONCURRENCY_LEVELS=(1 10 100 1000)

RESULTS_DIR="./bench_results_tp${TP}"
mkdir -p "$RESULTS_DIR"

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r model_path served_name <<< "$model_spec"

    echo "========================================================"
    echo "Starting vLLM: $served_name, TP=$TP"
    echo "========================================================"

    TOOL_PARSER="qwen3_coder"
    [[ "$served_name" == *"gpt-oss"* ]] && TOOL_PARSER="openai"

    vllm serve "$model_path" \
        --served-model-name "$served_name" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.9 \
        --max-model-len 65536 \
        --port 8000 \
        --enable-prefix-caching \
        --enable-auto-tool-choice \
        --tool-call-parser "$TOOL_PARSER" \
        > "$RESULTS_DIR/server_${served_name}_tp${TP}.log" 2>&1 &
    SERVER_PID=$!

    for i in {1..120}; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "Server ready after ${i}0s"
            break
        fi
        sleep 10
    done

    for case_spec in "${CASES[@]}"; do
        IFS=':' read -r case_name in_len out_len <<< "$case_spec"

        for conc in "${CONCURRENCY_LEVELS[@]}"; do
            result_file="$RESULTS_DIR/${served_name}_tp${TP}_${case_name}_c${conc}.json"

            if [ -f "$result_file" ]; then
                echo "SKIP: $result_file"
                continue
            fi

            echo "--- Running: $served_name TP=$TP case=$case_name conc=$conc ---"

            vllm bench serve \
                --host 0.0.0.0 --port 8000 \
                --model "$served_name" \
                --dataset-name random \
                --random-input-len "$in_len" \
                --random-output-len "$out_len" \
                --ignore-eos \
                --num-prompts 200 \
                --max-concurrency "$conc" \
                --save-result \
                --result-filename "$result_file" \
                --trust-remote-code
        done
    done

    kill $SERVER_PID 2>/dev/null
    sleep 15
done

echo "TP=$TP ALL DONE. Results in $RESULTS_DIR/"
