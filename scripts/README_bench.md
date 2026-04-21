# AgentCAP Serving Benchmark

Measures throughput, latency, and $/token for serving GPT-OSS-120B and Qwen3.5-27B under AgentCAP's real agent workload patterns.

## What this measures

Input/output rate pricing ($/M tokens) for each (model, TP size, workload pattern, concurrency), derived from measured throughput:
- `$/M_in  = (GPU_HR × TP) × 1e6 / prefill_throughput / 3600`
- `$/M_out = (GPU_HR × TP) × 1e6 / decode_throughput  / 3600`

## Configuration

| Axis | Values |
|---|---|
| Model | gpt-oss-120b (MXFP4), Qwen3.5-27B (FP16) |
| Tensor parallelism | 2, 4, 8 (single-GPU omitted: KV cache at 65K max-model-len requires >= 2 GPUs) |
| Workload pattern | long-exec (50K/170), balanced-plan (6K/300), short-burst (800/700) |
| Concurrency | 1, 10, 100, 1000 |

Workload patterns come from AgentCAP trace analysis (per-request token distributions):
- long-exec: MCP-Atlas executor calls
- balanced-plan: MedAgentBench planner calls
- short-burst: FinanceBench planner calls

Per script: 2 models x 3 cases x 4 concurrency = 24 runs, ~1.5 hours.

## Requirements

- 8x H100 80GB (for TP=8 script; smaller counts for lower TP)
- vLLM >= 0.19.0
- Model weights available at HF Hub or local cache (set `HF_HOME`)

```bash
pip install vllm==0.19.0
```

## Run

```bash
# Run each TP config on a separate node (or sequentially on one node)
bash bench_tp2.sh
bash bench_tp4.sh
bash bench_tp8.sh
```

Results written to `./bench_results_tp{2,4,8}/{model}_tp{N}_{case}_c{C}.json`.
Server logs: `./bench_results_tp{N}/server_*.log`.

## Output format

Each JSON (from `vllm bench serve --save-result`) contains:
- `output_throughput` — decode tok/s
- `total_token_throughput` — aggregate tok/s
- `request_throughput` — req/s
- `mean_ttft_ms`, `mean_tpot_ms` — latency
- `median_itl_ms`, `p99_itl_ms`
- `total_input_tokens`, `total_generated_tokens`, `duration`

## Notes

- `--enable-prefix-caching` mirrors production serving (shared prefixes across concurrent users)
- `--ignore-eos` fixes output length to `--random-output-len`
- Tool parser auto-selected: `qwen3_coder` for Qwen3.5, `openai` for GPT-OSS
- Existing result files are skipped; delete the `bench_results_tp*/` directory to rerun
- Concurrency 1000 may OOM on long-exec cases with low TP; vLLM will queue requests

## Troubleshooting

- **Server fails to start**: check `server_*.log`; usually OOM -> increase TP
- **Tool parser 400 error**: ensure correct `--tool-call-parser` for the model
- **vLLM version issues**: this script targets vLLM 0.19.0 API
