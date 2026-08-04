# Self-hosted serving guide

The `agent_cap.agents` framework only handles the wire-level **protocol**
(OpenAI chat completions / Harmony token completions / Mock). It does NOT
configure how the server parses the model's raw output into tool calls.

That parsing is server-side and depends on which inference engine you run
and which model family you serve. This page lists the launch commands.

## Two layers, who is responsible

```
                +--------------------------+
                |  agent_cap.agents (CLI)  |
                |                          |
                |  protocol:               |
                |   openai   /v1/chat/completions     standard tools=[]
                |   harmony  /v1/completions or       gpt-oss token-level
                |            /generate
                |   mock     in-process               smoke tests
                +-------------+------------+
                              |
                              v
                +-------------+------------+
                |  Inference server        |
                |  (vLLM, sglang, Ollama)  |
                |                          |
                |  tool-call-parser:       |
                |   qwen25 / hermes /      |
                |   llama3_json / etc.     |
                |                          |
                |  (chat template,         |
                |   token-level decoding)  |
                +-------------+------------+
                              |
                              v
                +-------------+------------+
                |  Model weights           |
                +--------------------------+
```

The framework sends OpenAI-shape requests (`{messages, tools}`); the server
decides how to embed `tools` into the prompt and how to extract `tool_calls`
from raw token output. If the server is missing or has the wrong tool parser
for the model, `tool_calls` will be empty or wrong even when the model
itself would have produced them.

## Launch commands by model family

### Qwen 2.5 Instruct (chat completions, OpenAI protocol)

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --port 30000 \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

```bash
# sglang
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --port 30000 \
  --tp 4 \
  --tool-call-parser qwen25
```

Then in agents:

```bash
python -m agent_cap.agents --strategy single \
  --model Qwen/Qwen2.5-72B-Instruct \
  --base-url http://localhost:30000/v1 \
  --api-key EMPTY \
  --task "..."
```

No `--protocol` / `--engine` needed; openai is the default.

### Llama 3.1 / 3.3 Instruct

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --port 30000 --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
```

```bash
# sglang
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --port 30000 --tp 4 \
  --tool-call-parser llama3
```

### Mistral / Mixtral

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Large-Instruct \
  --port 30000 --enable-auto-tool-choice \
  --tool-call-parser mistral
```

### DeepSeek-V3 / V3.2

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V3 \
  --port 30000 --enable-auto-tool-choice \
  --tool-call-parser deepseek_v3
```

### gpt-oss (Harmony protocol, no `tool-call-parser` flag needed)

gpt-oss models do NOT use a server-side tool parser the way Qwen/Llama do.
The Harmony chat template encodes/decodes tool calls at the token level,
and the framework handles that decoding inside `HarmonyClient`. You only
need to start the server with the right precision and context length.

```bash
# vLLM (engine=vllm)
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --kv-cache-dtype fp8_e4m3
```

```bash
# sglang (engine=sglang)
python -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --port 30000 \
  --tp 4 \
  --context-length 131072 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 8 \
  --mem-fraction-static 0.88
```

Then in agents:

```bash
# vLLM path
python -m agent_cap.agents --strategy single \
  --model gpt-oss-120b \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --engine vllm \
  --max-tokens 131072 \
  --task "..."

# sglang path (note: NO /v1 in base_url)
python -m agent_cap.agents --strategy single \
  --model gpt-oss-120b \
  --base-url http://localhost:30000 \
  --api-key EMPTY \
  --engine sglang \
  --max-tokens 131072 \
  --task "..."
```

This runs the default `openai` protocol: gpt-oss does not auto-route to
harmony, so add `--protocol harmony` when the raw token-level decode path is
what you want (see `docs/agents/protocols.md`). `--engine` picks the wire
format inside harmony: the sglang path streams and reports a measured TTFT,
while the vLLM path returns whole responses and reports no TTFT.

## Tool-parser quick reference

| Family | vLLM `--tool-call-parser` | sglang `--tool-call-parser` |
|---|---|---|
| Qwen 2.5 / 3 Instruct | `hermes` | `qwen25` |
| Llama 3.1 / 3.3 Instruct | `llama3_json` | `llama3` |
| Mistral / Mixtral Instruct | `mistral` | `mistral` |
| DeepSeek-V3 / V3.2 | `deepseek_v3` | `deepseek_v3` |
| Hermes / OpenHermes | `hermes` | `hermes` |
| gpt-oss | n/a (Harmony) | n/a (Harmony) |

Check your server's docs for the exact list; names drift between versions.

## Common pitfalls

- **`tool_calls` is empty when it should not be.** Server is missing the
  right `--tool-call-parser`, or the model itself was not fine-tuned for
  tool use (use an `-Instruct` / `-chat` variant, not a base model).
- **400 from sglang `/generate` when running gpt-oss.** `base_url` must NOT
  end in `/v1`. Use `http://host:port`, not `http://host:port/v1`.
- **Harmony returns garbled text.** Server is decoding tokens with the wrong
  chat template. Make sure the model id passed to vLLM/sglang is one that
  the engine recognizes as gpt-oss (it ships with the Harmony template).
- **First call hangs.** Cold-start compile (`--enable-torch-compile` etc.).
  Wait for the server's "ready" log line, then retry.

## "How do I check it works?"

```bash
curl http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "messages": [{"role":"user","content":"What is 1+1? Use the calc tool."}],
  "tools": [{
    "type":"function",
    "function":{
      "name":"calc",
      "description":"compute arithmetic",
      "parameters":{"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}
    }
  }]
}'
```

A correctly configured server returns `choices[0].message.tool_calls` with
the calc call. If you only see `content: "I would call calc..."`, the
server lacks the right `--tool-call-parser`.

For gpt-oss, run the framework's `--list-protocols` smoke instead; harmony
does its own token decoding so the curl test above is not applicable.
