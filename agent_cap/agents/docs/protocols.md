# LLM protocols

Each protocol is one `LLMClient` class registered with
`@register_protocol(name, model_pattern=...)`. The framework picks one per
agent at run time.

## Built-ins

| Name | Auto-pattern | Endpoint API | When to use |
|---|---|---|---|
| `openai` (default) | always wins as fallback | `/v1/chat/completions` | OpenAI, OpenRouter, sglang, vLLM (chat mode), Ollama, LM Studio, llama.cpp server, any OpenAI-compatible service. |
| `harmony` | `(?i)gpt-?oss` | `/v1/completions` with token ids | gpt-oss-120b, gpt-oss-20b, GPT-OSS, gptoss. Encoded via `openai_harmony`. |
| `mock` | `(?i)^mock(-\|$)` | none | Offline tests, CI smoke. Activated by `--mock` or model name `mock` / `mock-*`. |

## Routing rules

1. If `endpoint.protocol` is set (CLI `--agent ...,protocol=NAME` or YAML
   `protocol: NAME`), use it.
2. Otherwise, the first registered protocol whose `model_pattern` matches
   `endpoint.name` wins (regex `.search`).
3. Otherwise the protocol marked `default=True` is used (built-in: `openai`).

Verify what a name routes to:

```python
from agent_cap.agents.types import ModelEndpoint
from agent_cap.agents.llm import resolve_protocol_name

resolve_protocol_name(ModelEndpoint(name="gpt-oss-120b"))            # 'harmony'
resolve_protocol_name(ModelEndpoint(name="Qwen/Qwen2.5-72B"))        # 'openai'
resolve_protocol_name(ModelEndpoint(name="my-model", protocol="harmony"))  # 'harmony'
```

## Adding a new protocol

Create a new file (anywhere importable), then load it with
`--load-module my.module`:

```python
# my_protocols.py
from typing import Any, Dict, List, Optional
import aiohttp

from agent_cap.agents.llm import register_protocol
from agent_cap.agents.llm.base import LLMReply
from agent_cap.agents.types import ModelEndpoint, Usage


@register_protocol("anthropic-native", model_pattern=r"(?i)^claude-")
class AnthropicNativeClient:
    def __init__(self, session: Optional[aiohttp.ClientSession] = None, **_: Any):
        self._session = session

    async def chat(
        self,
        endpoint: ModelEndpoint,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMReply:
        # POST to https://api.anthropic.com/v1/messages, translate response
        # to OpenAI-style {role, content, tool_calls} dict.
        ...
        return LLMReply(assistant=..., usage=Usage(...), latency_s=..., raw=...)
```

Then:

```bash
python -m agent_cap.agents \
  --load-module my_protocols \
  --agent agent=name=claude-3.5-sonnet,base_url=...,api_key=$ANTHROPIC_KEY \
  --task "..."
```

The framework does not need to be touched. The new protocol shows up in
`--list-protocols` and auto-routes any name matching `^claude-`.

## What an LLMClient must return

The single method is:

```python
async def chat(
    self,
    endpoint: ModelEndpoint,
    messages: List[Dict[str, Any]],   # OpenAI-style {role, content, tool_calls}
    tools: Optional[List[Dict[str, Any]]] = None,   # OpenAI tool schemas
) -> LLMReply
```

`LLMReply.assistant` must be an OpenAI-style dict:

```python
{
    "role": "assistant",
    "content": "text or empty",
    "tool_calls": [             # optional
        {
            "id": "call_...",
            "type": "function",
            "function": {"name": "calc", "arguments": "{...json...}"},
        }
    ],
}
```

The strategy and Agent code only ever sees OpenAI-style dicts. Whatever
encoding your protocol uses on the wire is hidden inside the client.

## Notes on harmony

- The harmony client uses `openai_harmony` to encode/decode at the token
  level. The wire format depends on the serving engine.
- Tool calls in the harmony return path are detected by inspecting
  `last_message.recipient` for `"python"`. Adapt the heuristic in
  `harmony_client.py` if you want richer tool routing.
- Streaming is not used. Streaming + harmony adds complexity that
  `run_imo_answerbench_4/5.py` handles directly; the framework client favors
  simplicity. Enable it later by switching the POST to a streaming call and
  accumulating token ids.

### Engine: vLLM vs sglang

Both vLLM and sglang can serve gpt-oss with harmony, but their native HTTP
shapes differ. Pick via `endpoint.engine`:

| Engine | Endpoint | Payload | Response |
|---|---|---|---|
| `vllm` (default) | `<base_url>/completions` | `{model, prompt: [token_ids], max_tokens, temperature, stop_token_ids}` | OpenAI-style `{choices: [{text, token_ids?}], usage}` |
| `sglang` | `<base_url>/generate` | `{input_ids: [...], rid, sampling_params: {max_new_tokens, temperature, top_p, stop_token_ids, skip_special_tokens: false, ...}, stream: false}` | `{text, output_ids, meta_info}` |

CLI:

```bash
# vLLM (default — no engine field needed)
--agent agent=name=gpt-oss-120b,base_url=http://localhost:8000/v1,api_key=EMPTY

# sglang — base_url WITHOUT /v1
--agent agent=name=gpt-oss-120b,base_url=http://localhost:30000,api_key=EMPTY,engine=sglang
```

YAML:

```yaml
agents:
  agent:
    name: gpt-oss-120b
    base_url: http://localhost:30000
    api_key: EMPTY
    engine: sglang
```

These mirror `run_imo_answerbench_4.py` (vLLM path) and
`run_imo_answerbench_5.py::sglang_generate_with_ids` (sglang path) respectively.
