"""OpenAI Harmony protocol client for gpt-oss family models.

Models like `gpt-oss-120b` / `gpt-oss-20b` use the Harmony chat template,
which encodes/decodes conversations at the token level via `openai_harmony`.

Two serving engines are supported via `endpoint.engine`:

  engine="vllm" or "" (default)
    POST `<base_url>/completions` with payload {model, prompt: [token_ids],
    max_tokens, temperature, stop_token_ids}. Mirrors run_imo_answerbench_4.py.

  engine="sglang"
    POST `<base_url>/generate` with payload {input_ids: [token_ids], rid,
    sampling_params: {max_new_tokens, temperature, top_p, stop_token_ids,
    skip_special_tokens: false, ...}, stream: false}. Mirrors
    run_imo_answerbench_5.py::sglang_generate_with_ids. base_url should NOT
    include /v1 (sglang's native endpoint is at the root).

This client converts OpenAI-style messages+tools to Harmony Messages, renders
them to token ids, calls the chosen engine, then decodes the response back
into OpenAI-style `tool_calls` so the rest of the framework stays
protocol-agnostic.

Auto-routes on model names matching `(?i)gpt-?oss`. Override with
`--agent ...,protocol=harmony,engine=sglang` etc.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import aiohttp

from agent_cap.agents.llm.base import LLMReply
from agent_cap.agents.llm.registry import register_protocol
from agent_cap.agents.types import ModelEndpoint, Usage


@register_protocol("harmony", model_pattern=r"(?i)gpt-?oss")
class HarmonyClient:
    def __init__(self, session: Optional[aiohttp.ClientSession] = None, **_: Any) -> None:
        self._session = session
        self._encoding = None
        self._stop_token_ids: List[int] = []

    def _ensure_encoding(self) -> Any:
        if self._encoding is not None:
            return self._encoding
        try:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding
        except ImportError as exc:
            raise RuntimeError(
                "harmony protocol requires `openai_harmony`. "
                "Install it or use `--agent ...,protocol=openai`."
            ) from exc
        self._encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        try:
            self._stop_token_ids = list(
                self._encoding.stop_tokens_for_assistant_actions()
            )
        except Exception:
            self._stop_token_ids = [200002, 200012]
        return self._encoding

    async def chat(
        self,
        endpoint: ModelEndpoint,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMReply:
        if self._session is None:
            raise RuntimeError(
                "HarmonyClient requires an aiohttp.ClientSession. "
                "Construct it inside `async with aiohttp.ClientSession() as session:`."
            )

        from openai_harmony import (
            Author,
            Conversation,
            Message,
            ReasoningEffort,
            Role,
            SystemContent,
            ToolNamespaceConfig,
        )

        encoding = self._ensure_encoding()

        sys_content = SystemContent.new().with_reasoning_effort(
            reasoning_effort=ReasoningEffort.HIGH
        )
        if tools:
            tool_cfg = _harmony_tool_config(tools, ToolNamespaceConfig)
            if tool_cfg is not None:
                sys_content = sys_content.with_tools(tool_cfg)

        harmony_messages: List[Any] = []
        system_used = False
        for m in messages:
            role = str(m.get("role", "")).lower()
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content) if content is not None else ""

            if role == "system":
                merged = sys_content.with_model_identity(content) if content else sys_content
                harmony_messages.append(Message.from_role_and_content(Role.SYSTEM, merged))
                system_used = True
            elif role == "user":
                harmony_messages.append(Message.from_role_and_content(Role.USER, content))
            elif role == "assistant":
                tool_calls = m.get("tool_calls") or []
                if tool_calls:
                    if content:
                        harmony_messages.append(
                            Message.from_role_and_content(Role.ASSISTANT, content)
                        )
                    for call in tool_calls:
                        fn = (call or {}).get("function") or {}
                        tool_name = str(fn.get("name") or "python")
                        args_str = fn.get("arguments", "")
                        if not isinstance(args_str, str):
                            args_str = json.dumps(args_str)
                        call_msg = Message.from_role_and_content(Role.ASSISTANT, args_str)
                        call_msg.recipient = tool_name
                        harmony_messages.append(call_msg)
                else:
                    harmony_messages.append(
                        Message.from_role_and_content(Role.ASSISTANT, content)
                    )
            elif role == "tool":
                tool_name = str(m.get("name") or m.get("tool_call_id") or "python")
                msg = Message.from_role_and_content(Role.TOOL, content)
                msg.author = Author(role=Role.TOOL, name=tool_name)
                harmony_messages.append(msg)

        if not system_used:
            harmony_messages.insert(
                0, Message.from_role_and_content(Role.SYSTEM, sys_content)
            )

        conversation = Conversation.from_messages(harmony_messages)
        prompt_ids = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

        engine = (endpoint.engine or "").strip().lower() or "vllm"
        t0 = time.perf_counter()

        if engine == "sglang":
            token_ids, completion_text, raw, usage_in, usage_out = await self._call_sglang(
                endpoint=endpoint, prompt_ids=prompt_ids,
            )
        else:
            token_ids, completion_text, raw, usage_in, usage_out = await self._call_vllm(
                endpoint=endpoint, prompt_ids=prompt_ids,
            )

        elapsed = time.perf_counter() - t0

        assistant = _decode_harmony_response(
            encoding=encoding,
            token_ids=token_ids,
            fallback_text=completion_text,
            Role=Role,
        )

        usage_raw = raw.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_raw.get("prompt_tokens", usage_in) or usage_in),
            output_tokens=int(usage_raw.get("completion_tokens", usage_out) or usage_out),
            cached_tokens=int(
                (usage_raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            ),
            requests=1,
        )
        return LLMReply(assistant=assistant, usage=usage, latency_s=elapsed, raw=raw)

    async def _call_vllm(
        self, *, endpoint: ModelEndpoint, prompt_ids: List[int],
    ):
        url = endpoint.base_url.rstrip("/") + "/completions"
        payload: Dict[str, Any] = {
            "model": endpoint.name,
            "prompt": prompt_ids,
            "max_tokens": max(1, int(endpoint.max_tokens) - len(prompt_ids)),
            "temperature": float(endpoint.temperature),
            "top_p": float(endpoint.top_p),
            "stop_token_ids": list(self._stop_token_ids),
        }
        if endpoint.seed is not None:
            payload["seed"] = int(endpoint.seed)
        raw = await self._post_json(url, payload, endpoint.api_key, "vLLM /completions")
        choices = raw.get("choices") or []
        raw_token_ids = choices[0].get("token_ids") if choices else None
        token_ids: List[int] = []
        if isinstance(raw_token_ids, list):
            try:
                token_ids = [int(t) for t in raw_token_ids]
            except (TypeError, ValueError):
                token_ids = []
        completion_text = (choices[0].get("text", "") if choices else "") or ""
        return token_ids, completion_text, raw, len(prompt_ids), len(token_ids)

    async def _call_sglang(
        self, *, endpoint: ModelEndpoint, prompt_ids: List[int],
    ):
        import uuid as _uuid

        url = endpoint.base_url.rstrip("/") + "/generate"
        max_new = max(1, int(endpoint.max_tokens) - len(prompt_ids))
        sampling_params: Dict[str, Any] = {
            "max_new_tokens": max_new,
            "temperature": float(endpoint.temperature),
            "top_p": float(endpoint.top_p),
            "stop_token_ids": list(self._stop_token_ids),
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "no_stop_trim": True,
        }
        if endpoint.seed is not None:
            sampling_params["seed"] = int(endpoint.seed)
        payload: Dict[str, Any] = {
            "input_ids": [int(t) for t in prompt_ids],
            "rid": str(_uuid.uuid4()),
            "sampling_params": sampling_params,
            "stream": False,
        }
        raw = await self._post_json(url, payload, endpoint.api_key, "sglang /generate")

        text = raw.get("text", "") or ""
        output_ids = raw.get("output_ids") or []
        output_ids = [int(t) for t in output_ids]

        if len(output_ids) > len(prompt_ids) and output_ids[: len(prompt_ids)] == list(prompt_ids):
            output_ids = output_ids[len(prompt_ids):]

        meta = raw.get("meta_info") or {}
        usage_in = int(meta.get("prompt_tokens", len(prompt_ids)) or len(prompt_ids))
        usage_out = int(meta.get("completion_tokens", len(output_ids)) or len(output_ids))
        return output_ids, text, raw, usage_in, usage_out

    async def _post_json(
        self, url: str, payload: Dict[str, Any], api_key: str, label: str,
    ) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if self._session is None:
            raise RuntimeError("HarmonyClient: no aiohttp session")
        async with self._session.post(url, json=payload, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"{label} failed ({resp.status}): {text[:500]}")
            return json.loads(text)


def _harmony_tool_config(tools: List[Dict[str, Any]], ToolNamespaceConfig: Any) -> Any:
    if not tools:
        return None
    first = tools[0].get("function") or {}
    name = first.get("name") or "python"
    desc = first.get("description") or (
        "Use this tool to execute the necessary action. "
        "Always finalize with an ANSWER line."
    )
    return ToolNamespaceConfig(name=name, description=desc, tools=[])


def _decode_harmony_response(
    *,
    encoding: Any,
    token_ids: List[int],
    fallback_text: str,
    Role: Any,
) -> Dict[str, Any]:
    if not token_ids:
        return {"role": "assistant", "content": fallback_text}
    try:
        parsed = encoding.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)
    except Exception:
        return {"role": "assistant", "content": fallback_text}

    if not parsed:
        return {"role": "assistant", "content": fallback_text}

    last = parsed[-1]
    recipient = getattr(last, "recipient", None)
    content_text = _stringify_harmony_content(last)

    if recipient is not None and "python" in str(recipient).lower():
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": content_text}),
                    },
                }
            ],
        }
    return {"role": "assistant", "content": content_text}


def _stringify_harmony_content(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for blk in content:
            text = getattr(blk, "text", None)
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(blk, dict):
                t = blk.get("text") or blk.get("content")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return str(content) if content is not None else ""


__all__ = ["HarmonyClient"]
