"""Regression tests for the Harmony transcript replay/decode format.

gpt-oss conditions its next tool call on the shape of the replayed
transcript, so replaying a call in a form the model was never trained on
degrades the tool loop mid-run (observed on imo-answerbench: after a few
turns the model emits its code as plain analysis text with no recipient,
the loop ends, and no final answer is produced).

Two independent bugs, two independent fixes:
- replay (_build_replay_call_message): built-in `python` calls must replay
  as raw code on the analysis channel, not commentary+json like
  functions.* tools;
- decode (_decode_harmony_response): a tool call's code must not be swept
  into reasoning_content, or the poisoned history reproduces the same
  broken shape on replay.
"""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("openai_harmony")

from openai_harmony import Message, Role  # noqa: E402

from agent_cap.agents.llm.harmony_client import (  # noqa: E402
    _build_replay_call_message,
    _decode_harmony_response,
    _stringify_harmony_content,
)


def _replay(tool_name, args_str):
    return _build_replay_call_message(tool_name, args_str, Message=Message, Role=Role)


def test_builtin_python_call_replays_as_analysis_raw_code():
    call = _replay("python", json.dumps({"code": "print(1+1)"}))

    assert call.channel == "analysis"
    assert call.recipient == "python"
    assert _stringify_harmony_content(call) == "print(1+1)"
    assert not getattr(call, "content_type", None), (
        "built-in python calls must not carry a json constraint"
    )


def test_builtin_python_call_with_non_json_arguments_replays_verbatim():
    call = _replay("python", "print(2+2)")

    assert call.channel == "analysis"
    assert call.recipient == "python"
    assert _stringify_harmony_content(call) == "print(2+2)"


def test_function_tool_call_replays_as_constrained_commentary_json():
    args = json.dumps({"query": "hello"})
    call = _replay("search", args)

    assert call.channel == "commentary"
    assert call.recipient == "functions.search"
    assert call.content_type == "<|constrain|>json"
    assert _stringify_harmony_content(call) == args


class _FakeEncoding:
    def __init__(self, parsed):
        self._parsed = parsed

    def parse_messages_from_completion_tokens(self, token_ids, role):
        return self._parsed


def _decode(parsed):
    return _decode_harmony_response(
        encoding=_FakeEncoding(parsed),
        token_ids=[1, 2, 3],
        fallback_text="FALLBACK",
        Role=Role,
    )


def test_decode_python_call_excludes_code_from_reasoning():
    reply = _decode(
        [
            SimpleNamespace(channel="analysis", recipient=None, content="thinking hard"),
            SimpleNamespace(channel="analysis", recipient="python", content="print(1+1)"),
        ]
    )

    assert reply["tool_calls"][0]["function"]["name"] == "python"
    assert json.loads(reply["tool_calls"][0]["function"]["arguments"]) == {
        "code": "print(1+1)"
    }
    assert reply["reasoning_content"] == "thinking hard"
    assert "print(1+1)" not in reply["reasoning_content"]


def test_decode_final_answer_with_reasoning():
    reply = _decode(
        [
            SimpleNamespace(channel="analysis", recipient=None, content="thinking hard"),
            SimpleNamespace(channel="final", recipient=None, content="the answer is 3"),
        ]
    )

    assert "tool_calls" not in reply
    assert reply["content"] == "the answer is 3"
    assert reply["reasoning_content"] == "thinking hard"
