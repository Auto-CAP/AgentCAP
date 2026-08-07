"""Regression tests for the Harmony transcript replay format.

gpt-oss conditions its next tool call on the shape of the replayed
transcript, so replaying a call in a form the model was never trained on
degrades the tool loop mid-run (observed on imo-answerbench: after a few
turns the model emits its code as plain analysis text with no recipient,
the loop ends, and no final answer is produced).

The two shapes under test:
- built-in `python` tool: raw code, analysis channel, no json constraint;
- `functions.*` tools: json arguments, commentary channel, <|constrain|>json.
"""

import json

import pytest

pytest.importorskip("openai_harmony")

from openai_harmony import Author, Message, Role, SystemContent  # noqa: E402

from agent_cap.agents.llm.harmony_client import (  # noqa: E402
    _messages_to_harmony,
    _stringify_harmony_content,
)


def _build(messages):
    return _messages_to_harmony(
        messages, SystemContent.new(), Message=Message, Role=Role, Author=Author
    )


def _python_history(arguments):
    return [
        {"role": "user", "content": "compute something"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Let me run some code.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "python", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "2"},
    ]


def test_builtin_python_call_replays_as_analysis_raw_code():
    msgs = _build(_python_history(json.dumps({"code": "print(1+1)"})))

    calls = [m for m in msgs if getattr(m, "recipient", None) == "python"]
    assert len(calls) == 1, "expected exactly one replayed python call"
    call = calls[0]
    assert call.channel == "analysis"
    assert _stringify_harmony_content(call) == "print(1+1)"
    assert not getattr(call, "content_type", None), (
        "built-in python calls must not carry a json constraint"
    )


def test_builtin_python_call_with_non_json_arguments_replays_verbatim():
    msgs = _build(_python_history("print(2+2)"))

    call = [m for m in msgs if getattr(m, "recipient", None) == "python"][0]
    assert call.channel == "analysis"
    assert _stringify_harmony_content(call) == "print(2+2)"


def test_python_tool_result_addressed_back_to_assistant():
    msgs = _build(_python_history(json.dumps({"code": "print(1+1)"})))

    results = [m for m in msgs if getattr(m, "author", None) and m.author.role == Role.TOOL]
    assert len(results) == 1
    result = results[0]
    assert result.author.name == "python"
    assert result.channel == "commentary"
    assert result.recipient == "assistant"


def test_function_tool_call_replays_as_constrained_commentary_json():
    args = json.dumps({"query": "hello"})
    msgs = _build(
        [
            {"role": "user", "content": "look it up"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "search", "arguments": args},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "found it"},
        ]
    )

    calls = [m for m in msgs if getattr(m, "recipient", None) == "functions.search"]
    assert len(calls) == 1
    call = calls[0]
    assert call.channel == "commentary"
    assert call.content_type == "<|constrain|>json"
    assert _stringify_harmony_content(call) == args

    result = [m for m in msgs if getattr(m, "author", None) and m.author.role == Role.TOOL][0]
    assert result.author.name == "functions.search"


def test_reasoning_content_replayed_as_analysis_before_call():
    msgs = _build(_python_history(json.dumps({"code": "print(1+1)"})))

    texts = [
        (getattr(m, "channel", None), _stringify_harmony_content(m)) for m in msgs
    ]
    analysis_idx = texts.index(("analysis", "Let me run some code."))
    call_idx = texts.index(("analysis", "print(1+1)"))
    assert analysis_idx < call_idx
