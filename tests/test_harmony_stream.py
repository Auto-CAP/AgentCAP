import asyncio
import json

from agent_cap.agents.llm.harmony_client import HarmonyClient


class _Content:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _Response:
    status = 200

    def __init__(self, chunks):
        self.content = _Content(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _Session:
    def __init__(self, chunks):
        self._chunks = chunks

    def post(self, *_args, **_kwargs):
        return _Response(self._chunks)


def _event(payload, newline="\n"):
    return f"data: {json.dumps(payload, ensure_ascii=False)}{newline}{newline}".encode()


def test_sglang_sse_accepts_cumulative_event_larger_than_64_kib():
    final = {"text": "x" * 70_000, "output_ids": [1, 2, 3]}
    wire = _event({"text": "x", "output_ids": [1]}) + _event(final) + b"data: [DONE]\n\n"
    client = HarmonyClient(session=_Session([wire]))

    latest, ttft_s = asyncio.run(client._post_stream("url", {}, "", "test"))

    assert latest == final
    assert ttft_s is not None


def test_sglang_sse_reassembles_split_events_and_crlf_delimiters():
    first = {"text": "α", "output_ids": [1]}
    final = {"text": "αβ", "output_ids": [1, 2]}
    wire = _event(first, "\r\n") + _event(final) + b"data: [DONE]\n\n"
    cut_points = [1, 7, 19, 31, 47, len(wire) - 3]
    chunks = []
    start = 0
    for end in cut_points + [len(wire)]:
        chunks.append(wire[start:end])
        start = end
    client = HarmonyClient(session=_Session(chunks))

    latest, ttft_s = asyncio.run(client._post_stream("url", {}, "", "test"))

    assert latest == final
    assert ttft_s is not None
