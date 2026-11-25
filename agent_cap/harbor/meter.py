"""Host-side metering proxy: per-request TTFT, decode time and tokens for any harness.

Codex, Claude Code, OpenHands and the other Harbor agents call the model from
inside the task container and report, at best, total tokens. To compare
harnesses on the same terms as CAPAgent, the metered agents
(``agent_cap.harbor.metered``) point the agent's ``--base-url`` at this proxy.
It forwards every request unchanged to the real endpoint, streams the response
back, and records one row per request:

* ``ttft_s``: request received -> first content/reasoning/tool-call token;
* ``decode_s``: first -> last token; ``latency_s``: request -> end of response;
* tokens from the response's own usage block (input includes cached tokens).

Streams understood: OpenAI chat completions, OpenAI Responses and Anthropic
Messages (SSE or plain JSON), and the Responses API over WebSocket (Codex's
default transport; each ``response.create`` message is one request). Other
paths are forwarded and recorded without token fields. The proxy runs in its own thread and event loop so that Harbor's
loop does not add latency to the measurements.

URL layout: ``http://<host>:<port>/<token>/<path>`` -> ``<upstream>/<path>``,
one token per trial, so concurrent trials are kept apart.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import aiohttp
from aiohttp import web

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailers", "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding",
}
WS_HANDSHAKE = {"sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions",
                "sec-websocket-protocol", "sec-websocket-accept"}
WS_END_EVENTS = ("response.completed", "response.failed", "response.incomplete", "error")
LLM_PATH_SUFFIXES = ("chat/completions", "completions", "responses", "messages")


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


class ResponseMeter:
    """Feeds on response bytes; finds token times, usage and tool calls."""

    def __init__(self, streaming: bool) -> None:
        self.streaming = streaming
        self.first_token: Optional[float] = None
        self.last_token: Optional[float] = None
        self.usage: Dict[str, int] = {}
        self.tool_calls = 0
        self._tool_keys: set = set()
        self._buf = b""
        self._body: List[bytes] = []

    # -- feeding ------------------------------------------------------------ #

    def feed(self, chunk: bytes, now: float) -> None:
        if not self.streaming:
            self._body.append(chunk)
            return
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if isinstance(event, dict):
                self._on_event(event, now)

    def feed_event(self, event: Mapping[str, Any], now: float) -> None:
        """One already-decoded event (a WebSocket message)."""
        self._on_event(event, now)

    def finish(self, now: float) -> None:
        if self.streaming:
            if self._buf.strip():
                self.feed(b"\n", now)
            return
        try:
            body = json.loads(b"".join(self._body) or b"null")
        except ValueError:
            return
        if isinstance(body, dict):
            self._on_body(body)

    # -- streamed events ---------------------------------------------------- #

    def _token(self, now: float) -> None:
        if self.first_token is None:
            self.first_token = now
        self.last_token = now

    def _on_event(self, ev: Mapping[str, Any], now: float) -> None:
        kind = ev.get("type")
        if isinstance(kind, str):
            if kind.startswith("response."):
                self._responses_event(kind, ev, now)
                return
            if kind in ("message_start", "message_delta", "content_block_start", "content_block_delta"):
                self._anthropic_event(kind, ev, now)
                return
        for choice in ev.get("choices") or []:  # chat completions chunk
            delta = (choice or {}).get("delta") or {}
            if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls", "refusal")):
                self._token(now)
            for call in delta.get("tool_calls") or []:
                key = (choice.get("index", 0), call.get("index", call.get("id")))
                if key not in self._tool_keys:
                    self._tool_keys.add(key)
                    self.tool_calls += 1
        if isinstance(ev.get("usage"), dict):
            self._chat_usage(ev["usage"])

    def _responses_event(self, kind: str, ev: Mapping[str, Any], now: float) -> None:
        if kind.endswith(".delta"):
            self._token(now)
        elif kind == "response.output_item.added":
            item = ev.get("item") or {}
            if str(item.get("type", "")).endswith("call"):
                self.tool_calls += 1
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            response = ev.get("response") or {}
            if isinstance(response.get("usage"), dict):
                self._responses_usage(response["usage"])

    def _anthropic_event(self, kind: str, ev: Mapping[str, Any], now: float) -> None:
        if kind == "content_block_delta":
            self._token(now)
        elif kind == "content_block_start":
            if (ev.get("content_block") or {}).get("type") in ("tool_use", "server_tool_use"):
                self.tool_calls += 1
        elif kind == "message_start":
            self._anthropic_usage((ev.get("message") or {}).get("usage") or {})
        elif kind == "message_delta" and isinstance(ev.get("usage"), dict):
            self._anthropic_usage(ev["usage"])

    # -- whole bodies --------------------------------------------------------- #

    def _on_body(self, body: Mapping[str, Any]) -> None:
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        if body.get("object") == "response" or "output" in body:
            self._responses_usage(usage)
            self.tool_calls = sum(1 for item in body.get("output") or [] if str(item.get("type", "")).endswith("call"))
        elif body.get("type") == "message":
            self._anthropic_usage(usage)
            self.tool_calls = sum(1 for b in body.get("content") or [] if b.get("type") == "tool_use")
        else:
            self._chat_usage(usage)
            for choice in body.get("choices") or []:
                self.tool_calls += len(((choice or {}).get("message") or {}).get("tool_calls") or [])

    # -- usage blocks ------------------------------------------------------------ #

    def _chat_usage(self, u: Mapping[str, Any]) -> None:
        self.usage = {
            "input_tokens": _int(u.get("prompt_tokens")),
            "output_tokens": _int(u.get("completion_tokens")),
            "cached_tokens": _int((u.get("prompt_tokens_details") or {}).get("cached_tokens")),
            "reasoning_tokens": _int((u.get("completion_tokens_details") or {}).get("reasoning_tokens")),
        }

    def _responses_usage(self, u: Mapping[str, Any]) -> None:
        self.usage = {
            "input_tokens": _int(u.get("input_tokens")),
            "output_tokens": _int(u.get("output_tokens")),
            "cached_tokens": _int((u.get("input_tokens_details") or {}).get("cached_tokens")),
            "reasoning_tokens": _int((u.get("output_tokens_details") or {}).get("reasoning_tokens")),
        }

    def _anthropic_usage(self, u: Mapping[str, Any]) -> None:
        # Anthropic counts cache reads/writes outside input_tokens; AgentCAP's input includes them.
        cur = dict(self.usage)
        if any(k in u for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")):
            cached = _int(u.get("cache_read_input_tokens"))
            cur["input_tokens"] = _int(u.get("input_tokens")) + cached + _int(u.get("cache_creation_input_tokens"))
            cur["cached_tokens"] = cached
        if "output_tokens" in u:
            cur["output_tokens"] = _int(u.get("output_tokens"))
        cur.setdefault("reasoning_tokens", 0)
        self.usage = cur


# --------------------------------------------------------------------------- #
# Proxy server
# --------------------------------------------------------------------------- #

@dataclass
class MeterSession:
    """One trial's route through the proxy."""

    token: str
    upstream: str
    label: str
    port: int
    records: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def url(self, host: str) -> str:
        return f"http://{host}:{self.port}/{self.token}"

    def add(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self.records.append(record)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.records]


class MeterServer:
    """The proxy, on 0.0.0.0:<port> in a daemon thread. Use :meth:`shared`."""

    _shared: Optional["MeterServer"] = None
    _shared_lock = threading.Lock()

    def __init__(self, port: int = 0, bind: str = "0.0.0.0") -> None:
        self.bind = bind
        self.port = port
        self._sessions: Dict[str, MeterSession] = {}
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._serve, name="agentcap-meter", daemon=True)
        self._thread.start()
        self._ready.wait(30)
        if self._error is not None:
            raise RuntimeError(f"metering proxy failed to start: {self._error}")

    @classmethod
    def shared(cls) -> "MeterServer":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls(port=int(os.environ.get("AGENTCAP_METER_PORT", "0") or 0))
            return cls._shared

    # -- sessions ----------------------------------------------------------- #

    def open(self, upstream: str, label: str = "") -> MeterSession:
        session = MeterSession(uuid.uuid4().hex[:16], upstream.rstrip("/"), label, self.port)
        with self._lock:
            self._sessions[session.token] = session
        return session

    def close(self, session: MeterSession) -> None:
        with self._lock:
            self._sessions.pop(session.token, None)

    # -- server thread ---------------------------------------------------------- #

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start())
        except BaseException as exc:  # reported to the constructor
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        loop.run_forever()

    async def _start(self) -> None:
        self._client = aiohttp.ClientSession(
            trust_env=True,  # honour HTTPS_PROXY on the host
            timeout=aiohttp.ClientTimeout(total=None, connect=60, sock_read=1800),
            auto_decompress=False,
        )
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_route("*", "/{token}/{tail:.*}", self._handle)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind, self.port))
        self.port = sock.getsockname()[1]
        await web.SockSite(runner, sock).start()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        with self._lock:
            session = self._sessions.get(request.match_info["token"])
        if session is None:
            return web.json_response({"error": "unknown metering route"}, status=404)
        tail = request.match_info["tail"]
        url = f"{session.upstream}/{tail}" + (f"?{request.query_string}" if request.query_string else "")
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self._handle_ws(request, session, tail, url)
        t0 = time.perf_counter()
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["Accept-Encoding"] = "identity"
        req_json = _json_object(body)
        record = _new_record(request.method, tail, req_json)
        resp: Optional[web.StreamResponse] = None
        meter: Optional[ResponseMeter] = None
        try:
            async with self._client.request(request.method, url, headers=headers, data=body or None,
                                            allow_redirects=False) as up:
                ctype = up.headers.get("Content-Type", "")
                meter = ResponseMeter(streaming="text/event-stream" in ctype)
                out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP_BY_HOP}
                resp = web.StreamResponse(status=up.status, headers=out_headers)
                await resp.prepare(request)
                record["status"] = up.status
                async for chunk in up.content.iter_any():
                    meter.feed(chunk, time.perf_counter())
                    await resp.write(chunk)
                meter.finish(time.perf_counter())
                await resp.write_eof()
        except Exception as exc:  # upstream error or client gone; keep what was measured
            record["error"] = f"{type(exc).__name__}: {exc}"[:300]
            if resp is None:
                resp = web.json_response({"error": f"metering proxy: {record['error']}"}, status=502)
                record.setdefault("status", 502)
        finally:
            _finalize(record, meter, t0, time.perf_counter())
            record["llm"] = request.method == "POST" and tail.rstrip("/").endswith(LLM_PATH_SUFFIXES)
            session.add(record)
        return resp

    async def _handle_ws(self, request: web.Request, session: MeterSession, tail: str, url: str) -> web.StreamResponse:
        """Relay a WebSocket; each client ``response.create`` message is one metered request."""
        skip = HOP_BY_HOP | WS_HANDSHAKE
        headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
        protocols = [p.strip() for p in request.headers.get("Sec-WebSocket-Protocol", "").split(",") if p.strip()]
        ws_url = "ws" + url[4:] if url.startswith("http") else url
        try:
            upstream = await self._client.ws_connect(ws_url, headers=headers, protocols=protocols,
                                                     max_msg_size=0, autoping=True)
        except aiohttp.WSServerHandshakeError as exc:  # e.g. the endpoint has no WebSocket mode
            session.add({**_new_record("WS", tail, {}), "status": exc.status, "llm": False, "latency_s": 0.0,
                         "error": f"websocket handshake: {exc.status}"})
            return web.Response(status=exc.status or 502, text=str(exc.message or "websocket handshake failed"))
        except Exception as exc:
            return web.json_response({"error": f"metering proxy: {type(exc).__name__}: {exc}"[:300]}, status=502)
        client = web.WebSocketResponse(protocols=protocols, max_msg_size=0, autoping=True)
        await client.prepare(request)
        current: Dict[str, Any] = {}

        def close_current(error: Optional[str] = None) -> None:
            if not current:
                return
            record, meter, t0 = current.pop("record"), current.pop("meter"), current.pop("t0")
            if error:
                record["error"] = error
            _finalize(record, meter, t0, time.perf_counter())
            record["llm"] = True
            session.add(record)

        async def client_to_upstream() -> None:
            async for msg in client:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    payload = _json_object(msg.data.encode() if isinstance(msg.data, str) else msg.data)
                    if payload.get("type", "response.create") == "response.create":
                        close_current("superseded by a new request")
                        body = payload.get("response") if isinstance(payload.get("response"), dict) else payload
                        current.update(record={**_new_record("WS", tail, body), "stream": True, "status": 101},
                                       meter=ResponseMeter(streaming=True), t0=time.perf_counter())
                    await upstream.send_str(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await upstream.send_bytes(msg.data)
                else:
                    break

        async def upstream_to_client() -> None:
            async for msg in upstream:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = _json_object(msg.data.encode())
                    if current and event:
                        current["meter"].feed_event(event, time.perf_counter())
                    await client.send_str(msg.data)
                    if event.get("type") in WS_END_EVENTS:
                        close_current("upstream error event" if event.get("type") == "error" else None)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await client.send_bytes(msg.data)
                else:
                    break

        tasks = [asyncio.ensure_future(client_to_upstream()), asyncio.ensure_future(upstream_to_client())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            close_current("websocket closed before the response completed")
            await upstream.close()
            await client.close()
        return client


def _json_object(raw: bytes) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _new_record(method: str, tail: str, body: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "start_ms": int(time.time() * 1000),
        "method": method,
        "path": "/" + tail,
        "model": body.get("model"),
        "stream": bool(body.get("stream")),
        # Responses API warm-up (Codex sends one per WebSocket): no generation requested.
        **({"warmup": True} if body.get("generate") is False else {}),
    }


def _finalize(record: Dict[str, Any], meter: Optional[ResponseMeter], t0: float, end: float) -> None:
    record["latency_s"] = round(end - t0, 6)
    if meter is None:
        return
    first = meter.first_token if meter.first_token is not None else (None if meter.streaming else end)
    last = meter.last_token if meter.last_token is not None else first
    record["ttft_s"] = round(first - t0, 6) if first is not None else None
    record["decode_s"] = round(last - first, 6) if first is not None and last is not None else None
    record["num_tool_calls"] = meter.tool_calls
    record.update(meter.usage)


# --------------------------------------------------------------------------- #
# Records -> AgentCAP row
# --------------------------------------------------------------------------- #

_TOKEN_KEYS = ("input_tokens", "output_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "requests")


def records_to_row(
    records: List[Mapping[str, Any]],
    *,
    task_id: str,
    e2e_latency_s: float,
    errors: List[str],
    meta: Mapping[str, Any],
    output_text: str = "",
) -> Dict[str, Any]:
    """The AgentCAP per-task row (as in results.jsonl) for one trial's model requests.

    Role is ``agent``; when a harness uses several models (e.g. a cheap model for
    titles or subagents), each model is its own role.
    """
    calls = [r for r in records if r.get("llm") and not r.get("warmup")]
    warmups = [r for r in records if r.get("llm") and r.get("warmup")]
    models = {str(r.get("model") or "unknown") for r in calls}
    per_role: Dict[str, Dict[str, int]] = {}
    turn_stats: List[Dict[str, Any]] = []
    for r in sorted(calls, key=lambda r: r.get("start_ms") or 0):
        role = "agent" if len(models) <= 1 else str(r.get("model") or "unknown")
        usage = per_role.setdefault(role, {k: 0 for k in _TOKEN_KEYS})
        out = _int(r.get("output_tokens"))
        for key, value in (("input_tokens", _int(r.get("input_tokens"))), ("output_tokens", out),
                           ("completion_tokens", out), ("reasoning_tokens", _int(r.get("reasoning_tokens"))),
                           ("cached_tokens", _int(r.get("cached_tokens"))), ("requests", 1)):
            usage[key] += value
        turn_stats.append({
            "role": role,
            "model": r.get("model"),
            "start_ms": r.get("start_ms"),
            "status": r.get("status"),
            "stream": r.get("stream"),
            "input_tokens": _int(r.get("input_tokens")),
            "output_tokens": out,
            "completion_tokens": out,
            "reasoning_tokens": _int(r.get("reasoning_tokens")),
            "cached_tokens": _int(r.get("cached_tokens")),
            "latency_s": r.get("latency_s") or 0.0,
            "ttft_s": r.get("ttft_s") if r.get("ttft_s") is not None else 0.0,
            "decode_s": r.get("decode_s") if r.get("decode_s") is not None else 0.0,
            "num_tool_calls": _int(r.get("num_tool_calls")),
            **({"error": r["error"]} if r.get("error") else {}),
        })
    total = {k: sum(u[k] for u in per_role.values()) for k in _TOKEN_KEYS}
    failed = [t for t in turn_stats if (t.get("status") or 0) >= 400 or t.get("error")]
    return {
        "task_id": task_id,
        "strategy": meta.get("strategy"),
        "output_text": output_text,
        "e2e_latency_s": round(e2e_latency_s, 4),
        "per_role_usage": per_role,
        "total_usage": total,
        "errors": list(errors) + ([f"{len(failed)} model request(s) failed"] if failed else []),
        "num_turns": total["requests"],
        "tool_calls": sum(t["num_tool_calls"] for t in turn_stats),
        "turn_stats": turn_stats,
        "agentcard": {
            **dict(meta),
            "metering": "meter-proxy",
            "roles": meta.get("roles") or {
                role: (role if role != "agent" else next(iter(models), None)) for role in per_role},
            # Not model turns, but real requests: keep them for cost accounting.
            "warmup_requests": {"requests": len(warmups),
                                "input_tokens": sum(_int(r.get("input_tokens")) for r in warmups),
                                "output_tokens": sum(_int(r.get("output_tokens")) for r in warmups)},
        },
    }


__all__ = ["MeterServer", "MeterSession", "ResponseMeter", "records_to_row"]
