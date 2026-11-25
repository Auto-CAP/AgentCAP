"""One interface for every harness on Harbor: connection flags, metering proxy, CLI, export."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

if importlib.util.find_spec("harbor") is None:
    pytest.skip("harbor not installed", allow_module_level=True)

import yaml  # noqa: E402
from harbor.agents.base import BaseAgent  # noqa: E402
from harbor.agents.capabilities import AgentCapabilities  # noqa: E402
from harbor.models.model_api_format import ModelApiFormat  # noqa: E402
from harbor.utils.import_path import import_class  # noqa: E402

from agent_cap.harbor import METER_FILENAME, cli, jobspec  # noqa: E402
from agent_cap.harbor.connection import connection_spec  # noqa: E402
from agent_cap.harbor.export_teas import trial_row  # noqa: E402
from agent_cap.harbor.meter import MeterServer, ResponseMeter, records_to_row  # noqa: E402
from agent_cap.harbor.metered import gateway_from_proc_route, metered  # noqa: E402

# --------------------------------------------------------------------------- #
# Stream parsing
# --------------------------------------------------------------------------- #


def _sse(events, named=False):
    out = b""
    for ev in events:
        if named:
            out += f"event: {ev['type']}\n".encode()
        out += b"data: " + json.dumps(ev).encode() + b"\n\n"
    return out


def _feed(meter, payload, t0=1.0, step=0.1):
    for i, line in enumerate(payload.split(b"\n\n")):
        meter.feed(line + b"\n\n", t0 + i * step)
    meter.finish(t0 + 99)


def test_meter_chat_completions_stream():
    events = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},  # no token yet
        {"choices": [{"index": 0, "delta": {"reasoning_content": "hm"}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "bash"}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}]},
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 7,
                                  "prompt_tokens_details": {"cached_tokens": 40},
                                  "completion_tokens_details": {"reasoning_tokens": 3}}},
    ]
    m = ResponseMeter(streaming=True)
    _feed(m, _sse(events) + b"data: [DONE]\n\n")
    assert m.first_token == pytest.approx(1.1) and m.last_token == pytest.approx(1.3)
    assert m.tool_calls == 1
    assert m.usage == {"input_tokens": 100, "output_tokens": 7, "cached_tokens": 40, "reasoning_tokens": 3}


def test_meter_responses_and_anthropic_streams():
    responses = [
        {"type": "response.created", "response": {"id": "r"}},
        {"type": "response.output_item.added", "item": {"type": "function_call"}},
        {"type": "response.function_call_arguments.delta", "delta": "{"},
        {"type": "response.completed", "response": {"usage": {
            "input_tokens": 50, "output_tokens": 9, "input_tokens_details": {"cached_tokens": 10},
            "output_tokens_details": {"reasoning_tokens": 4}}}},
    ]
    m = ResponseMeter(streaming=True)
    _feed(m, _sse(responses, named=True))
    assert m.first_token == pytest.approx(1.2) and m.tool_calls == 1
    assert m.usage == {"input_tokens": 50, "output_tokens": 9, "cached_tokens": 10, "reasoning_tokens": 4}

    anthropic = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5, "cache_read_input_tokens": 20,
                                                        "cache_creation_input_tokens": 3, "output_tokens": 1}}},
        {"type": "content_block_start", "content_block": {"type": "tool_use"}},
        {"type": "content_block_delta", "delta": {"type": "input_json_delta"}},
        {"type": "message_delta", "usage": {"output_tokens": 12}},
    ]
    m = ResponseMeter(streaming=True)
    _feed(m, _sse(anthropic, named=True))
    assert m.tool_calls == 1 and m.first_token == pytest.approx(1.2)
    assert m.usage["input_tokens"] == 28 and m.usage["cached_tokens"] == 20 and m.usage["output_tokens"] == 12


def test_meter_plain_json_body():
    m = ResponseMeter(streaming=False)
    m.feed(json.dumps({"choices": [{"message": {"tool_calls": [{}, {}]}}],
                       "usage": {"prompt_tokens": 3, "completion_tokens": 2}}).encode(), 1.0)
    m.finish(2.0)
    assert m.tool_calls == 2 and m.usage["input_tokens"] == 3 and m.first_token is None


def test_gateway_from_proc_route():
    route = ("Iface\tDestination\tGateway \tFlags\n"
             "eth0\t00000000\t010012AC\t0003\n"
             "eth0\t000012AC\t00000000\t0001\n")
    assert gateway_from_proc_route(route) == "172.18.0.1"
    assert gateway_from_proc_route("Iface\tDestination\tGateway\n") is None


# --------------------------------------------------------------------------- #
# Proxy end to end
# --------------------------------------------------------------------------- #

CHUNKS = [
    {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
    {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
    {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
    {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2}},
]


class _Upstream(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        _Upstream.seen.append((self.path, self.headers.get("authorization"), json.loads(body)))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        import time

        for c in CHUNKS:
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture()
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_proxy_forwards_and_records(upstream):
    server = MeterServer(port=0, bind="127.0.0.1")
    session = server.open(upstream, label="t")
    req = urllib.request.Request(
        session.url("127.0.0.1") + "/chat/completions",
        data=json.dumps({"model": "m1", "stream": True, "messages": []}).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer k"},
    )
    body = urllib.request.urlopen(req, timeout=10).read()
    assert body.count(b"data:") == len(CHUNKS) + 1  # streamed through unchanged
    path, auth, sent = _Upstream.seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer k" and sent["model"] == "m1"
    (rec,) = session.snapshot()
    assert rec["llm"] and rec["status"] == 200 and rec["model"] == "m1"
    assert rec["input_tokens"] == 11 and rec["output_tokens"] == 2
    assert 0.05 <= rec["ttft_s"] < rec["latency_s"] and rec["decode_s"] >= 0.04
    row = records_to_row(session.snapshot(), task_id="t", e2e_latency_s=1.0, errors=[], meta={"strategy": "x"})
    assert row["per_role_usage"]["agent"]["requests"] == 1 and row["agentcard"]["metering"] == "meter-proxy"
    server.close(session)
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(urllib.request.Request(session.url("127.0.0.1") + "/chat/completions", data=b"{}"))


# --------------------------------------------------------------------------- #
# Metered agents
# --------------------------------------------------------------------------- #


class _ToyAgent(BaseAgent):
    """Calls its model once through Harbor's connection, like an installed agent would."""

    capabilities = AgentCapabilities(base_url=True)
    MODEL_CONNECTION = connection_spec(ModelApiFormat.OPENAI_CHAT_COMPLETIONS)

    @staticmethod
    def name():
        return "toy"

    def version(self):
        return "1"

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        url = self._execution_base_url + "/chat/completions"
        data = json.dumps({"model": self.model_connection.model_id, "stream": True}).encode()
        await asyncio.to_thread(lambda: urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10).read())


def test_metered_agent_records_requests(tmp_path, upstream):
    from tests.harbor_fakes import LocalEnvironment

    cls = metered(_ToyAgent)
    assert cls.name() == "toy" and issubclass(cls, _ToyAgent)
    cls.preflight(kwargs={"meter_host": "127.0.0.1"}, env={})  # our kwargs never reach the agent's options
    logs = tmp_path / "task-1__abc" / "agent"
    agent = cls(logs_dir=logs, model_name="toy-model", base_url=upstream, meter_host="127.0.0.1")
    env = LocalEnvironment(tmp_path)

    async def go():
        await agent.setup(env)
        await agent.run("hi", env, None)

    asyncio.run(go())
    out = json.loads((logs / METER_FILENAME).read_text())
    assert out["metered"] and out["task_id"] == "task-1"
    assert out["agentcard"]["upstream"] == upstream and out["agentcard"]["agent"] == "toy"
    assert out["total_usage"]["requests"] == 1 and out["turn_stats"][0]["ttft_s"] > 0


def test_metered_agent_without_endpoint_runs_unmetered(tmp_path):
    from tests.harbor_fakes import LocalEnvironment

    class NoUrl(_ToyAgent):
        capabilities = AgentCapabilities(base_url=False)

        async def run(self, instruction, environment, context):
            pass

    agent = metered(NoUrl)(logs_dir=tmp_path / "agent", model_name="m", meter_host="127.0.0.1")
    asyncio.run(agent.run("hi", LocalEnvironment(tmp_path), None))
    out = json.loads((tmp_path / "agent" / METER_FILENAME).read_text())
    assert out["metered"] is False and "does not support --base-url" in out["reason"]


def test_metered_names_resolve_for_harbor_agents():
    for name in ("codex", "claude-code", "mini-swe-agent", "dsh"):
        cls = import_class(f"agent_cap.harbor.metered:{name}")
        assert cls.import_path() == f"agent_cap.harbor.metered:{name}"
        assert import_class(cls.import_path()) is cls
    assert import_class("agent_cap.harbor:CAPAgent").name() == "capagent"


# --------------------------------------------------------------------------- #
# Harbor's connection on AgentCAP's own harnesses
# --------------------------------------------------------------------------- #


def test_capagent_team_from_connection(tmp_path, monkeypatch):
    from agent_cap.harbor.agent import CAPAgent

    monkeypatch.setenv("MY_KEY", "k-123")
    agent = CAPAgent(logs_dir=tmp_path, model_name="Qwen/Qwen3.5-4B", base_url="http://h:8000/v1",
                     model_api_key_name="MY_KEY", strategy="plan-execute", role_options={"max_tokens": 512})
    specs = agent.specs
    assert sorted(specs) == ["executor", "planner"]
    ep = specs["planner"].endpoint
    assert (ep.name, ep.base_url, ep.api_key, ep.max_tokens) == ("Qwen/Qwen3.5-4B", "http://h:8000/v1", "k-123", 512)
    with pytest.raises(ValueError):
        CAPAgent(logs_dir=tmp_path)


def test_dsh_endpoint_from_connection(tmp_path, monkeypatch):
    from agent_cap.harbor.dsh import DSHAgent

    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    ep = DSHAgent(logs_dir=tmp_path, model_name="deepseek/deepseek-chat").endpoint()
    assert ep == {"model": "deepseek-chat", "base_url": "https://api.deepseek.com",
                  "protocol": "openai-completions", "api_key": "ds-key"}
    ep = DSHAgent(logs_dir=tmp_path, model_name="m", base_url="http://h/v1",
                  model_api_format="anthropic_messages").endpoint()
    assert ep["protocol"] == "anthropic-messages" and ep["base_url"] == "http://h/v1"
    legacy = DSHAgent(logs_dir=tmp_path, model="m", endpoint_url="http://old/v1")
    assert legacy.endpoint()["base_url"] == "http://old/v1" and legacy.upstream_base_url() == "http://old/v1"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _job(tmp_path, *argv):
    out = tmp_path / "job.yaml"
    assert cli.main(["job", "--benchmark", "nl2repo", "--dataset", str(tmp_path / "ds"), "--out", str(out), *argv]) == 0
    config = yaml.safe_load(out.read_text())
    jobspec.validate_job_config(config)
    return config["agents"][0]


def test_cli_same_flags_for_every_harness(tmp_path):
    common = ["--model", "openai/gpt-5.4", "--base-url", "http://h/v1", "--api-key-name", "OPENAI_API_KEY"]
    codex = _job(tmp_path, "--harness", "codex", *common, "--ak", "version=0.130.0", "--ae", "FOO=${BAR}")
    assert codex["import_path"] == "agent_cap.harbor.metered:codex"
    assert (codex["model_name"], codex["base_url"], codex["model_api_key_name"]) == ("openai/gpt-5.4", "http://h/v1", "OPENAI_API_KEY")
    assert codex["kwargs"] == {"version": "0.130.0"} and codex["env"] == {"FOO": "${BAR}"}

    plain = _job(tmp_path, "--harness", "claude-code", "--model", "anthropic/claude-sonnet-4-5", "--no-meter")
    assert plain["name"] == "claude-code" and "import_path" not in plain

    cap = _job(tmp_path, "--harness", "capagent", *common, "--strategy", "plan-execute")
    assert cap["import_path"] == jobspec.AGENT_IMPORT_PATH and "roles" not in cap["kwargs"]
    assert cap["kwargs"]["strategy"] == "plan-execute" and cap["kwargs"]["tool_backend"] == "shell"

    dsh = _job(tmp_path, "--harness", "dsh", "--model", "deepseek/deepseek-chat", "--ak", "teammates=2")
    assert dsh["import_path"] == jobspec.DSH_IMPORT_PATH and dsh["kwargs"]["teammates"] == 2
    assert _job(tmp_path, "--harness", "dsh", "--model", "deepseek/deepseek-chat", "--meter")["import_path"] == \
        "agent_cap.harbor.metered:dsh"

    custom = _job(tmp_path, "--harness", "my_pkg.agents:MyAgent", "--model", "m", "--no-meter")
    assert custom["import_path"] == "my_pkg.agents:MyAgent"


def test_cli_rejects_unknown_harness_and_misplaced_options(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["job", "--benchmark", "nl2repo", "--dataset", "d", "--out", str(tmp_path / "j.yaml"),
                  "--harness", "not-an-agent", "--model", "m"])
    with pytest.raises(SystemExit):
        cli.main(["job", "--benchmark", "nl2repo", "--dataset", "d", "--out", str(tmp_path / "j.yaml"),
                  "--harness", "codex", "--model", "m", "--role", "agent=name=x"])


# --------------------------------------------------------------------------- #
# Export for any harness
# --------------------------------------------------------------------------- #


def _trial(tmp_path, name, *, agent_result=None, agent_files=None):
    trial = tmp_path / name
    (trial / "agent").mkdir(parents=True)
    result = {
        "task_name": "nl2repo/math-verify",
        "agent_info": {"name": "codex", "version": "0.130.0"},
        "config": {"agent": {"model_name": "openai/gpt-5.4"}},
        "agent_result": agent_result,
        "verifier_result": {"rewards": {"reward": 0.5, "pass_all": 0}},
        "agent_execution": {"started_at": "2026-11-25T10:00:00Z", "finished_at": "2026-11-25T10:01:30Z"},
    }
    (trial / "result.json").write_text(json.dumps(result))
    for fname, content in (agent_files or {}).items():
        (trial / "agent" / fname).write_text(json.dumps(content))
    return trial


def test_export_harbor_context_and_atif(tmp_path):
    traj = {"steps": [
        {"source": "user", "message": "x"},
        {"source": "agent", "model_name": "gpt-5.4", "tool_calls": [{}], "metrics": {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 60}},
        {"source": "agent", "model_name": "gpt-5.4", "metrics": {"prompt_tokens": 150, "completion_tokens": 20}},
    ]}
    trial = _trial(tmp_path, "math-verify__a", agent_result={"n_input_tokens": 250, "n_output_tokens": 30, "n_cache_tokens": 60},
                   agent_files={"trajectory.json": traj, METER_FILENAME: {"metered": False, "reason": "no endpoint"}})
    row = trial_row(trial)
    assert row["agentcard"]["metering"] == "harbor-context" and row["agentcard"]["roles"] == {"agent": "openai/gpt-5.4"}
    assert row["total_usage"]["input_tokens"] == 250 and row["total_usage"]["requests"] == 2
    assert row["tool_calls"] == 1 and row["e2e_latency_s"] == 90.0
    assert row["errors"] == ["unmetered: no endpoint"] and row["eval_score"] == 0.5 and row["eval_passed"] is False


def test_export_prefers_meter_rows(tmp_path):
    meter = records_to_row([{"llm": True, "model": "gpt-5.4", "start_ms": 1, "status": 200, "input_tokens": 9,
                             "output_tokens": 1, "ttft_s": 0.2, "decode_s": 0.1, "latency_s": 0.4}],
                           task_id="math-verify", e2e_latency_s=2.0, errors=[], meta={"agent": "codex", "strategy": "codex"})
    meter["metered"] = True
    trial = _trial(tmp_path, "math-verify__b", agent_result={"n_input_tokens": 9}, agent_files={METER_FILENAME: meter})
    row = trial_row(trial)
    assert row["agentcard"]["metering"] == "meter-proxy" and row["turn_stats"][0]["ttft_s"] == 0.2
    assert row["agentcard"]["harbor_context"]["n_input_tokens"] == 9 and "requests_raw" not in row


def test_proxy_relays_and_meters_websocket():
    """Responses API over WebSocket (Codex's default transport) goes through the proxy unchanged."""
    import aiohttp
    from aiohttp import web

    async def ws_upstream(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            body = json.loads(msg.data)
            assert body["type"] == "response.create"
            for ev in ({"type": "response.created"}, {"type": "response.output_text.delta", "delta": "hi"},
                       {"type": "response.completed", "response": {"usage": {"input_tokens": 7, "output_tokens": 3}}}):
                await asyncio.sleep(0.03)
                await ws.send_str(json.dumps(ev))
        return ws

    async def go():
        app = web.Application()
        app.router.add_get("/v1/responses", ws_upstream)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        server = MeterServer(port=0, bind="127.0.0.1")
        session = server.open(f"http://127.0.0.1:{port}/v1")
        got = []
        async with aiohttp.ClientSession() as client:
            async with client.ws_connect(session.url("127.0.0.1") + "/responses",
                                         headers={"Authorization": "Bearer k"}) as ws:
                for _ in range(2):  # two requests on one socket
                    await ws.send_str(json.dumps({"type": "response.create", "model": "gpt-x", "input": []}))
                    while True:
                        ev = json.loads((await ws.receive()).data)
                        got.append(ev["type"])
                        if ev["type"] == "response.completed":
                            break
        await runner.cleanup()
        return session.snapshot(), got

    records, got = asyncio.run(go())
    assert got.count("response.completed") == 2
    llm = [r for r in records if r["llm"]]
    assert len(llm) == 2 and all(r["method"] == "WS" and r["model"] == "gpt-x" for r in llm)
    assert llm[0]["input_tokens"] == 7 and llm[0]["output_tokens"] == 3
    assert 0.05 <= llm[0]["ttft_s"] < llm[0]["latency_s"] and "error" not in llm[0]
