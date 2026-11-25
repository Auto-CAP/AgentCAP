#!/usr/bin/env python3
"""Persistent Jupyter kernel behind a Unix socket, for use inside a Harbor task container.

CAPAgent uploads this file into the task container and starts it once per trial::

    python3 pykernel_server.py serve --socket /tmp/capagent/py.sock

Each ``python`` tool call then runs::

    printf %s <base64 code> | base64 -d | python3 pykernel_server.py run --socket ...

The execution semantics mirror ``agent_cap.backends.math_python_backend.PythonSandbox``
(same kernel environment, preload, timeout-and-restart behaviour, error formatting
and "no output" warning), so tool outputs match what AgentCAP produces when the
kernel runs on the host. The last-expression ``print`` rewrite is applied on the
host by CAPAgent before the code is sent here.

This file must stay self-contained: it runs with the container's Python and only
needs ``jupyter_client`` and ``ipykernel`` installed there.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import re
import socket
import sys
import tempfile
import time
import uuid
from typing import List, Optional

_MAX_REQUEST_BYTES = 64 * 1024 * 1024


class KernelSandbox:
    """Copy of PythonSandbox's behaviour; see agent_cap/backends/math_python_backend.py."""

    def __init__(self, startup_timeout: float, exec_timeout: float, preload: str) -> None:
        self._startup_timeout = startup_timeout
        self._exec_timeout = exec_timeout
        self._preload = preload
        self._km = None
        self._client = None
        self._env = os.environ.copy()
        self._env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        self._env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        self._env["JUPYTER_PLATFORM_DIRS"] = "1"
        self._env["PYTHONWARNINGS"] = "ignore"
        self._env["MPLBACKEND"] = "Agg"
        self._extra_args = [
            "--Application.log_level=CRITICAL",
            "--HistoryManager.enabled=False",
        ]
        self._start_kernel()
        self._preload_modules()

    def _start_kernel(self) -> None:
        from jupyter_client import KernelManager

        connection_file = os.path.join(
            tempfile.gettempdir(), f"python-sandbox-{uuid.uuid4().hex}.json"
        )
        self._km = KernelManager(connection_file=connection_file)
        self._km.start_kernel(env=self._env, extra_arguments=self._extra_args)
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._startup_timeout)

    def _preload_modules(self) -> None:
        if self._preload == "minimal":
            self.execute(
                "import math\nimport mpmath\nmpmath.mp.dps = 64\n",
                timeout=self._startup_timeout,
            )
        elif self._preload == "full":
            self.execute(
                "import math\nimport numpy\nimport sympy\nimport itertools\n"
                "import collections\nimport mpmath\nmpmath.mp.dps = 64\n",
                timeout=self._startup_timeout,
            )
        elif self._preload != "none":
            raise ValueError(f"Unknown preload={self._preload!r}")

    def restart(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
        self._client = None
        self._km = None
        self._start_kernel()
        self._preload_modules()

    @staticmethod
    def _format_error(traceback: List[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if 'File "' in clean_frame and "ipython-input" not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return "".join(clean_lines)

    def execute(self, code: str, timeout: Optional[float] = None) -> str:
        client = self._client
        effective_timeout = timeout if timeout is not None else self._exec_timeout
        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        start_time = time.time()
        while True:
            if time.time() - start_time > effective_timeout:
                with contextlib.suppress(Exception):
                    self._km.interrupt_kernel()
                with contextlib.suppress(Exception):
                    self.restart()
                return (
                    f"[ERROR] Execution timed out after {effective_timeout} seconds "
                    "(kernel restarted)"
                )
            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg.get("msg_type")
            content = msg.get("content", {})
            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                stderr_parts.append(self._format_error(content.get("traceback", [])))
            elif msg_type in {"execute_result", "display_data"}:
                text = content.get("data", {}).get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

    def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.stop_channels()
        if self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)


def _recv_line(conn: socket.socket) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if chunk.endswith(b"\n") or total > _MAX_REQUEST_BYTES:
            break
    return b"".join(chunks)


def serve(args: argparse.Namespace) -> int:
    sandbox = KernelSandbox(args.startup_timeout, args.exec_timeout, args.preload)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(args.socket)
    os.makedirs(os.path.dirname(args.socket) or ".", exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(8)
    print(f"pykernel ready on {args.socket}", flush=True)
    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    request = json.loads(_recv_line(conn).decode("utf-8") or "{}")
                    if request.get("op") == "ping":
                        reply = {"output": "pong"}
                    else:
                        timeout = request.get("timeout")
                        output = sandbox.execute(
                            str(request.get("code", "")),
                            timeout=float(timeout) if timeout is not None else None,
                        )
                        reply = {"output": output}
                except Exception as exc:  # keep serving after a bad request
                    reply = {"output": f"[ERROR] {type(exc).__name__}: {exc}"}
                conn.sendall(json.dumps(reply).encode("utf-8") + b"\n")
    finally:
        sandbox.close()
        server.close()


def run(args: argparse.Namespace) -> int:
    code = sys.stdin.read() if not args.ping else ""
    payload = {"op": "ping"} if args.ping else {"code": code, "timeout": args.timeout}
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(args.socket)
    except OSError as exc:
        print(f"[CAPAGENT_KERNEL_UNAVAILABLE] {exc}", file=sys.stderr)
        return 3
    with client:
        client.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        reply = json.loads(_recv_line(client).decode("utf-8"))
    sys.stdout.write(reply.get("output", ""))
    sys.stdout.flush()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--socket", required=True)
    p_serve.add_argument("--startup-timeout", type=float, default=30.0)
    p_serve.add_argument("--exec-timeout", type=float, default=30.0)
    p_serve.add_argument("--preload", default="minimal", choices=["minimal", "full", "none"])
    p_run = sub.add_parser("run")
    p_run.add_argument("--socket", required=True)
    p_run.add_argument("--timeout", type=float, default=None)
    p_run.add_argument("--ping", action="store_true")
    args = parser.parse_args(argv)
    return serve(args) if args.cmd == "serve" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
