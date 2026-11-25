"""Tool providers that execute AgentCAP tools inside a Harbor environment.

``HarborTools`` implements AgentCAP's ``ToolProvider`` protocol
(``list_tools`` / ``call``) on top of Harbor's ``BaseEnvironment.exec``. The
agent loop stays on the host; only tool execution crosses into the task
container. Backends:

``shell``
    SWE-agent style ``bash``, ``str_replace_editor`` and ``submit`` (used for
    NL2Repo and SWE-bench), with the tool schemas of
    ``agent_cap.backends.tool_executor``. ``bash`` and ``submit`` reuse
    ``ToolExecutor`` unchanged, routed through ``environment.exec``;
    ``str_replace_editor`` runs SWE-agent's official editor script
    (``agent_cap/backends/swe_agent_tools``) inside the container.
``math-python``
    The ``python`` tool backed by a persistent Jupyter kernel running in the
    container (``pykernel_server.py``), with the same semantics as
    ``agent_cap.backends.math_python_backend.MathPythonBackend``.
``none``
    No tools.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKENDS = ("none", "shell", "math-python")

KERNEL_DIR = "/tmp/capagent"
KERNEL_SCRIPT = f"{KERNEL_DIR}/pykernel_server.py"
KERNEL_SOCKET = f"{KERNEL_DIR}/py.sock"
KERNEL_LOG = f"{KERNEL_DIR}/pykernel.log"
KERNEL_PID = f"{KERNEL_DIR}/pykernel.pid"
_KERNEL_UNAVAILABLE = "[CAPAGENT_KERNEL_UNAVAILABLE]"


class HarborToolError(RuntimeError):
    """Raised when a tool backend cannot be prepared inside the environment."""


class HarborTools:
    """AgentCAP ``ToolProvider`` whose tools run in a Harbor environment."""

    def __init__(
        self,
        environment: Any,
        backend: str = "none",
        *,
        workdir: str = "/workspace",
        shell_timeout: int = 450,
        max_output_chars: int = 100_000,
        python_bin: str = "python3",
        python_preload: str = "minimal",
        python_startup_timeout: float = 30.0,
        python_exec_timeout: float = 30.0,
        auto_print_last_expr: bool = True,
        kernel_dir: str = KERNEL_DIR,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"unknown tool backend {backend!r}; choose one of {BACKENDS}")
        self.environment = environment
        self.backend = backend
        self.workdir = workdir
        self.shell_timeout = int(shell_timeout)
        self.max_output_chars = int(max_output_chars)
        self.python_bin = python_bin
        self.python_preload = python_preload
        self.python_startup_timeout = float(python_startup_timeout)
        self.python_exec_timeout = float(python_exec_timeout)
        self.auto_print_last_expr = auto_print_last_expr
        self.kernel_dir = kernel_dir.rstrip("/")
        self._kernel_script = f"{self.kernel_dir}/pykernel_server.py"
        self._kernel_socket = f"{self.kernel_dir}/py.sock"
        self._kernel_log = f"{self.kernel_dir}/pykernel.log"
        self._kernel_pid = f"{self.kernel_dir}/pykernel.pid"
        self._editor_dir = f"{self.kernel_dir}/swe_tools"
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor: Any = None
        self._last_print: Any = None
        self._prepared = False

    # ------------------------------------------------------------------ #
    # ToolProvider protocol
    # ------------------------------------------------------------------ #

    async def list_tools(self) -> List[Dict[str, Any]]:
        if self.backend == "shell":
            from agent_cap.backends.tool_executor import TOOL_DEFINITIONS

            return [dict(t) for t in TOOL_DEFINITIONS]
        if self.backend == "math-python":
            from agent_cap.backends.math_python_backend import PYTHON_TOOL_DEFINITIONS

            return [dict(t) for t in PYTHON_TOOL_DEFINITIONS]
        return []

    async def call(self, name: str, arguments: Dict[str, Any]) -> str:
        if not self._prepared:
            await self.prepare()
        if self.backend == "shell":
            return await self._call_shell(name, arguments)
        if self.backend == "math-python":
            return await self._call_python(name, arguments)
        return f"[ERROR] Unknown tool: {name}"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def prepare(self) -> None:
        """Start whatever the backend needs inside the environment. Idempotent."""
        if self._prepared:
            return
        self._loop = asyncio.get_running_loop()
        if self.backend == "shell":
            from agent_cap.backends.tool_executor import ToolExecutor

            await self._exec(f"mkdir -p {shlex.quote(self.workdir)}", timeout=60, cwd="/")
            self._executor = ToolExecutor(
                workspace_dir=self.workdir,
                shell_timeout=self.shell_timeout,
                max_output_chars=self.max_output_chars,
                exec_fn=self._sync_exec,
            )
            await self._install_editor()
        elif self.backend == "math-python":
            from agent_cap.backends.math_python_backend import MathPythonBackend

            # Only the AST rewrite is used on the host; no kernel starts here.
            self._last_print = MathPythonBackend(auto_print_last_expr=True)
            await self._start_kernel()
        self._prepared = True

    async def teardown(self) -> None:
        if self.backend == "math-python" and self._prepared:
            await self._stop_kernel()

    async def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to ``path`` inside the environment."""
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "content"
            local.write_text(content, encoding="utf-8")
            parent = str(Path(path).parent)
            await self._exec(f"mkdir -p {shlex.quote(parent)}", timeout=60, cwd="/")
            await self.environment.upload_file(local, path)

    # ------------------------------------------------------------------ #
    # shell backend
    # ------------------------------------------------------------------ #

    async def _call_shell(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "str_replace_editor":
            return await self._call_editor(arguments or {})
        result = await asyncio.to_thread(self._executor.execute, name, "", arguments or {})
        return result.output

    async def _install_editor(self) -> None:
        tools_dir = Path(__file__).resolve().parent.parent / "backends" / "swe_agent_tools"
        await self._exec(f"mkdir -p {shlex.quote(self._editor_dir)}", timeout=60, cwd="/")
        for local, name in (
            (tools_dir / "str_replace_editor", "str_replace_editor"),
            (tools_dir / "registry.py", "registry.py"),
            (Path(__file__).with_name("editor_call.py"), "editor_call.py"),
        ):
            await self.environment.upload_file(local, f"{self._editor_dir}/{name}")

    async def _call_editor(self, arguments: Dict[str, Any]) -> str:
        if not arguments.get("command") or not arguments.get("path"):
            return "ERROR: 'command' and 'path' are required."
        args_path = f"{self._editor_dir}/args-{uuid.uuid4().hex}.json"
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "args.json"
            local.write_text(json.dumps(arguments), encoding="utf-8")
            await self.environment.upload_file(local, args_path)
        proc = await self._exec(
            f"{shlex.quote(self.python_bin)} {self._editor_dir}/editor_call.py {args_path} 2>&1; "
            f"_ec=$?; rm -f {args_path}; exit $_ec",
            timeout=self.shell_timeout,
        )
        output = proc.stdout or proc.stderr or ""
        if len(output) > self.max_output_chars:  # same clipping as ToolExecutor.execute
            output = output[: self.max_output_chars] + "\n<response clipped>"
        return output

    def _sync_exec(self, cmd: str, timeout: Optional[int] = None) -> Optional[subprocess.CompletedProcess]:
        """Bridge for ToolExecutor (sync, worker thread) onto the async environment."""
        if self._loop is None:
            raise HarborToolError("HarborTools.prepare() was not awaited")
        limit = int(timeout or self.shell_timeout)
        future = asyncio.run_coroutine_threadsafe(self._exec(cmd, timeout=limit), self._loop)
        try:
            return future.result(timeout=limit + 60)
        except Exception:
            future.cancel()
            return None

    # ------------------------------------------------------------------ #
    # math-python backend
    # ------------------------------------------------------------------ #

    async def _stop_kernel(self) -> None:
        # A PID file rather than `pkill -f`: the pattern would also match the shell running it.
        await self._exec(
            f"if [ -f {self._kernel_pid} ]; then kill $(cat {self._kernel_pid}) 2>/dev/null; rm -f {self._kernel_pid}; fi; true",
            timeout=30,
            cwd="/",
        )

    async def _start_kernel(self) -> None:
        script = Path(__file__).with_name("pykernel_server.py")
        await self._exec(f"mkdir -p {shlex.quote(self.kernel_dir)}", timeout=60, cwd="/")
        await self.environment.upload_file(script, self._kernel_script)
        await self._stop_kernel()
        start = (
            f"rm -f {self._kernel_socket}; "
            f"nohup {shlex.quote(self.python_bin)} {self._kernel_script} serve --socket {self._kernel_socket} "
            f"--startup-timeout {self.python_startup_timeout} "
            f"--exec-timeout {self.python_exec_timeout} --preload {self.python_preload} "
            f"> {self._kernel_log} 2>&1 & echo $! > {self._kernel_pid}"
        )
        await self._exec(start, timeout=60, cwd="/")
        deadline = asyncio.get_running_loop().time() + self.python_startup_timeout + 60
        while asyncio.get_running_loop().time() < deadline:
            probe = await self._exec(
                f"{shlex.quote(self.python_bin)} {self._kernel_script} run --socket {self._kernel_socket} --ping",
                timeout=30,
                cwd="/",
            )
            if probe.returncode == 0 and (probe.stdout or "").strip() == "pong":
                return
            await asyncio.sleep(1.0)
        log = await self._exec(f"tail -n 40 {self._kernel_log}", timeout=30, cwd="/")
        raise HarborToolError(f"Python kernel did not start in the environment:\n{log.stdout}")

    async def _call_python(self, name: str, arguments: Dict[str, Any]) -> str:
        # Same argument handling and messages as MathPythonBackend.execute.
        if "python" not in name:
            return f"[ERROR] Unknown tool: {name}"
        arguments = arguments or {}
        code = arguments.get("code", "")
        if code == "":
            code = arguments.get("raw", "")
        if not isinstance(code, str) or not code.strip():
            return (
                "[ERROR] Tool argument 'code' must be a non-empty string. Currently accepted "
                "keys in arg-dict: 'code' and 'raw'. Please check arg-dict again."
            )
        final_code = self._last_print._ensure_last_print(code) if self.auto_print_last_expr else code
        output = await self._run_in_kernel(final_code)
        if output is None:  # server gone (e.g. OOM): restart once, as a host kernel would
            await self._start_kernel()
            output = await self._run_in_kernel(final_code)
        if output is None:
            return "[ERROR] Python kernel is unavailable in the task environment."
        return output

    async def _run_in_kernel(self, code: str) -> Optional[str]:
        b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
        cmd = (
            f"printf %s {b64} | base64 -d | {shlex.quote(self.python_bin)} {self._kernel_script} run "
            f"--socket {self._kernel_socket} --timeout {self.python_exec_timeout}"
        )
        # The kernel enforces exec_timeout and restarts itself; allow for that.
        proc = await self._exec(cmd, timeout=int(self.python_exec_timeout + self.python_startup_timeout + 60))
        if proc.returncode == 3 or _KERNEL_UNAVAILABLE in (proc.stderr or ""):
            return None
        if proc.returncode != 0:
            return f"[ERROR] {(proc.stderr or proc.stdout or '').strip()}"
        return proc.stdout or ""

    # ------------------------------------------------------------------ #
    # environment.exec wrapper
    # ------------------------------------------------------------------ #

    async def _exec(
        self, cmd: str, timeout: int, cwd: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        try:
            res = await self.environment.exec(
                cmd, cwd=cwd or self.workdir, timeout_sec=int(timeout)
            )
        except Exception as exc:  # timeouts and transport errors become tool output
            return subprocess.CompletedProcess(
                args=cmd, returncode=124, stdout="", stderr=f"{type(exc).__name__}: {exc}"
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=int(res.return_code),
            stdout=res.stdout or "",
            stderr=res.stderr or "",
        )


__all__ = ["HarborTools", "HarborToolError", "BACKENDS"]
