import ast
import contextlib
import os
import queue
import re
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from jupyter_client import KernelManager

from agent_cap.core.tool_backend import ToolBackend, ToolResult


PYTHON_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute Python code in a persistent Jupyter kernel. "
                "Use this for calculations, symbolic manipulation, numerical checks, "
                "and small experiments. Always print important final values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    }
]


class PythonSandbox:
    def __init__(
        self,
        startup_timeout: float = 30.0,
        exec_timeout: float = 30.0,
        preload: str = "minimal",
    ):
        self._startup_timeout = startup_timeout
        self._exec_timeout = exec_timeout
        self._preload = preload

        self._client = None
        self._km = None
        self._connection_file: Optional[str] = None
        self._owns_kernel = False

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
        self._connection_file = os.path.join(
            tempfile.gettempdir(),
            f"python-sandbox-{uuid.uuid4().hex}.json",
        )

        self._km = KernelManager(connection_file=self._connection_file)
        self._km.start_kernel(env=self._env, extra_arguments=self._extra_args)

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._startup_timeout)
        self._owns_kernel = True

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
        elif self._preload == "none":
            pass
        else:
            raise ValueError(f"Unknown preload={self._preload!r}")

    def restart(self) -> None:
        try:
            self.close()
        except Exception:
            pass

        self._client = None
        self._km = None
        self._owns_kernel = False

        self._start_kernel()
        self._preload_modules()

    def _format_error(self, traceback: List[str]) -> str:
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

        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )

        stdout_parts: List[str] = []
        stderr_parts: List[str] = []

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > effective_timeout:
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
                traceback_list = content.get("traceback", [])
                stderr_parts.append(self._format_error(traceback_list))

            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr

        return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

    def close(self) -> None:
        km = self._km
        client = self._client

        self._client = None
        self._km = None
        self._owns_kernel = False

        with contextlib.suppress(Exception):
            if client:
                client.stop_channels()

        if km is not None:
            with contextlib.suppress(Exception):
                km.shutdown_kernel(now=True)

            with contextlib.suppress(Exception):
                km.kill_kernel(now=True)

            with contextlib.suppress(Exception):
                km.cleanup_resources()

        with contextlib.suppress(Exception):
            if self._connection_file:
                os.remove(self._connection_file)
                self._connection_file = None

    def __del__(self) -> None:
        self.close()


class MathPythonBackend(ToolBackend):
    def __init__(
        self,
        startup_timeout: float = 30.0,
        exec_timeout: float = 30.0,
        preload: str = "minimal",
        auto_print_last_expr: bool = True,
    ):
        self.startup_timeout = startup_timeout
        self.exec_timeout = exec_timeout
        self.preload = preload
        self.auto_print_last_expr = auto_print_last_expr

        self._sandbox: Optional[PythonSandbox] = None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return PYTHON_TOOL_DEFINITIONS

    def setup(self, task_config: Dict[str, Any]) -> bool:
        preload = task_config.get("python_preload", self.preload)
        startup_timeout = float(task_config.get("python_startup_timeout", self.startup_timeout))
        exec_timeout = float(task_config.get("python_exec_timeout", self.exec_timeout))

        with self._init_lock:
            if self._sandbox is None:
                self._sandbox = PythonSandbox(
                    startup_timeout=startup_timeout,
                    exec_timeout=exec_timeout,
                    preload=preload,
                )
        return True

    def execute(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        started = time.perf_counter()

        if tool_name != "python":
            latency_ms = (time.perf_counter() - started) * 1000.0
            return ToolResult(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                output=f"[ERROR] Unknown tool: {tool_name}",
                latency_ms=latency_ms,
                success=False,
            )

        code = arguments.get("code", "")
        if not isinstance(code, str) or not code.strip():
            latency_ms = (time.perf_counter() - started) * 1000.0
            return ToolResult(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                output="[ERROR] Tool argument 'code' must be a non-empty string.",
                latency_ms=latency_ms,
                success=False,
            )

        self._ensure_session()

        final_code = self._ensure_last_print(code) if self.auto_print_last_expr else code

        with self._execution_lock:
            try:
                output = self._sandbox.execute(final_code)
                success = not output.startswith("[ERROR]")
            except Exception as exc:
                output = f"[ERROR] {type(exc).__name__}: {exc}"
                success = False

        latency_ms = (time.perf_counter() - started) * 1000.0
        return ToolResult(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            output=output,
            latency_ms=latency_ms,
            success=success,
        )

    def teardown(self) -> None:
        if self._sandbox is not None:
            self._sandbox.close()
            self._sandbox = None

    def get_patch(self) -> str:
        return ""

    def cleanup(self) -> None:
        self.teardown()

    def _ensure_session(self) -> None:
        if self._sandbox is None:
            with self._init_lock:
                if self._sandbox is None:
                    self._sandbox = PythonSandbox(
                        startup_timeout=self.startup_timeout,
                        exec_timeout=self.exec_timeout,
                        preload=self.preload,
                    )

    def _ensure_last_print(self, code: str) -> str:
        stripped = code.rstrip()
        if not stripped:
            return code

        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            return code

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    return code

        if not tree.body:
            return code

        last_stmt = tree.body[-1]

        defined_names = set()
        for stmt in tree.body[:-1]:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        for name in self._extract_assigned_names(target):
                            defined_names.add(name)
                elif isinstance(node, ast.AnnAssign):
                    for name in self._extract_assigned_names(node.target):
                        defined_names.add(name)
                elif isinstance(node, ast.AugAssign):
                    for name in self._extract_assigned_names(node.target):
                        defined_names.add(name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined_names.add(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name.split(".")[0])

        if isinstance(last_stmt, ast.Expr) and isinstance(last_stmt.value, ast.Name):
            if last_stmt.value.id in defined_names:
                lines = stripped.split("\n")
                indent = self._leading_whitespace(lines[-1])
                lines[-1] = f"{indent}print({last_stmt.value.id})"
                return "\n".join(lines)

        if isinstance(last_stmt, ast.Expr):
            expr_src = ast.get_source_segment(stripped, last_stmt.value)
            if expr_src:
                lines = stripped.split("\n")
                indent = self._leading_whitespace(lines[-1])
                lines[-1] = f"{indent}print({expr_src})"
                return "\n".join(lines)

        return code

    @staticmethod
    def _extract_assigned_names(target: ast.AST) -> List[str]:
        names: List[str] = []
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                names.extend(MathPythonBackend._extract_assigned_names(elt))
        return names

    @staticmethod
    def _leading_whitespace(s: str) -> str:
        return s[: len(s) - len(s.lstrip())]
