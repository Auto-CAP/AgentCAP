"""Tool execution against a local workspace directory.

All tool calls run as local subprocesses in the workspace directory
which has been set up by LocalWorkspace (git clone + conda env + deps).
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to repo root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parent dirs as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to repo root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command in the repo directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Grep for a pattern in the repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Subdirectory to search (default: whole repo).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


@dataclass
class ToolCallResult:
    tool_name: str
    tool_call_id: str
    arguments: Dict[str, Any]
    output: str
    latency_ms: float
    success: bool


class ToolExecutor:
    def __init__(
        self,
        workspace_dir: str | Path = "/app",
        shell_timeout: int = 30,
        max_output_chars: int = 16_000,
        container_id: Optional[str] = None,
        modal_sandbox: Optional[Any] = None,
        exec_fn: Optional[Any] = None,
    ) -> None:
        self.workspace = Path(workspace_dir).resolve()
        self.shell_timeout = shell_timeout
        self.max_output_chars = max_output_chars
        self.container_id = container_id
        self.modal_sandbox = modal_sandbox
        self.exec_fn = exec_fn

    def execute(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: Dict[str, Any],
    ) -> ToolCallResult:
        t0 = time.perf_counter()
        try:
            output = self._dispatch(tool_name, arguments)
            success = True
        except Exception as exc:
            output = f"ERROR: {type(exc).__name__}: {exc}"
            success = False
        latency_ms = (time.perf_counter() - t0) * 1000

        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + "\n... (truncated)"

        return ToolCallResult(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            output=output,
            latency_ms=latency_ms,
            success=success,
        )

    def _dispatch(self, name: str, args: Dict[str, Any]) -> str:
        if name == "read_file":
            return self._read_file(args)
        if name == "write_file":
            return self._write_file(args)
        if name == "run_shell":
            return self._run_shell(args)
        if name == "search_code":
            return self._search_code(args)
        raise ValueError(f"Unknown tool: {name}")

    def _exec(self, cmd: str) -> subprocess.CompletedProcess:
        if self.exec_fn:
            return self.exec_fn(cmd, timeout=self.shell_timeout)
        if self.modal_sandbox:
            process = self.modal_sandbox.exec(
                "bash", "-c", f"cd {self.workspace} && {cmd}"
            )
            process.wait()
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
            return result
        if self.container_id:
            return subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    str(self.workspace),
                    self.container_id,
                    "bash",
                    "-c",
                    cmd,
                ],
                capture_output=True,
                text=True,
                timeout=self.shell_timeout,
            )
        return subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.shell_timeout,
            cwd=str(self.workspace),
        )

    def _read_file(self, args: Dict[str, Any]) -> str:
        path = args["path"]
        proc = self._exec(f"cat {path!r}")
        if proc.returncode != 0:
            return f"File not found: {path}\n{proc.stderr}"
        return proc.stdout

    def _write_file(self, args: Dict[str, Any]) -> str:
        path = args["path"]
        content = args["content"]
        escaped = content.replace("\\", "\\\\").replace("'", "'\\''")
        proc = self._exec(
            f"mkdir -p $(dirname {path!r}) && printf '%s' '{escaped}' > {path!r}"
        )
        if proc.returncode != 0:
            return f"Write failed: {proc.stderr}"
        return f"Wrote {len(content)} chars to {path}"

    def _run_shell(self, args: Dict[str, Any]) -> str:
        cmd = args["command"]
        try:
            proc = self._exec(cmd)
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.shell_timeout}s: {cmd}"
        parts = []
        if proc.stdout:
            parts.append(proc.stdout)
        if proc.stderr:
            parts.append(f"[stderr]\n{proc.stderr}")
        parts.append(f"[exit code {proc.returncode}]")
        return "\n".join(parts)

    def _search_code(self, args: Dict[str, Any]) -> str:
        pattern = args["pattern"]
        sub_path = args.get("path", ".") or "."
        proc = self._exec(f"grep -rn -E {pattern!r} {sub_path!r} || true")
        if proc.stdout:
            return proc.stdout
        return f"No matches found for: {pattern}"
        return f"No matches found for: {pattern}"
