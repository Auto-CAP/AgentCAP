"""Real tool execution for single-agent benchmarking.

Executes tool calls against an actual workspace directory:
- read_file:    reads real files from disk
- write_file:   writes real files to disk
- run_shell:    runs real subprocess commands
- search_code:  runs real grep over the workspace

All operations are sandboxed to ``workspace_dir``.  Each call is timed
with ``perf_counter`` so that tool-call latency is measured accurately.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
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
                        "description": "File path (relative to workspace root).",
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
                        "description": "File path (relative to workspace root).",
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
            "description": "Execute a shell command in the workspace directory.",
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
            "description": "Grep for a pattern in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Subdirectory to search (default: whole workspace).",
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
    """Execute tool calls against a real filesystem workspace.

    Args:
        workspace_dir: Root directory for all file operations.
        shell_timeout: Max seconds for ``run_shell`` commands.
        max_output_chars: Truncate tool output beyond this length.
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        shell_timeout: int = 30,
        max_output_chars: int = 16_000,
    ) -> None:
        self.workspace = Path(workspace_dir).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.shell_timeout = shell_timeout
        self.max_output_chars = max_output_chars

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

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        target = (self.workspace / rel_path).resolve()
        if not str(target).startswith(str(self.workspace)):
            raise PermissionError(f"Path escapes workspace: {rel_path}")
        return target

    def _read_file(self, args: Dict[str, Any]) -> str:
        path = self._resolve(args["path"])
        if not path.is_file():
            return f"File not found: {args['path']}"
        return path.read_text(encoding="utf-8", errors="replace")

    def _write_file(self, args: Dict[str, Any]) -> str:
        path = self._resolve(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"Wrote {len(args['content'])} chars to {args['path']}"

    def _run_shell(self, args: Dict[str, Any]) -> str:
        cmd = args["command"]
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.shell_timeout,
                cwd=str(self.workspace),
            )
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
        sub_path = args.get("path", ".")
        search_dir = self._resolve(sub_path)
        if not search_dir.is_dir():
            return f"Directory not found: {sub_path}"

        try:
            proc = subprocess.run(
                [
                    "grep",
                    "-rn",
                    "--include=*.py",
                    "--include=*.js",
                    "--include=*.ts",
                    "--include=*.java",
                    "--include=*.c",
                    "--include=*.cpp",
                    "--include=*.h",
                    "--include=*.go",
                    "--include=*.rs",
                    "--include=*.rb",
                    "--include=*.txt",
                    "--include=*.md",
                    "--include=*.yaml",
                    "--include=*.yml",
                    "--include=*.json",
                    "--include=*.xml",
                    "--include=*.html",
                    "--include=*.css",
                    "--include=*.sh",
                    "-E",
                    pattern,
                    str(search_dir),
                ],
                capture_output=True,
                text=True,
                timeout=self.shell_timeout,
                cwd=str(self.workspace),
            )
        except subprocess.TimeoutExpired:
            return f"Search timed out after {self.shell_timeout}s"

        if proc.returncode == 1:
            return f"No matches found for: {pattern}"
        if proc.stdout:
            return proc.stdout
        return f"No matches found for: {pattern}"
