"""Tool execution inside Docker containers for agentic SWE-bench.

All tool calls run via ``docker exec`` inside a pre-built SWE-bench
container that has the repo checked out and dependencies installed.
"""

import json
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
    """Execute tool calls inside a Docker container via ``docker exec``.

    The container must already be running with the repo mounted/cloned.
    """

    def __init__(
        self,
        container_id: str,
        workdir: str = "/testbed",
        shell_timeout: int = 30,
        max_output_chars: int = 16_000,
    ) -> None:
        self.container_id = container_id
        self.workdir = workdir
        self.shell_timeout = shell_timeout
        self.max_output_chars = max_output_chars

    def _docker_exec(
        self, cmd: str, timeout: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "docker",
                "exec",
                "-w",
                self.workdir,
                self.container_id,
                "bash",
                "-c",
                cmd,
            ],
            capture_output=True,
            text=True,
            timeout=timeout or self.shell_timeout,
        )

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

    def _read_file(self, args: Dict[str, Any]) -> str:
        path = args["path"]
        proc = self._docker_exec(f"cat {path!r}")
        if proc.returncode != 0:
            return f"File not found: {path}\n{proc.stderr}"
        return proc.stdout

    def _write_file(self, args: Dict[str, Any]) -> str:
        path = args["path"]
        content = args["content"]
        escaped = content.replace("\\", "\\\\").replace("'", "'\\''")
        proc = self._docker_exec(
            f"mkdir -p $(dirname {path!r}) && printf '%s' '{escaped}' > {path!r}"
        )
        if proc.returncode != 0:
            return f"Write failed: {proc.stderr}"
        return f"Wrote {len(content)} chars to {path}"

    def _run_shell(self, args: Dict[str, Any]) -> str:
        cmd = args["command"]
        try:
            proc = self._docker_exec(cmd)
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
        proc = self._docker_exec(f"grep -rn -E {pattern!r} {sub_path!r} || true")
        if proc.stdout:
            return proc.stdout
        return f"No matches found for: {pattern}"


def build_swebench_container(
    instance_id: str,
    repo: str,
    base_commit: str,
    test_patch: str = "",
) -> Optional[str]:
    """Build and start a SWE-bench Docker container for a task.

    Uses swebench.harness utilities to build the environment image,
    then starts a container with the repo at base_commit.

    Returns container_id or None on failure.
    """
    try:
        proc = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                f"agentcap-{instance_id[:40]}",
                "-w",
                "/testbed",
                f"sweb.eval.x86_64.{instance_id}:latest",
                "sleep",
                "infinity",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            container_id = proc.stdout.strip()
            if test_patch:
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        container_id,
                        "bash",
                        "-c",
                        f"cd /testbed && echo {json.dumps(test_patch)} | python3 -c \"import sys,json; open('/tmp/test.patch','w').write(json.loads(sys.stdin.read()))\" && git apply /tmp/test.patch",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            return container_id
    except Exception:
        pass
    return None


def stop_container(container_id: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_id],
        capture_output=True,
        timeout=15,
    )


def get_git_diff(container_id: str, workdir: str = "/testbed") -> str:
    try:
        proc = subprocess.run(
            ["docker", "exec", "-w", workdir, container_id, "git", "diff"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return ""


def run_tests_in_container(
    container_id: str,
    fail_to_pass: str,
    workdir: str = "/testbed",
    timeout: int = 300,
) -> Dict[str, Any]:
    if not fail_to_pass:
        return {"passed": False, "reason": "no tests defined"}

    try:
        tests = (
            json.loads(fail_to_pass) if isinstance(fail_to_pass, str) else fail_to_pass
        )
    except json.JSONDecodeError:
        tests = [fail_to_pass]

    passed_count = 0
    total = len(tests)
    details = []

    for test_spec in tests:
        test_file = test_spec.split("::")[0].split(" | ")[0].strip()
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    workdir,
                    container_id,
                    "python",
                    "-m",
                    "pytest",
                    test_file,
                    "-x",
                    "--tb=short",
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            ok = proc.returncode == 0
            if ok:
                passed_count += 1
            details.append(
                {
                    "test": test_spec[:100],
                    "passed": ok,
                    "output": (proc.stdout + proc.stderr)[-500:],
                }
            )
        except subprocess.TimeoutExpired:
            details.append(
                {"test": test_spec[:100], "passed": False, "output": "timeout"}
            )
        except Exception as exc:
            details.append(
                {"test": test_spec[:100], "passed": False, "output": str(exc)}
            )

    return {
        "passed": passed_count == total,
        "passed_count": passed_count,
        "total": total,
        "details": details,
    }
