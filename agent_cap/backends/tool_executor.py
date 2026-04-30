"""Tool execution against a workspace (Docker / K8s / Modal / local).

Implements the EXACT same tools as official SWE-agent (used by SWE-bench Pro):
  - bash:              run arbitrary shell commands
  - str_replace_editor: view/create/str_replace/insert/undo_edit files
  - submit:            signal task completion (no-op, patch collected via git diff)

Reference: https://github.com/SWE-agent/SWE-agent  tools/edit_anthropic + tools/submit
Config:    https://github.com/scaleapi/SWE-bench_Pro-os  config/tool_use.yaml
"""

import base64
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Official SWE-agent tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    # 1. bash  (enable_bash_tool: true in tool_use.yaml)
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "runs the given command directly in bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    # 2. str_replace_editor  (tools/edit_anthropic bundle)
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": (
                "Custom editing tool for viewing, creating and editing files\n"
                "* State is persistent across command calls and discussions with the user\n"
                "* If `path` is a file, `view` displays the result of applying `cat -n`. "
                "If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n"
                "* The `create` command cannot be used if the specified `path` already exists as a file\n"
                "* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`\n"
                "* The `undo_edit` command will revert the last edit made to the file at `path`\n\n"
                "Notes for using the `str_replace` command:\n"
                "* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. "
                "Be mindful of whitespaces!\n"
                "* If the `old_str` parameter is not unique in the file, the replacement will not be performed. "
                "Make sure to include enough context in `old_str` to make it unique\n"
                "* The `new_str` parameter should contain the edited lines that should replace the `old_str`"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Required parameter of `create` command, with the content of the file to be created.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": (
                            "Optional parameter of `str_replace` command containing the new string "
                            "(if not given, no string will be added). "
                            "Required parameter of `insert` command containing the string to insert."
                        ),
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional parameter of `view` command when `path` points to a file. "
                            "If none is given, the full file is shown. If provided, the file will be shown "
                            "in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. "
                            "Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file."
                        ),
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
    # 3. submit  (tools/submit bundle)
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "submits the current file",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
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
        shell_timeout: int = 450,
        max_output_chars: int = 100_000,
        container_id: Optional[str] = None,
        modal_sandbox: Optional[Any] = None,
        exec_fn: Optional[Any] = None,
        write_file_fn: Optional[Any] = None,
    ) -> None:
        self.workspace = Path(workspace_dir).resolve()
        self.shell_timeout = shell_timeout
        self.max_output_chars = max_output_chars
        self.container_id = container_id
        self.modal_sandbox = modal_sandbox
        self.exec_fn = exec_fn
        self.write_file_fn = write_file_fn
        # undo_edit stack: path -> list of previous file contents
        self._undo_stack: Dict[str, List[str]] = {}
        # Persistent cwd tracking (simulates persistent bash session)
        self._cwd: str = str(self.workspace)
        # Shell init commands (e.g. conda activate for SWE-bench Lite)
        self._shell_init: str = ""

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
            output = output[: self.max_output_chars] + "\n<response clipped>"

        return ToolCallResult(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            output=output,
            latency_ms=latency_ms,
            success=success,
        )

    # Official SWE-agent MAX_RESPONSE_LEN for str_replace_editor
    MAX_RESPONSE_LEN = 16000

    @staticmethod
    def _maybe_truncate(content: str, max_len: int = 16000) -> str:
        """Truncate content matching official maybe_truncate()."""
        if len(content) <= max_len:
            return content
        return content[:max_len] + "<response clipped>"

    # ------------------------------------------------------------------ #
    #  Dispatch
    # ------------------------------------------------------------------ #

    def _dispatch(self, name: str, args: Dict[str, Any]) -> str:
        if name == "bash":
            return self._bash(args)
        if name == "str_replace_editor":
            return self._str_replace_editor(args)
        if name == "submit":
            # Return diff for review (matches official SWE-agent submit review)
            diff_proc = self._exec("git add -A && git diff --cached")
            diff = diff_proc.stdout.strip() if diff_proc and diff_proc.stdout else ""
            if diff:
                return f"Submission received. Here is your diff:\n\n{diff[:8000]}"
            return "Submission successful."
        raise ValueError(f"Unknown tool: {name}")

    # ------------------------------------------------------------------ #
    #  Low-level exec (Docker / K8s / Modal / local)
    # ------------------------------------------------------------------ #

    def _exec(self, cmd: str) -> subprocess.CompletedProcess:
        if self.exec_fn:
            result = self.exec_fn(cmd, timeout=self.shell_timeout)
            if result is None:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=1, stdout="", stderr="Command timed out"
                )
            return result
        if self.modal_sandbox:
            process = self.modal_sandbox.exec(
                "bash", "-c", f"cd {self.workspace} && {cmd}"
            )
            process.wait()
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        if self.container_id:
            return subprocess.run(
                [
                    "docker", "exec", "-w", str(self.workspace),
                    self.container_id, "bash", "-c", cmd,
                ],
                capture_output=True, text=True, timeout=self.shell_timeout,
            )
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=self.shell_timeout, cwd=str(self.workspace),
        )

    # ------------------------------------------------------------------ #
    #  bash tool
    # ------------------------------------------------------------------ #

    _CWD_MARKER = "__SWE_CWD__"

    def _bash(self, args: Dict[str, Any]) -> str:
        cmd = args.get("command", "")
        if not cmd:
            return "ERROR: no command provided"
        # Write command to a temp script file on the remote, then execute it.
        # This avoids issues with heredocs, quotes, and special chars in wrapping.
        marker = self._CWD_MARKER
        # Escape single quotes in cmd for the echo
        script = f"{self._shell_init}\ncd {self._cwd}\n{cmd}\n"
        b64_script = base64.b64encode(script.encode()).decode()
        wrapped = (
            f"echo {b64_script} | base64 -d > /tmp/_swe_cmd.sh && "
            f"bash /tmp/_swe_cmd.sh ; _ec=$?; "
            f"echo '{marker}'$(pwd)'{marker}'; "
            f"exit $_ec"
        )
        try:
            proc = self._exec(wrapped)
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.shell_timeout}s: {cmd}"
        stdout = proc.stdout or ""
        # Strip ANSI escape codes for cleaner output
        import re as _re
        stdout = _re.sub(r'\x1b\[[0-9;]*[mGKHF]', '', stdout)
        stderr = proc.stderr or ""
        stderr = _re.sub(r'\x1b\[[0-9;]*[mGKHF]', '', stderr)
        # Extract and update cwd from output
        if marker in stdout:
            before, _, after = stdout.partition(marker)
            new_cwd, _, remainder = after.partition(marker)
            new_cwd = new_cwd.strip()
            if new_cwd:
                self._cwd = new_cwd
            stdout = before.rstrip()
        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(stderr)
        if not parts:
            return "Your last command ran successfully and did not produce any output."
        if proc.returncode != 0:
            parts.append(f"[exit code {proc.returncode}]")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    #  str_replace_editor — runs official SWE-agent script in container
    # ------------------------------------------------------------------ #

    _EDITOR_SCRIPT = "/opt/swe_agent_tools/str_replace_editor"

    def _str_replace_editor(self, args: Dict[str, Any]) -> str:
        """Call the official SWE-agent str_replace_editor script in the container."""
        command = args.get("command", "")
        path = args.get("path", "")
        if not command or not path:
            return "ERROR: 'command' and 'path' are required."

        # Build CLI args matching the script's argparse
        cmd_parts = [
            f"PYTHONPATH=/opt/swe_agent_tools python3 {self._EDITOR_SCRIPT}",
            command,
            path,
        ]
        # For create/str_replace/insert, write args to temp files to avoid
        # kubectl exec URI length limits, then pass via CLI
        write_needed = {}
        if args.get("file_text") is not None:
            write_needed["file_text"] = args["file_text"]
        if args.get("old_str") is not None:
            write_needed["old_str"] = args["old_str"]
        if args.get("new_str") is not None:
            write_needed["new_str"] = args["new_str"]

        # Write large string args via write_file_fn (kubectl cp) to avoid URI limits
        for arg_name, arg_val in write_needed.items():
            tmp_path = f"/tmp/_swe_arg_{arg_name}"
            if self.write_file_fn:
                self.write_file_fn(tmp_path, arg_val)
            else:
                # Docker/local: direct write via exec
                b64 = base64.b64encode(arg_val.encode()).decode()
                self._exec(
                    f'python3 -c "import base64,sys; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read()))" '
                    f"<<'__B64EOF__' > {tmp_path}\n{b64}\n__B64EOF__"
                )
            cmd_parts.append(f"--{arg_name} \"$(cat {tmp_path})\"")

        if args.get("view_range") is not None:
            vr = args["view_range"]
            if isinstance(vr, list) and len(vr) == 2:
                cmd_parts.append(f"--view_range {vr[0]} {vr[1]}")
        if args.get("insert_line") is not None:
            cmd_parts.append(f"--insert_line {args['insert_line']}")

        full_cmd = " ".join(cmd_parts)
        proc = self._exec(full_cmd)
        if proc.returncode == 0:
            return proc.stdout if proc.stdout else "Your command ran successfully and did not produce any output."
        # Script uses sys.exit(N) for errors — stdout has the error message
        output = proc.stdout.strip() if proc.stdout else ""
        if proc.stderr:
            output += f"\n{proc.stderr.strip()}" if output else proc.stderr.strip()
        return output if output else f"Command failed with exit code {proc.returncode}"
