"""Tool-call simulation for single-agent benchmarking.

Provides a default set of OpenAI function-calling tool definitions that
simulate an agentic SWE-Bench workflow (file read, file write, shell
execution, search).  These are passed to the model so it produces
``tool_calls`` in its response, allowing measurement of tool-call-related
metrics even when no real execution backend is present.
"""

import time
import random
from typing import Any, Dict, List, Optional


# Default tool definitions in OpenAI function-calling format
DEFAULT_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
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
                        "description": "Absolute file path to read.",
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
            "description": "Write content to a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file.",
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
            "description": "Execute a shell command and return its output.",
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
            "description": "Search for a pattern in the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex or string pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search within.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


def simulate_tool_execution(
    tool_name: str,
    arguments: Dict[str, Any],
    latency_range_ms: tuple = (10, 100),
) -> Dict[str, Any]:
    """Simulate execution of a tool call with artificial latency.

    This allows benchmarking the full agentic loop (model generates
    tool call -> we "execute" and return result -> model continues)
    without a real tool backend.

    Args:
        tool_name: Name of the tool to simulate.
        arguments: Tool arguments from the model.
        latency_range_ms: (min, max) artificial latency in ms.

    Returns:
        dict with ``result``, ``latency_ms``, ``tool_name``.
    """
    t0 = time.perf_counter()

    # Add artificial latency
    delay_ms = random.uniform(*latency_range_ms)
    time.sleep(delay_ms / 1000.0)

    # Produce a stub result
    result_text = _generate_stub_result(tool_name, arguments)

    t1 = time.perf_counter()
    actual_latency_ms = (t1 - t0) * 1000

    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result_text,
        "latency_ms": actual_latency_ms,
    }


def _generate_stub_result(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Generate a plausible stub result for the simulated tool."""
    if tool_name == "read_file":
        path = arguments.get("path", "unknown")
        return f"# Contents of {path}\nimport os\nimport sys\n\ndef main():\n    pass\n"
    if tool_name == "write_file":
        path = arguments.get("path", "unknown")
        return f"Successfully wrote to {path}"
    if tool_name == "run_shell":
        cmd = arguments.get("command", "")
        return f"$ {cmd}\nCommand executed successfully (exit code 0)"
    if tool_name == "search_code":
        pattern = arguments.get("pattern", "")
        return (
            f"Found 3 matches for '{pattern}':\n"
            "  src/main.py:42: matching line 1\n"
            "  src/utils.py:17: matching line 2\n"
            "  tests/test_main.py:8: matching line 3\n"
        )
    return f"Tool '{tool_name}' executed successfully."
