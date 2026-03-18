import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_cap.core.tool_backend import ToolBackend, ToolResult
from agent_cap.single_agent.tool_executor import TOOL_DEFINITIONS


class SWEBenchBackend(ToolBackend):
    def __init__(self, runtime: str = "modal", shell_timeout: int = 30):
        self.runtime = runtime
        self.shell_timeout = shell_timeout
        self._workspace = None
        self._executor = None

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return TOOL_DEFINITIONS

    def setup(self, task_config: Dict[str, Any]) -> bool:
        if self.runtime == "modal":
            from agent_cap.single_agent.modal_env import ModalWorkspace

            self._workspace = ModalWorkspace(task_config)
        else:
            from agent_cap.single_agent.docker_env import DockerWorkspace

            self._workspace = DockerWorkspace(task_config)

        if not self._workspace.setup():
            return False

        from agent_cap.single_agent.tool_executor import ToolExecutor

        modal_sb = getattr(self._workspace, "_sandbox", None)
        container_id = getattr(self._workspace, "container_id", None)
        self._executor = ToolExecutor(
            workspace_dir=self._workspace.workspace,
            shell_timeout=self.shell_timeout,
            container_id=container_id,
            modal_sandbox=modal_sb,
        )
        return True

    def execute(
        self, tool_name: str, tool_call_id: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        result = self._executor.execute(tool_name, tool_call_id, arguments)
        return ToolResult(
            tool_name=result.tool_name,
            tool_call_id=result.tool_call_id,
            output=result.output,
            latency_ms=result.latency_ms,
            success=result.success,
        )

    def teardown(self) -> None:
        if self._workspace:
            if hasattr(self._workspace, "_exec"):
                self._workspace._exec("git checkout . 2>/dev/null")
            elif hasattr(self._workspace, "_docker_exec"):
                self._workspace._docker_exec("git checkout .", timeout=10)

    def get_patch(self) -> str:
        if self._workspace:
            return self._workspace.get_git_diff()
        return ""

    def cleanup(self) -> None:
        if self._workspace:
            self._workspace.cleanup()
