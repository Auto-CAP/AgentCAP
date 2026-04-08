from typing import Any, Dict, List

from agent_cap.core.tool_backend import ToolBackend, ToolResult
from agent_cap.backends.tool_executor import TOOL_DEFINITIONS


class SWEBenchBackend(ToolBackend):
    def __init__(self, runtime: str = "modal", shell_timeout: int = 30):
        self.runtime = runtime
        self.shell_timeout = shell_timeout
        self._workspace: Any = None
        self._executor: Any = None

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return TOOL_DEFINITIONS

    def setup(self, task_config: Dict[str, Any]) -> bool:
        if self.runtime == "modal":
            from agent_cap.backends.modal_env import ModalWorkspace

            self._workspace = ModalWorkspace(task_config)
        else:
            from agent_cap.backends.docker_env import DockerWorkspace

            self._workspace = DockerWorkspace(task_config)

        if not self._workspace.setup():
            return False

        from agent_cap.backends.tool_executor import ToolExecutor

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
        if self._executor is None:
            return ToolResult(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                output="ERROR: backend not set up",
                latency_ms=0.0,
                success=False,
            )
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
            exec_fn = getattr(self._workspace, "_exec", None)
            if callable(exec_fn):
                exec_fn("git checkout . 2>/dev/null")
            else:
                docker_exec_fn = getattr(self._workspace, "_docker_exec", None)
                if callable(docker_exec_fn):
                    docker_exec_fn("git checkout .", timeout=10)

    def get_patch(self) -> str:
        if self._workspace:
            return self._workspace.get_git_diff()
        return ""

    def cleanup(self) -> None:
        if self._workspace:
            self._workspace.cleanup()
