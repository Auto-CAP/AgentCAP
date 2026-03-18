from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ToolResult:
    tool_name: str
    tool_call_id: str
    output: str
    latency_ms: float
    success: bool


class ToolBackend(ABC):
    @abstractmethod
    def get_tool_definitions(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def execute(
        self, tool_name: str, tool_call_id: str, arguments: Dict[str, Any]
    ) -> ToolResult: ...

    @abstractmethod
    def setup(self, task_config: Dict[str, Any]) -> bool: ...

    @abstractmethod
    def teardown(self) -> None: ...

    @abstractmethod
    def get_patch(self) -> str: ...
