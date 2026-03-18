from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvalResult:
    passed: bool
    score: float
    details: Dict[str, Any]


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, task_config: Dict[str, Any], backend: Any) -> EvalResult: ...
