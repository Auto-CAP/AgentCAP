from typing import Any, Dict

from agent_cap.core.evaluator import Evaluator, EvalResult


class GTFAEvaluator(Evaluator):
    def evaluate(self, task_config: Dict[str, Any], backend: Any) -> EvalResult:
        return EvalResult(
            passed=False,
            score=0.0,
            details={"note": "GTFA evaluation requires LLM-as-judge, run separately"},
        )
