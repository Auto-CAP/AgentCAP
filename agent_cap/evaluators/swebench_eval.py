import json
from typing import Any, Dict

from agent_cap.core.evaluator import Evaluator, EvalResult


class SWEBenchEvaluator(Evaluator):
    def evaluate(self, task_config: Dict[str, Any], backend: Any) -> EvalResult:
        ws = backend._workspace
        if not ws:
            return EvalResult(
                passed=False, score=0.0, details={"error": "workspace not ready"}
            )

        fail_to_pass = task_config.get(
            "FAIL_TO_PASS", task_config.get("fail_to_pass", "")
        )
        if not fail_to_pass:
            return EvalResult(
                passed=False, score=0.0, details={"error": "no tests defined"}
            )

        instance_id = task_config.get("instance_id", "")
        test_result = ws.run_tests()

        return EvalResult(
            passed=test_result.get("passed", False),
            score=1.0 if test_result.get("passed") else 0.0,
            details=test_result,
        )
