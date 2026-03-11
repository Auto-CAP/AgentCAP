"""Quality evaluator for model outputs.

Supports evaluation strategies:
- code_exec: Extract Python code, run with test assertions
- numerical: Extract final number, compare to expected answer
- keyword: Check for required keywords/concepts in response
- humaneval: Execute generated function against HumanEval tests
- multiple_choice: Extract answer option letter and compare
"""

from dataclasses import dataclass, field
import re
import subprocess
from typing import Any, Dict, List, Optional


@dataclass
class EvalConfig:
    type: str
    test_code: str = ""
    entry_point: str = ""
    expected: float = 0.0
    expected_answer: str = ""
    tolerance: float = 0.01
    required_keywords: List[str] = field(default_factory=list)
    min_keywords: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalConfig":
        return cls(
            type=str(data.get("type", "")).strip(),
            test_code=str(data.get("test_code", "")),
            entry_point=str(data.get("entry_point", "")),
            expected=float(data.get("expected", 0.0)),
            expected_answer=str(data.get("expected_answer", "")),
            tolerance=float(data.get("tolerance", 0.01)),
            required_keywords=[str(k) for k in data.get("required_keywords", [])],
            min_keywords=int(data.get("min_keywords", 0)),
        )


@dataclass
class EvalResult:
    task_success: bool
    quality_score: float
    explanation: str


def evaluate(output_text: str, eval_config: EvalConfig) -> EvalResult:
    clean_text = _strip_think_tags(output_text)

    if eval_config.type == "code_exec":
        return _eval_code_exec(clean_text, eval_config)
    if eval_config.type == "numerical":
        return _eval_numerical(clean_text, eval_config)
    if eval_config.type == "keyword":
        return _eval_keyword(clean_text, eval_config)
    if eval_config.type == "humaneval":
        return _eval_humaneval(clean_text, eval_config)
    if eval_config.type == "multiple_choice":
        return _eval_multiple_choice(clean_text, eval_config)
    if eval_config.type == "bigcodebench":
        return _eval_bigcodebench(clean_text, eval_config)

    return EvalResult(False, 0.0, f"Unknown eval type: {eval_config.type}")


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def _eval_code_exec(text: str, config: EvalConfig) -> EvalResult:
    code_blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if not code_blocks:
        return EvalResult(False, 0.0, "No Python code block found")

    combined_code = "\n\n".join([*code_blocks, config.test_code])

    try:
        proc = subprocess.run(
            ["python3", "-c", combined_code],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return EvalResult(False, 0.5, "Code execution timed out")

    if proc.returncode == 0:
        return EvalResult(True, 5.0, "All test assertions passed")

    stderr = (proc.stderr or "").strip()
    if "AssertionError" in stderr:
        return EvalResult(False, 1.0, f"Assertion failed: {stderr}")
    if "SyntaxError" in stderr:
        return EvalResult(False, 0.0, f"Syntax error: {stderr}")

    return EvalResult(False, 0.5, f"Code execution failed: {stderr}")


def _eval_numerical(text: str, config: EvalConfig) -> EvalResult:
    found = _extract_numerical_answer(text)
    if found is None:
        return EvalResult(False, 0.0, "No numerical answer found")

    diff = abs(found - config.expected)
    if diff <= config.tolerance:
        return EvalResult(True, 5.0, f"Correct: {found}")
    if diff <= 10 * config.tolerance:
        return EvalResult(False, 3.0, f"Close: {found} vs {config.expected}")
    return EvalResult(False, 0.0, f"Wrong: {found} vs {config.expected}")


def _extract_numerical_answer(text: str) -> Optional[float]:
    m = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", text)
    if m:
        return float(m.group(1).replace(",", ""))

    m = re.search(r"\\boxed\{([-+]?[\d,]*\.?\d+)\}", text)
    if m:
        return float(m.group(1).replace(",", ""))

    m = re.search(r"(?:the\s+)?answer\s+is[:\s]*([-+]?[\d,]*\.?\d+)", text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))

    matches = re.findall(r"[-+]?(?:\d+,)*\d*\.?\d+", text)
    if matches:
        return float(matches[-1].replace(",", ""))

    return None


def _eval_keyword(text: str, config: EvalConfig) -> EvalResult:
    normalized_text = text.lower()
    required = [k.lower() for k in config.required_keywords]

    if not required:
        return EvalResult(True, 5.0, "No required keywords specified")

    found = [k for k in required if k in normalized_text]
    missing = [k for k in required if k not in normalized_text]
    found_count = len(found)
    min_needed = config.min_keywords if config.min_keywords > 0 else len(required)

    score = (found_count / len(required)) * 5.0
    task_success = found_count >= min_needed
    explanation = (
        f"Found {found_count}/{len(required)} keywords "
        f"(need {min_needed}). Found={found}, Missing={missing}"
    )
    return EvalResult(task_success, score, explanation)


def _eval_humaneval(text: str, config: EvalConfig) -> EvalResult:
    code_blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if not code_blocks:
        candidate_code = text.strip()
    else:
        candidate_code = "\n\n".join(code_blocks)

    entry_point = config.entry_point
    test_code = config.test_code
    full_script = f"{candidate_code}\n\n{test_code}\n\ncheck({entry_point})\n"

    try:
        proc = subprocess.run(
            ["python3", "-c", full_script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return EvalResult(False, 0.5, "Code execution timed out")

    if proc.returncode == 0:
        return EvalResult(True, 5.0, "All HumanEval tests passed")

    stderr = (proc.stderr or "").strip()
    if "AssertionError" in stderr:
        return EvalResult(False, 1.0, f"Test failed: {stderr[:200]}")
    if "SyntaxError" in stderr:
        return EvalResult(False, 0.0, f"Syntax error: {stderr[:200]}")
    return EvalResult(False, 0.5, f"Execution failed: {stderr[:200]}")


def _eval_bigcodebench(text: str, config: EvalConfig) -> EvalResult:
    code_blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if not code_blocks:
        candidate_code = text.strip()
    else:
        candidate_code = "\n\n".join(code_blocks)

    code_prompt_parts = config.test_code.split("|||TESTS|||")
    if len(code_prompt_parts) == 2:
        test_code = code_prompt_parts[1]
    else:
        test_code = config.test_code

    full_script = (
        f"{candidate_code}\n\n"
        f"{test_code}\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main(exit=False, verbosity=0)\n"
    )

    try:
        proc = subprocess.run(
            ["python3", "-c", full_script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return EvalResult(False, 0.5, "Code execution timed out (30s)")

    stderr = (proc.stderr or "").strip()

    ok_match = re.search(r"Ran (\d+) tests? in [\d.]+s\s*\n\s*\n\s*OK", stderr)
    if ok_match:
        total = int(ok_match.group(1))
        return EvalResult(True, 5.0, f"All {total} BigCodeBench tests passed")

    failed_match = re.search(
        r"Ran (\d+) tests? in [\d.]+s\s*\n\s*\n\s*FAILED\s*\(([^)]*)\)", stderr
    )
    if failed_match:
        total = int(failed_match.group(1))
        detail = failed_match.group(2)
        failures = 0
        errors = 0
        f_match = re.search(r"failures=(\d+)", detail)
        e_match = re.search(r"errors=(\d+)", detail)
        if f_match:
            failures = int(f_match.group(1))
        if e_match:
            errors = int(e_match.group(1))
        passed = total - failures - errors
        score = (passed / total) * 5.0 if total > 0 else 0.0
        return EvalResult(False, score, f"Tests: {passed}/{total} passed. {stderr[-300:]}")

    if "SyntaxError" in stderr:
        return EvalResult(False, 0.0, f"Syntax error: {stderr[:300]}")
    if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        return EvalResult(False, 0.0, f"Import error: {stderr[:300]}")

    if proc.returncode == 0:
        return EvalResult(True, 5.0, "All BigCodeBench tests passed")

    return EvalResult(False, 0.5, f"Execution failed: {stderr[:300]}")


def _eval_multiple_choice(text: str, config: EvalConfig) -> EvalResult:
    expected = config.expected_answer.strip().upper()
    extracted = _extract_multiple_choice_letter(text)
    if extracted is None:
        return EvalResult(False, 0.0, "Could not extract answer letter")

    if extracted.upper() == expected:
        return EvalResult(True, 5.0, f"Correct: {extracted}")
    return EvalResult(False, 0.0, f"Wrong: {extracted} vs {expected}")


def _extract_multiple_choice_letter(text: str) -> Optional[str]:
    matches = re.findall(r"(?:the\s+answer\s+is|answer:)\s*([A-J])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    matches = re.findall(r"\b([A-J])\)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    matches = re.findall(r"\\boxed\{\s*([A-J])\s*\}", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    matches = re.findall(r"^\s*([A-J])\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    matches = re.findall(r"\b([A-J])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return None
