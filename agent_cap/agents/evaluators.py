"""Pluggable evaluator registry.

Built-ins are registered lazily so they don't pull heavy deps when unused.

To add a custom evaluator:

    from agent_cap.agents import register_evaluator, EvalResult

    @register_evaluator("exact")
    class ExactMatchEvaluator:
        def evaluate(self, task_meta, output_text):
            gold = task_meta.get("answer", "")
            ok = output_text.strip() == gold.strip()
            return EvalResult(passed=ok, score=1.0 if ok else 0.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Type, TypeVar


@dataclass
class EvalResult:
    passed: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    def evaluate(self, task_meta: Dict[str, Any], output_text: str) -> EvalResult: ...


_T = TypeVar("_T")
_EVALUATORS: Dict[str, Type] = {}


def register_evaluator(name: str) -> Callable[[Type[_T]], Type[_T]]:
    key = str(name).strip()
    if not key:
        raise ValueError("evaluator name must be non-empty")

    def deco(cls: Type[_T]) -> Type[_T]:
        _EVALUATORS[key] = cls
        return cls

    return deco


def get_evaluator(name: str, /, **kwargs: Any) -> Optional[object]:
    if name not in _EVALUATORS:
        return None
    return _EVALUATORS[name](**kwargs)


def list_evaluators() -> list:
    return sorted(_EVALUATORS.keys())


@register_evaluator("none")
class NullEvaluator:
    def __init__(self, **_: Any) -> None:
        pass

    def evaluate(self, task_meta: Dict[str, Any], output_text: str) -> EvalResult:
        return EvalResult(passed=False, score=0.0, details={"evaluator": "none"})


@register_evaluator("llm-judge")
@register_evaluator("judge")
class LLMJudgeEvaluator:
    """Generic LLM-as-a-judge. Plugs any OpenAI-compatible endpoint as judge.

    Config (passed via CLI `--judge k=v,k=v` or YAML `judge:`):
      name, base_url, api_key, model_path (alias of name)
      temperature, max_tokens, timeout_s
      system_prompt, user_template
        Placeholders in user_template: {predicted} {gold} {output} {question}
      decision_field: JSON key to look for (default 'equivalent')
      score_field   : JSON key for numeric score (optional, e.g. 'score')

    Default judge prompt asks "are PREDICTED and EXPECTED mathematically
    equivalent? reply JSON {\"equivalent\": true|false}".

    Auto-extracts a `predicted` answer from output_text via boxed/final-answer
    scan (same heuristic as the IMO evaluator). Set `extract: raw` in the
    config to skip extraction and pass the full output to the judge.
    """

    DEFAULT_SYSTEM = (
        "You are a strict judge. Decide if the PREDICTED answer is correct "
        "given the EXPECTED answer. Reply with JSON only: "
        '{"equivalent": true} or {"equivalent": false}.'
    )
    DEFAULT_USER_TEMPLATE = "PREDICTED: {predicted}\nEXPECTED: {gold}"

    def __init__(
        self,
        name: str = "",
        base_url: str = "",
        api_key: str = "",
        model_path: str = "",
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout_s: float = 60.0,
        system_prompt: str = "",
        user_template: str = "",
        decision_field: str = "equivalent",
        score_field: str = "",
        extract: str = "answer",
        **_: Any,
    ) -> None:
        self.model = name or model_path
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.user_template = user_template or self.DEFAULT_USER_TEMPLATE
        self.decision_field = decision_field
        self.score_field = score_field
        self.extract = extract  # "answer" | "raw"

    def evaluate(self, task_meta: Dict[str, Any], output_text: str) -> EvalResult:
        if not self.model or not self.base_url:
            return EvalResult(
                passed=False, score=0.0,
                details={
                    "evaluator": "llm-judge",
                    "skipped": True,
                    "reason": "no judge model configured. "
                              "Pass --judge name=...,base_url=...,api_key=... "
                              "or evaluator config `judge:` in YAML.",
                },
            )

        gold = _pick_gold(task_meta)
        question = (
            str(task_meta.get("question") or task_meta.get("user_prompt") or "")
        )
        if self.extract == "raw":
            predicted = output_text.strip()
        else:
            predicted = _scan_for_imo_answer(output_text) or output_text.strip()

        prompt = self.user_template.format(
            predicted=predicted, gold=gold, output=output_text, question=question,
        )
        decision, score, raw, err = _call_openai_judge(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            decision_field=self.decision_field,
            score_field=self.score_field,
        )

        if err:
            return EvalResult(
                passed=False, score=0.0,
                details={"evaluator": "llm-judge", "error": err, "model": self.model},
            )

        passed = bool(decision) if decision is not None else False
        final_score = float(score) if score is not None else (1.0 if passed else 0.0)
        return EvalResult(
            passed=passed, score=final_score,
            details={
                "evaluator": "llm-judge",
                "model": self.model,
                "predicted": predicted, "gold": gold,
                "decision": decision, "raw_score": score,
                "raw": (raw or "")[:400],
            },
        )


@register_evaluator("gtfa")
class GTFAEvaluatorAdapter:
    """Wraps `agent_cap.evaluators.gtfa_eval.GTFAEvaluator`."""

    def __init__(self, **_: Any) -> None:
        from agent_cap.evaluators.gtfa_eval import GTFAEvaluator

        self._impl = GTFAEvaluator()

    def evaluate(self, task_meta: Dict[str, Any], output_text: str) -> EvalResult:
        claims = task_meta.get("gtfa_claims") or (task_meta.get("eval_config") or {}).get("gtfa_claims") or []
        if not output_text.strip() or not claims:
            return EvalResult(passed=False, score=0.0, details={"evaluator": "gtfa", "reason": "empty"})
        ev = self._impl.evaluate({"gtfa_claims": claims, "response": output_text}, None)
        details = getattr(ev, "details", {}) or {}
        if not isinstance(details, dict):
            details = {"details": details}
        details.setdefault("evaluator", "gtfa")
        return EvalResult(
            passed=bool(getattr(ev, "passed", False)),
            score=float(getattr(ev, "score", 0.0) or 0.0),
            details=details,
        )


@register_evaluator("imo-answerbench")
@register_evaluator("imo")
class IMOAnswerBenchEvaluator:
    """Two-stage evaluator for IMO AnswerBench-style tasks.

    1. Extract predicted answer via `\\boxed{...}` / "final answer is" / bold.
    2. Try `math_verify` symbolic equivalence (fast, deterministic).
    3. If symbolic check is False, ask an LLM judge for semantic equivalence:
       - If `base_url` and `api_key` are passed, call that OpenAI-compatible endpoint directly
         (e.g. Gemini: base_url=https://generativelanguage.googleapis.com/v1beta/openai).
       - Otherwise falls back to OpenRouter if OPENROUTER_API_KEY is set.

    Expects `task_meta` to carry the gold answer under `answer` / `expected` /
    `gold` / inside `eval_config`.
    """

    def __init__(
        self,
        judge_model: str = "openrouter/elephant-alpha",
        judge_timeout_s: float = 60.0,
        name: str = "",
        base_url: str = "",
        api_key: str = "",
        **_: Any,
    ) -> None:
        import os
        self.judge_model = name or judge_model
        self.judge_timeout_s = judge_timeout_s
        self.base_url = base_url
        # YAML loading does not expand env vars; expandvars here so both the
        # YAML path (e.g. "${GEMINI_API_KEY}") and the CLI path work correctly.
        self.api_key = os.path.expandvars(api_key) if api_key else api_key

    def evaluate(self, task_meta: Dict[str, Any], output_text: str) -> EvalResult:
        gold = _pick_gold(task_meta)
        if not output_text.strip():
            return EvalResult(
                passed=False, score=0.0,
                details={"evaluator": "imo-answerbench", "reason": "empty_output", "gold": gold},
            )

        predicted = _scan_for_imo_answer(output_text)
        if predicted is None or not gold:
            return EvalResult(
                passed=False, score=0.0,
                details={
                    "evaluator": "imo-answerbench",
                    "reason": "no_predicted_or_gold",
                    "predicted": predicted,
                    "gold": gold,
                },
            )

        symbolic_ok, symbolic_err = _math_verify_equivalent(predicted, gold)
        if symbolic_ok:
            return EvalResult(
                passed=True, score=1.0,
                details={
                    "evaluator": "imo-answerbench",
                    "method": "math_verify",
                    "predicted": predicted, "gold": gold,
                },
            )

        if self.base_url and self.api_key:
            judge_decision, judge_meta = self._configured_judge(predicted, gold)
        else:
            judge_decision, judge_meta = _openrouter_judge(
                predicted, gold, model=self.judge_model, timeout_s=self.judge_timeout_s,
            )

        if judge_decision is True:
            return EvalResult(
                passed=True, score=1.0,
                details={
                    "evaluator": "imo-answerbench",
                    "method": "llm_judge",
                    "predicted": predicted, "gold": gold,
                    "judge": judge_meta,
                },
            )

        return EvalResult(
            passed=False, score=0.0,
            details={
                "evaluator": "imo-answerbench",
                "method": "math_verify+llm_judge" if judge_meta else "math_verify",
                "predicted": predicted, "gold": gold,
                "math_verify_error": symbolic_err,
                "judge": judge_meta,
            },
        )

    def _configured_judge(self, predicted: str, gold: str):
        decision, _, raw, err = _call_openai_judge(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.judge_model,
            system_prompt=_JUDGE_SYSTEM,
            user_prompt=f"PREDICTED: {predicted}\nEXPECTED: {gold}",
            temperature=0.0,
            max_tokens=128,
            timeout_s=self.judge_timeout_s,
            decision_field="equivalent",
            score_field="",
        )
        if err:
            return None, {"error": err}
        return decision, {"model": self.judge_model, "raw": (raw or "")[:400], "decision": decision}


@register_evaluator("exact")
class ExactMatchEvaluator:
    def __init__(self, **_: Any) -> None:
        pass

    def evaluate(self, task_meta: Dict[str, Any], output_text: str) -> EvalResult:
        gold = str(task_meta.get("answer") or task_meta.get("expected") or "").strip()
        pred = output_text.strip()
        ok = bool(gold) and pred == gold
        return EvalResult(
            passed=ok, score=1.0 if ok else 0.0,
            details={"evaluator": "exact", "gold": gold, "pred": pred[:200]},
        )


def _pick_gold(task_meta: Dict[str, Any]) -> str:
    for key in ("answer", "expected", "gold", "ground_truth"):
        v = task_meta.get(key)
        if v:
            return str(v).strip()
    cfg = task_meta.get("eval_config") or {}
    if isinstance(cfg, dict):
        for key in ("answer", "expected", "gold", "ground_truth"):
            v = cfg.get(key)
            if v:
                return str(v).strip()
    return ""


_BOXED_RE_PREFIX = "\\boxed"


def _last_boxed(text: str):
    import re as _re
    positions = [m.start() for m in _re.finditer(r"\\boxed\b", text)]
    if not positions:
        return None
    for start in reversed(positions):
        i = start + len(_BOXED_RE_PREFIX)
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            continue
        depth = 0
        j = i
        while j < len(text):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1 : j]
            j += 1
    return None


def _scan_for_imo_answer(text: str):
    import re as _re
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    matches = _re.findall(r"final\s+answer\s+is\s*(.+)", text, _re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    bold = _re.findall(r"(?:\*\*|__)\s*(.+?)\s*(?:\*\*|__)", text)
    if bold:
        return bold[-1].strip()
    return None


def _math_verify_equivalent(pred: str, gold: str):
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        return False, f"math_verify not installed: {exc}"
    try:
        p = pred if "$" in pred else f"${pred}$"
        g = gold if "$" in gold else f"${gold}$"
        return bool(verify(parse(g), parse(p))), None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


_JUDGE_SYSTEM = (
    "You are a strict equivalence judge for math contest answers. "
    "Decide whether the PREDICTED answer is mathematically equivalent to the EXPECTED answer. "
    "Reply with JSON only: {\"equivalent\": true} or {\"equivalent\": false}."
)


def _openrouter_judge(predicted: str, expected: str, *, model: str, timeout_s: float):
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None, {"skipped": True, "reason": "no_OPENROUTER_API_KEY"}
    decision, _, raw, err = _call_openai_judge(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model,
        system_prompt=_JUDGE_SYSTEM,
        user_prompt=f"PREDICTED: {predicted}\nEXPECTED: {expected}",
        temperature=0.0,
        max_tokens=128,
        timeout_s=timeout_s,
        decision_field="equivalent",
        score_field="",
    )
    if err:
        return None, {"error": err}
    return decision, {"model": model, "raw": (raw or "")[:400], "decision": decision}


def _call_openai_judge(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    decision_field: str,
    score_field: str,
):
    """Synchronous OpenAI-compatible /v1/chat/completions call for judging.

    Returns (decision, score, raw_text, error_str). decision/score may be None.
    """
    import json as _json
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    body = _json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return None, None, None, f"{type(exc).__name__}: {exc}"

    choices = payload.get("choices") or []
    raw = (choices[0].get("message", {}).get("content", "") if choices else "") or ""

    decision = _extract_json_field(raw, decision_field)
    if isinstance(decision, str):
        decision = decision.strip().lower() in {"true", "yes", "1"}
    elif decision is None:
        decision = _parse_judge_decision(raw)

    score: Optional[float] = None
    if score_field:
        raw_score = _extract_json_field(raw, score_field)
        try:
            score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            score = None

    return decision, score, raw, None


def _extract_json_field(text: str, field: str):
    import json as _json
    import re as _re
    if not field:
        return None
    try:
        data = _json.loads(text.strip())
        if isinstance(data, dict) and field in data:
            return data[field]
    except Exception:
        pass
    m = _re.search(r"\{.*\}", text, _re.DOTALL)
    if m:
        try:
            data = _json.loads(m.group(0))
            if isinstance(data, dict) and field in data:
                return data[field]
        except Exception:
            pass
    return None


def _parse_judge_decision(text: str):
    import json as _json
    import re as _re
    stripped = text.strip()
    try:
        data = _json.loads(stripped)
        if isinstance(data, dict) and "equivalent" in data:
            return bool(data["equivalent"])
    except Exception:
        pass
    m = _re.search(r"\{.*\}", stripped, _re.DOTALL)
    if m:
        try:
            data = _json.loads(m.group(0))
            if isinstance(data, dict) and "equivalent" in data:
                return bool(data["equivalent"])
        except Exception:
            pass
    low = stripped.lower()
    if low.startswith("yes") or "\"equivalent\": true" in low:
        return True
    if low.startswith("no") or "\"equivalent\": false" in low:
        return False
    return None


__all__ = [
    "EvalResult",
    "Evaluator",
    "register_evaluator",
    "get_evaluator",
    "list_evaluators",
]
