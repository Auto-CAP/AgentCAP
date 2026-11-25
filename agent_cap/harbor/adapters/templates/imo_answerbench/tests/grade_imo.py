"""Grade an IMO-AnswerBench answer inside the Harbor verifier.

Same decision procedure as AgentCAP's ``imo`` evaluator
(``agent_cap/agents/evaluators.py: IMOAnswerBenchEvaluator``):

1. take the last ``\\boxed{...}`` (else "final answer is ...", else the last bold span);
2. accept if ``math_verify`` finds it equivalent to the gold answer;
3. otherwise ask an LLM judge: the configured OpenAI-compatible endpoint
   (IMO_JUDGE_BASE_URL + IMO_JUDGE_API_KEY), else OpenRouter if
   OPENROUTER_API_KEY is set, else no judge.

Writes ``/logs/verifier/reward.json`` ({"reward": 0 or 1}) and
``/logs/verifier/grade.json`` (the evaluator's details). Self-contained:
standard library plus ``math_verify``.
"""

import json
import os
import re
import sys
import urllib.request

ANSWER_FILE = os.environ.get("ANSWER_FILE", "/workspace/answer.txt")
EXPECTED_FILE = os.environ.get("EXPECTED_FILE", "/tests/expected.json")
VERIFIER_DIR = os.environ.get("VERIFIER_DIR", "/logs/verifier")

_JUDGE_SYSTEM = (
    "You are a strict equivalence judge for math contest answers. "
    "Decide whether the PREDICTED answer is mathematically equivalent to the EXPECTED answer. "
    "Reply with JSON only: {\"equivalent\": true} or {\"equivalent\": false}."
)


def last_boxed(text):
    positions = [m.start() for m in re.finditer(r"\\boxed\b", text)]
    if not positions:
        return None
    for start in reversed(positions):
        i = start + len("\\boxed")
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


def scan_for_imo_answer(text):
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    matches = re.findall(r"final\s+answer\s+is\s*(.+)", text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    bold = re.findall(r"(?:\*\*|__)\s*(.+?)\s*(?:\*\*|__)", text)
    if bold:
        return bold[-1].strip()
    return None


def math_verify_equivalent(pred, gold):
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        return False, "math_verify not installed: %s" % exc
    try:
        p = pred if "$" in pred else "$%s$" % pred
        g = gold if "$" in gold else "$%s$" % gold
        return bool(verify(parse(g), parse(p))), None
    except Exception as exc:
        return False, "%s: %s" % (type(exc).__name__, exc)


def _extract_json_field(text, field):
    if not field:
        return None
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and field in data:
            return data[field]
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and field in data:
                return data[field]
        except Exception:
            pass
    return None


def _parse_judge_decision(text):
    stripped = text.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "equivalent" in data:
            return bool(data["equivalent"])
    except Exception:
        pass
    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
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


def call_judge(base_url, api_key, model, predicted, expected, timeout_s):
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": "PREDICTED: %s\nEXPECTED: %s" % (predicted, expected)},
        ],
        "temperature": 0.0,
        "max_tokens": 128,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return None, {"error": "%s: %s" % (type(exc).__name__, exc)}
    choices = payload.get("choices") or []
    raw = (choices[0].get("message", {}).get("content", "") if choices else "") or ""
    decision = _extract_json_field(raw, "equivalent")
    if isinstance(decision, str):
        decision = decision.strip().lower() in {"true", "yes", "1"}
    elif decision is None:
        decision = _parse_judge_decision(raw)
    return decision, {"model": model, "raw": raw[:400], "decision": decision}


def judge(predicted, gold):
    timeout_s = float(os.environ.get("IMO_JUDGE_TIMEOUT_S", "60"))
    base_url = os.environ.get("IMO_JUDGE_BASE_URL", "")
    api_key = os.environ.get("IMO_JUDGE_API_KEY", "")
    if base_url and api_key:
        model = os.environ.get("IMO_JUDGE_MODEL", "gemini-3.1-flash-lite")
        return call_judge(base_url, api_key, model, predicted, gold, timeout_s)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        return None, {"skipped": True, "reason": "no_OPENROUTER_API_KEY"}
    model = os.environ.get("IMO_JUDGE_OPENROUTER_MODEL", "openrouter/elephant-alpha")
    return call_judge("https://openrouter.ai/api/v1", openrouter_key, model, predicted, gold, timeout_s)


def evaluate(output_text, gold):
    """Return (passed, details) exactly as IMOAnswerBenchEvaluator.evaluate does."""
    if not output_text.strip():
        return False, {"evaluator": "imo-answerbench", "reason": "empty_output", "gold": gold}
    predicted = scan_for_imo_answer(output_text)
    if predicted is None or not gold:
        return False, {
            "evaluator": "imo-answerbench",
            "reason": "no_predicted_or_gold",
            "predicted": predicted,
            "gold": gold,
        }
    symbolic_ok, symbolic_err = math_verify_equivalent(predicted, gold)
    if symbolic_ok:
        return True, {
            "evaluator": "imo-answerbench",
            "method": "math_verify",
            "predicted": predicted,
            "gold": gold,
        }
    decision, judge_meta = judge(predicted, gold)
    if decision is True:
        return True, {
            "evaluator": "imo-answerbench",
            "method": "llm_judge",
            "predicted": predicted,
            "gold": gold,
            "judge": judge_meta,
        }
    return False, {
        "evaluator": "imo-answerbench",
        "method": "math_verify+llm_judge" if judge_meta else "math_verify",
        "predicted": predicted,
        "gold": gold,
        "math_verify_error": symbolic_err,
        "judge": judge_meta,
    }


def main():
    with open(EXPECTED_FILE, encoding="utf-8") as f:
        expected = json.load(f)
    gold = str(expected.get("answer", "")).strip()
    try:
        with open(ANSWER_FILE, encoding="utf-8") as f:
            output_text = f.read()
    except OSError:
        output_text = ""
    passed, details = evaluate(output_text, gold)
    details["problem_id"] = expected.get("problem_id")
    details["answer_file"] = ANSWER_FILE
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    with open(os.path.join(VERIFIER_DIR, "grade.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=1)
    with open(os.path.join(VERIFIER_DIR, "reward.json"), "w", encoding="utf-8") as f:
        json.dump({"reward": 1.0 if passed else 0.0}, f)
    print("passed" if passed else "failed", json.dumps(details, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
