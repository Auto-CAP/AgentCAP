"""
Accuracy metrics utilities for evaluating model predictions.

This module provides functions to compute various accuracy metrics
such as exact match (EM), used across different profilers and evaluation tasks.
"""

import re
from typing import List, Dict, Any, Optional
from decimal import Decimal, InvalidOperation  


def _norm_basic(s: str) -> str:
    """Light normalization for robust string matching (conservative)."""
    s = (s or "").strip()

    # remove surrounding quotes / backticks
    s = s.strip('`"\' ')

    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\left(", "(")
    s = s.replace("\\right)", ")")
    s = s.replace("\\left[", "[")
    s = s.replace("\\right]", "]")
    s = s.replace("\\left\\{", "{")
    s = s.replace("\\right\\}", "}")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\,", "")
    s = s.replace("~", "")
    s = s.replace("\\;", "")
    s = s.replace("\\!", "")
    s = s.replace("\\:", "")
    

    s = s.replace("\\leq", "\\le")
    s = s.replace("\\geq", "\\ge")
    s = s.replace("≤", "\\le")
    s = s.replace("≥", "\\ge")

    # normalize latex currency / percent
    s = s.replace("\\$", "$")
    s = s.replace("\\%", "%")

   # LaTeX wrapper stripping (e.g., \text{B} -> B) 
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathit\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathsf\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathtt\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\operatorname\s*\{([^{}]*)\}", r"\1", s)


    # strip surrounding $...$
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()

    # strip leading currency symbol
    if s.startswith("$"):
        s = s[1:].strip()

    # collapse whitespace
    s = re.sub(r"\s+", "", s)

    return s.lower()


def _parse_single_number(s: str) -> Optional[Decimal]:
    if s is None:
        return None

    t = str(s).strip()
    if not t:
        return None

    t = t.replace("\\$", "$").replace("\\%", "%")

    if len(t) >= 2 and t[0] == "$" and t[-1] == "$":
        t = t[1:-1].strip()

    t = re.sub(r"\s+", "", t)

    if t.startswith("$"):
        t = t[1:]

    if t.endswith("%"):
        t = t[:-1]

    t_no_comma = t.replace(",", "")

    nums = re.findall(r"-?\d+(?:\.\d+)?", t_no_comma)
    if len(nums) != 1:
        return None

    rest = re.sub(r"-?\d+(?:\.\d+)?", "", t_no_comma)
    rest = rest.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "")
    if rest not in ["", "+", "-", "."]:
        return None

    try:
        return Decimal(nums[0])
    except (InvalidOperation, ValueError):
        return None


def _extract_all_boxed(text: str) -> List[str]:
    results = []
    start = 0
    while True:
        idx = text.find(r"\boxed{", start)
        if idx == -1:
            break
        
        search_start = idx + 7
        stack = 1
        content = ""
        for i in range(search_start, len(text)):
            if text[i] == '{':
                stack += 1
            elif text[i] == '}':
                stack -= 1
            
            if stack == 0:
                results.append(content.strip())
                start = i + 1
                break
            content += text[i]
        else:
            break
    return results


def extract_answer(text: str, dataset_name: str) -> str:
    if not text or not text.strip():
        return ""

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    
    if not text:
        return ""

    ds = dataset_name.lower()

    if ds in ["numina", "numinamath"]:
        all_boxed = _extract_all_boxed(text)
        
        if all_boxed:
            return " ||| ".join(all_boxed)
        
        lines = text.strip().splitlines()
        tail_lines = lines[-10:] if len(lines) >= 10 else lines
        tail_text = "\n".join(tail_lines)
        
        marker_patterns = [
            r"####\s*(.+?)\s*$",
            r"(?:final\s+)?answer\s*:?\s*(.+?)\s*$",
        ]
        for pat in marker_patterns:
            m = re.search(pat, tail_text, re.IGNORECASE | re.MULTILINE)
            if m:
                candidate = m.group(1).strip().strip(".")
                nested_boxed = _extract_all_boxed(candidate)
                if nested_boxed:
                    return nested_boxed[0]
                return candidate
        

        tail = lines[-5:] if len(lines) >= 5 else lines
        for ln in reversed(tail):
            ln_stripped = ln.strip().strip(".")
            if any(left in ln_stripped for left in ["(-", "[-", "("]) and \
               any(right in ln_stripped for right in ["]", ")"]):
                if not re.match(r"^(so|thus|therefore|since|we have|this gives|consider)\s", ln_stripped, re.IGNORECASE):
                    return ln_stripped
        for ln in reversed(tail):
            ln_stripped = ln.strip().strip(".")
            if any(sym in ln_stripped for sym in ["\\le", "\\ge", "\\leq", "\\geq", "≤", "≥"]):
                if not re.match(r"^(so|thus|therefore|since|we have|this gives|when|if)\s", ln_stripped, re.IGNORECASE):
                    if len(ln_stripped) < 150:
                        return ln_stripped
        
        return ""

    if ds in ["gsm8k", "math"]:
        boxed_contents = _extract_all_boxed(text)
        if boxed_contents:
            num_match = re.search(r"(-?\d[\d,]*)", boxed_contents[-1])
            if num_match:
                return num_match.group(1).replace(",", "").strip()

        high_confidence_patterns = [
            r"####\s*(-?\d[\d,]*)",
            r"\*\*Final Answer:\*\*\s*\$?(-?\d[\d,]*)",
            r"\*\*Answer:\*\*\s*\$?(-?\d[\d,]*)",
            r"(?:final\s+)?answer\s*(?:is|:)\s*\$?(-?\d[\d,]*)\$?(?:\s*\.)?$",
        ]
        for pattern in high_confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(",", "").strip()

        return ""

    return text.strip()


def compute_exact_match(predictions: List[str], targets: List[str]) -> Dict[str, Any]:
    correct = 0
    no_answer = 0
    for pred, target in zip(predictions, targets):
        if not pred:
            no_answer += 1
            continue
            
        target_norm = _norm_basic(str(target))

        target_num = _parse_single_number(str(target))
        
        pred_candidates = str(pred).split(" ||| ")

        is_correct = False
        for c in pred_candidates:
            pred_num = _parse_single_number(c)
            if pred_num is not None and target_num is not None and pred_num == target_num:
                is_correct = True
                break
            if _norm_basic(c) == target_norm:
                is_correct = True
                break

        if is_correct:
            correct += 1

    em_score = correct / len(predictions) if len(predictions) > 0 else 0.0
    return {
        "exact_match": em_score,
        "correct": correct,
        "total": len(predictions),
        "no_answer": no_answer,
    }


def compute_accuracy_metrics(
    predictions: List[str],
    targets: List[str],
    dataset_name: str,
    extract_answers: bool = True,
) -> Dict[str, Any]:
    if extract_answers:
        extracted_predictions = [
            extract_answer(pred, dataset_name) for pred in predictions
        ]
    else:
        extracted_predictions = predictions

    return compute_exact_match(extracted_predictions, targets)


def format_accuracy_summary(metrics: Dict[str, Any]) -> str:
    em = metrics.get("exact_match", 0.0)
    correct = metrics.get("correct", 0)
    total = metrics.get("total", 0)
    no_answer = metrics.get("no_answer", 0)
    if no_answer > 0:
        return f"EM = {em:.4f} ({correct}/{total}, {no_answer} no answer)"
    return f"EM = {em:.4f} ({correct}/{total})"