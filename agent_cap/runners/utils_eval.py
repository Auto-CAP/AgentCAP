import re

BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")

def extract_boxed(text: str) -> str:
    text = text or ""
    ms = list(BOX_RE.finditer(text))
    if ms:
        return ms[-1].group(1).strip()
    return text.strip()

def normalize(s: str) -> str:
    return (s or "").strip()

def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))