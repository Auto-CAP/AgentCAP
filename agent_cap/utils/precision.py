"""Resolve served quantization precision and normalize checkpoint ids.

HuggingFace writes ``torch_dtype: bfloat16`` into ``config.json`` even for
quantized checkpoints, so the dtype alone mislabels MXFP4/INT4/FP8 models as
bf16. The checkpoint's ``quantization_config`` is the source of truth.

Agentic runs hit a *served* endpoint and may not have the checkpoint on disk,
so :func:`resolve_precision` is best-effort: it reads a local ``config.json``
if the model reference is a path, else tries ``AutoConfig`` (HF cache/hub), and
returns ``None`` if neither is reachable so callers can keep their default.
The pure :func:`resolve_precision_from_config` (dict in) is unit-tested.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _ct_weight_bits(qc: Dict[str, Any]):
    """(num_bits, type) of the weight quant in a compressed-tensors config."""
    groups = qc.get("config_groups") or {}
    for g in groups.values():
        w = (g or {}).get("weights") or {}
        if w.get("num_bits") is not None:
            return w.get("num_bits"), w.get("type")
    return qc.get("num_bits"), qc.get("type")


def _find_quant_config(cfg: Dict[str, Any]):
    """quantization_config, incl. multimodal nesting (text_config/thinker_config)."""
    if not isinstance(cfg, dict):
        return None
    qc = cfg.get("quantization_config")
    if qc:
        return qc
    for sub in ("text_config", "thinker_config", "language_config"):
        s = cfg.get(sub)
        if isinstance(s, dict):
            found = _find_quant_config(s)
            if found:
                return found
    return None


def _find_dtype(cfg: Dict[str, Any]):
    """Model dtype — HF uses ``torch_dtype`` or (newer) ``dtype``; may be nested."""
    if not isinstance(cfg, dict):
        return None
    for key in ("torch_dtype", "dtype"):
        v = cfg.get(key)
        if v:
            return str(v)
    for sub in ("text_config", "thinker_config", "language_config"):
        s = cfg.get(sub)
        if isinstance(s, dict):
            found = _find_dtype(s)
            if found:
                return found
    return None


def resolve_precision_from_config(cfg: Dict[str, Any]) -> Optional[str]:
    """Resolve the served precision from an HF config dict.

    Prefer ``quantization_config`` (possibly nested in ``text_config`` for
    multimodal checkpoints); fall back to the model dtype only for genuinely
    dense models. Returns e.g. ``mxfp4``/``int4``/``fp8``/``bfloat16`` or None.
    """
    if not isinstance(cfg, dict):
        return None
    qc = _find_quant_config(cfg)
    if not qc:
        return _find_dtype(cfg)
    qm = str(qc.get("quant_method", "")).lower()
    # compressed-tensors is identified by quant_method OR by config_groups /
    # a "*-quantized" format (kimi-k2.5 omits quant_method entirely).
    is_ct = (qm == "compressed-tensors" or qc.get("config_groups")
             or "quantized" in str(qc.get("format", "")).lower())
    if is_ct:
        nbits, wtype = _ct_weight_bits(qc)
        if nbits == 4:
            return "int4" if str(wtype or "").lower() == "int" else "mxfp4"
        if nbits == 8:
            return "fp8"
    if qm in ("mxfp4", "nvfp4"):
        return qm  # keep the method name; don't collapse to a generic "fp4"
    if qm:
        return qm  # fp8, awq, gptq, ...
    return _find_dtype(cfg)


def _load_hf_config(model: str) -> Dict[str, Any]:
    """Best-effort load of a checkpoint's config.json (local path or HF cache)."""
    if not model:
        return {}
    try:
        p = Path(model)
        cj = p / "config.json" if p.is_dir() else p
        if str(cj).endswith("config.json") and Path(cj).is_file():
            return json.loads(Path(cj).read_text())
    except Exception:
        pass
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model, trust_remote_code=True).to_dict()
    except Exception:
        return {}


def resolve_precision(model: str) -> Optional[str]:
    """Resolve served precision for a model id or checkpoint path (best-effort)."""
    return resolve_precision_from_config(_load_hf_config(model))


def normalize_model_id(name) -> str:
    """Reduce a checkpoint reference to an HF id (``org/model``).

    A bare local path like ``/workspace/models/gpt-oss-120b`` loses its org, so
    keep the last two path segments; leave a proper ``org/model`` id intact.
    """
    if not name:
        return name
    s = str(name).rstrip("/")
    if s.startswith("/") or s.count("/") >= 2:
        parts = [p for p in s.split("/") if p]
        return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else s)
    return s
