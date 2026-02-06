#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


def _get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def load_summary(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(s: Dict[str, Any]) -> Dict[str, Any]:
    em_mean = _get(s, "em.mean", None)
    em_correct = _get(s, "em.correct", None)

    latency_mean = _get(s, "latency_ms.mean", None)
    latency_p50 = _get(s, "latency_ms.p50", None)
    latency_p95 = _get(s, "latency_ms.p95", None)

    tool_call_rate = _get(s, "tool_call_rate", None)
    tool_calls_total = _get(s, "tool_calls_total", None)

    truncated_count = _get(s, "truncated_count", _get(s, "truncation.count", None))

    avg_q_tokens = _get(s, "tokens.avg_question_tokens", None)
    avg_out_tokens = _get(s, "tokens.avg_output_tokens", None)
    max_out_tokens = _get(s, "tokens.max_output_tokens", None)

    n_val = _as_int(_get(s, "n", _get(s, "runs", None)), default=None)

    return {
        "workload": _get(s, "workload", ""),
        "model_id": _get(s, "model_id", _get(s, "decoder.model", "")),
        "n": n_val,
        "em_mean": _as_float(em_mean, default=None),
        "em_correct": _as_int(em_correct, default=None),
        "lat_mean_ms": _as_float(latency_mean, default=None),
        "lat_p50_ms": _as_float(latency_p50, default=None),
        "lat_p95_ms": _as_float(latency_p95, default=None),
        "tool_call_rate": _as_float(tool_call_rate, default=None),
        "tool_calls_total": _as_int(tool_calls_total, default=None),
        "truncated_count": _as_int(truncated_count, default=None),
        "avg_question_tokens": _as_float(avg_q_tokens, default=None),
        "avg_output_tokens": _as_float(avg_out_tokens, default=None),
        "max_output_tokens": _as_float(max_out_tokens, default=None),
        "trace_dir": _get(s, "trace_dir", ""),
    }


def print_table(rows: List[Dict[str, Any]], labels: List[str]) -> None:
    cols = [
        ("label", 18),
        ("em_mean", 8),
        ("em_correct", 10),
        ("lat_p50_ms", 12),
        ("lat_p95_ms", 12),
        ("avg_output_tokens", 16),
        ("max_output_tokens", 16),
        ("tool_call_rate", 13),
        ("tool_calls_total", 16),
        ("truncated_count", 15),
    ]

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    header = " ".join([k.ljust(w) for k, w in cols])
    print(header)
    print("-" * len(header))
    for r, lab in zip(rows, labels):
        r2 = dict(r)
        r2["label"] = lab
        line = " ".join([fmt(r2.get(k, None)).ljust(w) for k, w in cols])
        print(line)


def write_csv(rows: List[Dict[str, Any]], labels: List[str], out_csv: Path) -> None:
    import csv
    fieldnames = ["label"] + sorted(set().union(*[set(r.keys()) for r in rows]))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r, lab in zip(rows, labels):
            rr = dict(r)
            rr["label"] = lab
            w.writerow(rr)


def _vals(rows: List[Dict[str, Any]], key: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for r in rows:
        v = r.get(key, None)
        if v is None:
            out.append(None)
        else:
            try:
                out.append(float(v))
            except Exception:
                out.append(None)
    return out


def _bar_group(ax, labels: List[str], series: List[List[Optional[float]]], series_names: List[str], title: str, ylabel: str):
    x = list(range(len(labels)))
    width = 0.8 / max(1, len(series))
    for i, ys in enumerate(series):
        vals = [0.0 if v is None else float(v) for v in ys]
        xi = [xx - 0.4 + width * (i + 0.5) for xx in x]
        ax.bar(xi, vals, width=width, label=series_names[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", help="Paths to summary JSON files")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels matching the summaries order")
    ap.add_argument("--out_prefix", type=str, default="summary_compare", help="Output prefix (name only; directory auto by n)")
    ap.add_argument("--base_dir", type=str, default="baselines", help="Base directory to create compare outputs")
    args = ap.parse_args()

    paths = [Path(p) for p in args.summaries]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Not found: {p}")

    labels = args.labels
    if labels is None or len(labels) == 0:
        labels = [p.stem for p in paths]
    if len(labels) != len(paths):
        raise SystemExit("--labels count must match number of summary files")

    rows: List[Dict[str, Any]] = []
    for p in paths:
        s = load_summary(p)
        rows.append(extract_metrics(s))

    # Decide output directory by n
    ns = [r.get("n") for r in rows if r.get("n") is not None]
    if len(ns) == 0:
        n_tag = "unknown"
    else:
        n_tag = str(ns[0]) if all(n == ns[0] for n in ns) else "mixed"

    out_dir = Path(args.base_dir) / f"compare_n{n_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = out_dir / args.out_prefix

    print_table(rows, labels)

    out_csv = out_prefix.with_suffix(".csv")
    write_csv(rows, labels, out_csv)
    print(f"\n[OK] wrote CSV: {out_csv}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel A: EM mean
    ems = _vals(rows, "em_mean")
    _bar_group(axes[0, 0], labels, [ems], ["EM"], "EM (mean)", "EM")

    # Panel B: Latency p50 vs p95
    p50 = _vals(rows, "lat_p50_ms")
    p95 = _vals(rows, "lat_p95_ms")
    _bar_group(axes[0, 1], labels, [p50, p95], ["p50", "p95"], "Latency (ms)", "ms")

    # Panel C: Avg / Max output tokens
    avg_out = _vals(rows, "avg_output_tokens")
    max_out = _vals(rows, "max_output_tokens")
    _bar_group(axes[1, 0], labels, [avg_out, max_out], ["avg_out", "max_out"], "Output tokens", "tokens")

    # Panel D: Tool call rate + truncated count (right axis)
    tool_rate = _vals(rows, "tool_call_rate")
    trunc = _vals(rows, "truncated_count")

    ax = axes[1, 1]
    _bar_group(ax, labels, [tool_rate], ["tool_call_rate"], "Tool + Truncation", "rate")

    ax2 = ax.twinx()
    x = list(range(len(labels)))
    trunc_vals = [0.0 if v is None else float(v) for v in trunc]
    ax2.plot(x, trunc_vals, marker="o")
    ax2.set_ylabel("truncated_count")

    plt.tight_layout()

    out_png = out_prefix.with_name(out_prefix.name + ".png")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    print(f"[OK] wrote ONE figure: {out_png}")
    print(f"[OK] outputs in: {out_dir}")


if __name__ == "__main__":
    main()