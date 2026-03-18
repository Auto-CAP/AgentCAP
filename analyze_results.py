#!/usr/bin/env python3
"""AgentCAP — Experiment Analysis & Paper Figure Generation.

Reads experiment results from SQLite databases and produces:
  - Paper-quality matplotlib figures (PNG + PDF)
  - Strategy comparison tables (text + LaTeX)
  - Pareto frontier analysis
  - Escalation analysis for cascade strategies
  - Summary JSON for programmatic consumption

Usage:
    python analyze_results.py \
        --db results/qwen_combo.db \
        --output-dir results/analysis/pairA/ \
        --pair-label "PairA: 4B vs 30B-A3B"
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent
if _PROJECT_ROOT.name == "scripts":
    _PROJECT_ROOT = _PROJECT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GPU_FALLBACK_MIN = 1e-3
ESCALATION_STRATEGIES = {"cascade", "adaptive-cascade"}

MULTI_AGENT = {"cascade", "adaptive-cascade", "vote", "generate-verify"}
SINGLE_AGENT = {
    "best-of-n-small",
    "best-of-n-large",
    "self-critique-small",
    "self-critique-large",
}

STRATEGY_COLORS = {
    "cascade": "#2196F3",
    "adaptive-cascade": "#FF9800",
    "vote": "#4CAF50",
    "generate-verify": "#9C27B0",
    "best-of-n-small": "#78909C",
    "best-of-n-large": "#607D8B",
    "self-critique-small": "#BDBDBD",
    "self-critique-large": "#9E9E9E",
}

STRATEGY_LABELS = {
    "cascade": "Cascade",
    "adaptive-cascade": "Adapt-Casc",
    "vote": "Vote",
    "generate-verify": "Gen-Verify",
    "best-of-n-small": "BoN-S",
    "best-of-n-large": "BoN-L",
    "self-critique-small": "SC-S",
    "self-critique-large": "SC-L",
}

STRATEGY_ORDER = [
    "cascade",
    "adaptive-cascade",
    "vote",
    "generate-verify",
    "best-of-n-small",
    "best-of-n-large",
    "self-critique-small",
    "self-critique-large",
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    experiment_name: str
    model_id: str
    strategy: str
    task_id: str
    task_success: bool
    gpu_seconds: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    combination_detail: str
    tool_call_count: Optional[int]

    @property
    def effective_gpu_seconds(self) -> float:
        gpu = max(0.0, self.gpu_seconds)
        lat_s = max(0.0, self.latency_ms / 1000.0)
        if gpu <= GPU_FALLBACK_MIN and lat_s > 0:
            return lat_s
        return gpu if gpu > 0 else lat_s


@dataclass
class StrategyStats:
    name: str
    tasks: int
    pass_count: int
    accuracy: float
    avg_gpu_seconds: float
    avg_latency_s: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_tool_calls: Optional[float] = None
    is_pareto_optimal: bool = False
    escalation_rate: Optional[float] = None


@dataclass
class EscalationStats:
    strategy: str
    total: int
    escalated: int
    not_escalated: int
    escalation_rate: float
    acc_escalated: Optional[float]
    acc_not_escalated: Optional[float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sf(v: Any, d: float = 0.0) -> float:
    try:
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


def _si(v: Any, d: int = 0) -> int:
    try:
        return int(v) if v is not None else d
    except (TypeError, ValueError):
        return d


def _sb(v: Any) -> bool:
    if v is None:
        return False
    return bool(int(v))


def _mean(vals: Sequence[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _color(name: str) -> str:
    return STRATEGY_COLORS.get(name, "#333333")


def _label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def _is_multi_agent(name: str) -> bool:
    return name in MULTI_AGENT


def _strategy_sort_key(name: str) -> tuple[int, str]:
    try:
        return (STRATEGY_ORDER.index(name), name)
    except ValueError:
        return (len(STRATEGY_ORDER), name)


def _ordered_stats(stats: List[StrategyStats]) -> List[StrategyStats]:
    return sorted(stats, key=lambda s: _strategy_sort_key(s.name))


def _add_subtitle(ax: Axes, pair_label: str) -> None:
    ax.text(
        0.5,
        1.02,
        pair_label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#666666",
    )


def _bar_style_kwargs(name: str) -> Dict[str, Any]:
    kw: Dict[str, Any] = {
        "color": _color(name),
        "edgecolor": "black",
        "linewidth": 0.8,
        "alpha": 0.9,
    }
    if name in SINGLE_AGENT:
        kw["hatch"] = "///"
    return kw


def _group_legend_handles() -> List[Patch]:
    return [
        Patch(facecolor="#FFFFFF", edgecolor="black", label="Multi-agent"),
        Patch(
            facecolor="#FFFFFF",
            edgecolor="black",
            hatch="///",
            label="Single-agent control",
        ),
    ]


def _model_short(name: str) -> str:
    tail = name.split("/")[-1]
    m = re.search(r"(\d+(?:\.\d+)?B(?:-[A-Za-z0-9]+)?)", tail)
    return m.group(1) if m else tail


def _auto_pair(runs: Sequence[RunRecord]) -> str:
    mids = sorted({r.model_id for r in runs if r.model_id})
    for mid in mids:
        if "+" in mid:
            parts = [p.strip() for p in mid.split("+") if p.strip()]
            if len(parts) >= 2:
                return f"{_model_short(parts[0])} vs {_model_short(parts[1])}"
    return "unknown"


def _benchmark(exp_name: str, db_path: Path) -> str:
    probe = f"{exp_name}|{db_path.name}".lower()
    if "bigcodebench" in probe:
        return "BigCodeBench"
    if "mcp" in probe or "atlas" in probe:
        return "MCP-Atlas"
    return exp_name


# ---------------------------------------------------------------------------
# DB Loading
# ---------------------------------------------------------------------------


def load_runs(db_path: Path) -> List[RunRecord]:
    conn = sqlite3.connect(str(db_path))
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
    has_tc = "tool_call_count" in cols

    q = """SELECT experiment_name, model_id, combination_strategy, task_id,
                  task_success, gpu_seconds, latency_e2e_ms,
                  input_tokens, output_tokens, combination_detail
                  {tc}
           FROM runs WHERE combination_strategy IS NOT NULL
    """.format(tc=", tool_call_count" if has_tc else "")

    records = []
    for row in conn.execute(q).fetchall():
        tc = _si(row[10]) if has_tc and len(row) > 10 else None
        records.append(
            RunRecord(
                experiment_name=str(row[0] or ""),
                model_id=str(row[1] or ""),
                strategy=str(row[2] or ""),
                task_id=str(row[3] or ""),
                task_success=_sb(row[4]),
                gpu_seconds=_sf(row[5]),
                latency_ms=_sf(row[6]),
                input_tokens=_si(row[7]),
                output_tokens=_si(row[8]),
                combination_detail=str(row[9] or ""),
                tool_call_count=tc,
            )
        )
    conn.close()
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_strategy_stats(runs: List[RunRecord]) -> List[StrategyStats]:
    from agent_cap.analysis.pareto import ParetoPoint, compute_pareto_frontier

    by_strat: Dict[str, List[RunRecord]] = {}
    for r in runs:
        by_strat.setdefault(r.strategy, []).append(r)

    stats = []
    for name, grp in sorted(by_strat.items()):
        n = len(grp)
        passes = sum(1 for r in grp if r.task_success)
        gpu_v = [r.effective_gpu_seconds for r in grp]
        lat_v = [r.latency_ms / 1000.0 for r in grp]
        in_t = [float(r.input_tokens) for r in grp]
        out_t = [float(r.output_tokens) for r in grp]
        tc_v = [r.tool_call_count for r in grp if r.tool_call_count is not None]

        stats.append(
            StrategyStats(
                name=name,
                tasks=n,
                pass_count=passes,
                accuracy=passes / n if n else 0.0,
                avg_gpu_seconds=_mean(gpu_v),
                avg_latency_s=_mean(lat_v),
                avg_input_tokens=_mean(in_t),
                avg_output_tokens=_mean(out_t),
                avg_tool_calls=_mean([float(v) for v in tc_v]) if tc_v else None,
            )
        )

    # Pareto
    pts = [
        ParetoPoint(config_id=s.name, quality=s.accuracy, gpu_seconds=s.avg_gpu_seconds)
        for s in stats
    ]
    frontier_names = {p.config_id for p in compute_pareto_frontier(pts)}
    for s in stats:
        s.is_pareto_optimal = s.name in frontier_names

    return stats


def compute_escalation(runs: List[RunRecord]) -> List[EscalationStats]:
    results = []
    by_strat: Dict[str, List[RunRecord]] = {}
    for r in runs:
        if r.strategy in ESCALATION_STRATEGIES:
            by_strat.setdefault(r.strategy, []).append(r)

    for name, grp in sorted(by_strat.items()):
        esc, nesc = [], []
        for r in grp:
            if _was_escalated(r):
                esc.append(r)
            else:
                nesc.append(r)
        ne, nn = len(esc), len(nesc)
        total = ne + nn
        acc_e = (sum(1 for r in esc if r.task_success) / ne) if ne else None
        acc_n = (sum(1 for r in nesc if r.task_success) / nn) if nn else None
        results.append(
            EscalationStats(
                strategy=name,
                total=total,
                escalated=ne,
                not_escalated=nn,
                escalation_rate=ne / total if total else 0.0,
                acc_escalated=acc_e,
                acc_not_escalated=acc_n,
            )
        )
    return results


def _was_escalated(run: RunRecord) -> bool:
    if not run.combination_detail:
        return False
    try:
        steps = json.loads(run.combination_detail)
        if not isinstance(steps, list):
            return False
        for step in steps:
            sn = str(step.get("step_name", "")).lower()
            if "large" in sn:
                return True
        return False
    except (json.JSONDecodeError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Figure Style
# ---------------------------------------------------------------------------


def _setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "savefig.dpi": 200,
        }
    )


def _save(fig, stem: Path):
    fig.savefig(str(stem) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    print(f"  Saved: {stem.name}.png / .pdf")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_pareto(stats: List[StrategyStats], out: Path, label: str):
    stats = _ordered_stats(stats)
    fig, ax = plt.subplots(figsize=(10, 7))
    for s in stats:
        sz = 180 if s.is_pareto_optimal else 110
        face = _color(s.name) if _is_multi_agent(s.name) else "none"
        ax.scatter(
            s.avg_gpu_seconds,
            s.accuracy * 100,
            color=_color(s.name),
            facecolors=face,
            s=sz,
            marker="o",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        ax.annotate(
            _label(s.name),
            (s.avg_gpu_seconds, s.accuracy * 100),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    pareto = sorted(
        [s for s in stats if s.is_pareto_optimal], key=lambda s: s.avg_gpu_seconds
    )
    if len(pareto) >= 2:
        ax.plot(
            [s.avg_gpu_seconds for s in pareto],
            [s.accuracy * 100 for s in pareto],
            "k--",
            alpha=0.5,
            linewidth=1.5,
            label="Pareto frontier",
        )

    ax.set_xlabel("Avg GPU-seconds / task")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cost-Accuracy Pareto Frontier")
    _add_subtitle(ax, label)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="black",
            linestyle="None",
            label="Multi-agent",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="none",
            linestyle="None",
            label="Single-agent control",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label="Pareto frontier",
        ),
    ]
    ax.legend(handles=legend_handles, loc="best")
    ax.grid(True, alpha=0.3)
    _save(fig, out / "pareto_frontier")
    plt.close(fig)


def plot_strategy_bars(stats: List[StrategyStats], out: Path, label: str):
    stats = _ordered_stats(stats)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    names = [s.name for s in stats]
    short_names = [_label(n) for n in names]
    accs = [s.accuracy * 100 for s in stats]
    gpus = [s.avg_gpu_seconds for s in stats]
    x = np.arange(len(names))
    w = 0.35

    for i, name in enumerate(names):
        ax1.bar(
            x[i] - w / 2,
            accs[i],
            w,
            **_bar_style_kwargs(name),
        )
    ax1.set_ylabel("Accuracy (%)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    for i, name in enumerate(names):
        gpu_style = _bar_style_kwargs(name)
        gpu_style["alpha"] = 0.45
        ax2.bar(x[i] + w / 2, gpus[i], w, **gpu_style)
    ax2.set_ylabel("Avg GPU-seconds / task", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=25, ha="right")
    ax1.set_title("Multi-Agent Combination Strategies on BigCodeBench")
    _add_subtitle(ax1, label)
    metric_handles = [
        Patch(facecolor="#666666", edgecolor="black", alpha=0.9, label="Accuracy (%)"),
        Patch(
            facecolor="#666666",
            edgecolor="black",
            alpha=0.45,
            label="Avg GPU-s/task",
        ),
    ]
    ax1.legend(handles=metric_handles + _group_legend_handles(), loc="upper left")
    fig.tight_layout()
    _save(fig, out / "strategy_comparison")
    plt.close(fig)


def plot_scatter(stats: List[StrategyStats], out: Path, label: str):
    stats = _ordered_stats(stats)
    fig, ax = plt.subplots(figsize=(10, 7))
    for s in stats:
        face = _color(s.name) if _is_multi_agent(s.name) else "none"
        ax.scatter(
            s.avg_gpu_seconds,
            s.accuracy * 100,
            color=_color(s.name),
            facecolors=face,
            s=120,
            marker="o",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        ax.annotate(
            _label(s.name),
            (s.avg_gpu_seconds, s.accuracy * 100),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )
    ax.set_xlabel("Avg GPU-seconds / task")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Strategy Cost-Accuracy Landscape")
    _add_subtitle(ax, label)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="black",
                linestyle="None",
                label="Multi-agent",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="none",
                linestyle="None",
                label="Single-agent control",
            ),
        ],
        loc="best",
    )
    ax.grid(True, alpha=0.3)
    _save(fig, out / "cost_accuracy_scatter")
    plt.close(fig)


def plot_escalation(esc: List[EscalationStats], out: Path, label: str):
    if not esc:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    esc = sorted(esc, key=lambda e: _strategy_sort_key(e.strategy))
    names = [e.strategy for e in esc]
    short_names = [_label(n) for n in names]
    rates = [e.escalation_rate * 100 for e in esc]

    ax1 = axes[0]
    bars = []
    for i, name in enumerate(names):
        bars.extend(ax1.bar(short_names[i], rates[i], **_bar_style_kwargs(name)))
    for bar, rate in zip(bars, rates):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.0f}%",
            ha="center",
            fontsize=11,
        )
    ax1.set_ylabel("Escalation Rate (%)")
    ax1.set_title("Escalation Rate")
    ax1.set_ylim(0, 105)

    ax2 = axes[1]
    x = np.arange(len(names))
    w = 0.35
    acc_not = [((e.acc_not_escalated or 0) * 100) for e in esc]
    acc_esc = [((e.acc_escalated or 0) * 100) for e in esc]
    for i, name in enumerate(names):
        base_style = _bar_style_kwargs(name)
        not_style = dict(base_style)
        not_style["alpha"] = 0.4
        esc_style = dict(base_style)
        esc_style["alpha"] = 0.9
        ax2.bar(x[i] - w / 2, acc_not[i], w, **not_style)
        ax2.bar(x[i] + w / 2, acc_esc[i], w, **esc_style)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy by Escalation")
    metric_handles = [
        Patch(facecolor="#666666", edgecolor="black", alpha=0.4, label="Not Escalated"),
        Patch(facecolor="#666666", edgecolor="black", alpha=0.9, label="Escalated"),
    ]
    ax2.legend(handles=metric_handles + _group_legend_handles(), loc="upper right")

    fig.suptitle("Escalation Behavior in Cascade Strategies", fontsize=14)
    fig.text(0.5, 0.965, label, ha="center", va="top", fontsize=11, color="#666666")
    fig.tight_layout()
    _save(fig, out / "escalation_analysis")
    plt.close(fig)


def plot_tokens(stats: List[StrategyStats], out: Path, label: str):
    stats = _ordered_stats(stats)
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [s.name for s in stats]
    short_names = [_label(n) for n in names]
    in_t = [s.avg_input_tokens for s in stats]
    out_t = [s.avg_output_tokens for s in stats]
    x = np.arange(len(names))
    for i, name in enumerate(names):
        base_style = _bar_style_kwargs(name)
        in_style = dict(base_style)
        in_style["alpha"] = 0.8
        out_style = dict(base_style)
        out_style["alpha"] = 0.45
        ax.bar(x[i], in_t[i], **in_style)
        ax.bar(x[i], out_t[i], bottom=in_t[i], **out_style)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.set_ylabel("Avg Tokens / Task")
    ax.set_title("Per-Strategy Token Usage Composition")
    _add_subtitle(ax, label)
    token_handles = [
        Patch(facecolor="#666666", edgecolor="black", alpha=0.8, label="Input Tokens"),
        Patch(
            facecolor="#666666", edgecolor="black", alpha=0.45, label="Output Tokens"
        ),
    ]
    ax.legend(handles=token_handles + _group_legend_handles(), loc="upper left")
    fig.tight_layout()
    _save(fig, out / "token_usage")
    plt.close(fig)


def plot_cost_per_correct(
    stats: List[StrategyStats], runs: List[RunRecord], out: Path, label: str
):
    by_strat: Dict[str, List[RunRecord]] = {}
    for r in runs:
        by_strat.setdefault(r.strategy, []).append(r)

    data = []
    for s in stats:
        grp = by_strat.get(s.name, [])
        total_gpu = sum(r.effective_gpu_seconds for r in grp)
        correct = s.pass_count
        cpc = total_gpu / correct if correct > 0 else 0
        data.append((s.name, cpc, s.accuracy * 100, correct))

    data.sort(key=lambda x: x[1])
    data = [d for d in data if d[1] > 0]

    data.sort(key=lambda x: (_strategy_sort_key(x[0]), x[1]))

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [d[0] for d in data]
    short_names = [_label(n) for n in names]
    costs = [d[1] for d in data]
    accs = [d[2] for d in data]
    x = np.arange(len(names))

    bars = []
    for i, name in enumerate(names):
        bars.extend(ax.bar(x[i], costs[i], **_bar_style_kwargs(name)))
    for i, (bar, acc, correct) in enumerate(zip(bars, accs, [d[3] for d in data])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{acc:.0f}%\n({correct})",
            ha="center",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.set_ylabel("GPU-seconds per Correct Answer")
    ax.set_title("Compute Cost per Correct Solution")
    _add_subtitle(ax, label)
    ax.legend(handles=_group_legend_handles(), loc="upper left")
    fig.tight_layout()
    _save(fig, out / "cost_per_correct")
    plt.close(fig)


def plot_task_difficulty(runs: List[RunRecord], out: Path, label: str):
    by_strat: Dict[str, Dict[str, bool]] = {}
    all_tasks: set = set()
    for r in runs:
        by_strat.setdefault(r.strategy, {})[r.task_id] = r.task_success
        all_tasks.add(r.task_id)

    strat_names = sorted(by_strat.keys(), key=_strategy_sort_key)
    strat_short = [_label(s) for s in strat_names]
    task_list = sorted(all_tasks)

    solve_counts = {
        t: sum(1 for s in strat_names if by_strat.get(s, {}).get(t, False))
        for t in task_list
    }
    task_list.sort(key=lambda t: -solve_counts[t])

    matrix = np.zeros((len(strat_names), len(task_list)))
    for i, s in enumerate(strat_names):
        for j, t in enumerate(task_list):
            matrix[i, j] = 1.0 if by_strat.get(s, {}).get(t, False) else 0.0

    fig, ax = plt.subplots(
        figsize=(max(14, len(task_list) * 0.3), max(4, len(strat_names) * 0.5 + 2))
    )
    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(strat_names)))
    ax.set_yticklabels(strat_short)
    ax.set_title("Task Difficulty Across Strategies")
    _add_subtitle(ax, label)

    n_easy = sum(1 for t in task_list if solve_counts[t] == len(strat_names))
    n_hard = sum(1 for t in task_list if solve_counts[t] == 0)
    ax.set_xlabel(
        f"Tasks sorted by solve count ({len(task_list)} total) — {n_easy} solved by all, {n_hard} solved by none"
    )
    ax.set_xticks([])

    fig.tight_layout()
    _save(fig, out / "task_difficulty_heatmap")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def text_table(stats: List[StrategyStats]) -> str:
    hdr = f"{'Strategy':<22} {'Tasks':>5} {'Pass':>5} {'Acc%':>6} {'GPU-s':>8} {'Lat-s':>8} {'Pareto':>7}"
    sep = "-" * len(hdr)
    lines = [hdr, sep]
    for s in sorted(stats, key=lambda x: -x.accuracy):
        p = "  *" if s.is_pareto_optimal else ""
        lines.append(
            f"{s.name:<22} {s.tasks:>5} {s.pass_count:>5} {s.accuracy * 100:>5.1f}% "
            f"{s.avg_gpu_seconds:>7.1f} {s.avg_latency_s:>7.1f} {p:>7}"
        )
    return "\n".join(lines)


def latex_table(stats: List[StrategyStats]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Strategy comparison across multi-agent combination methods.}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Strategy & Tasks & Pass & Acc\% & GPU-s & Pareto \\",
        r"\midrule",
    ]
    for s in sorted(stats, key=lambda x: -x.accuracy):
        p = r"$\star$" if s.is_pareto_optimal else ""
        nm = s.name.replace("_", r"\_")
        lines.append(
            f"{nm} & {s.tasks} & {s.pass_count} & {s.accuracy * 100:.1f}\\% & {s.avg_gpu_seconds:.1f} & {p} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def escalation_text(esc: List[EscalationStats]) -> str:
    if not esc:
        return "(no escalation strategies)"
    hdr = f"{'Strategy':<22} {'Total':>5} {'Esc':>5} {'Not':>5} {'Rate':>7} {'Acc(E)':>7} {'Acc(N)':>7}"
    sep = "-" * len(hdr)
    lines = [hdr, sep]
    for e in esc:
        ae = f"{e.acc_escalated * 100:.1f}%" if e.acc_escalated is not None else "N/A"
        an = (
            f"{e.acc_not_escalated * 100:.1f}%"
            if e.acc_not_escalated is not None
            else "N/A"
        )
        lines.append(
            f"{e.strategy:<22} {e.total:>5} {e.escalated:>5} {e.not_escalated:>5} "
            f"{e.escalation_rate * 100:>5.1f}% {ae:>7} {an:>7}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def make_summary(
    db_path: Path,
    pair_label: str,
    benchmark: str,
    stats: List[StrategyStats],
    esc: List[EscalationStats],
) -> Dict[str, Any]:
    best_acc = max(stats, key=lambda s: s.accuracy) if stats else None
    pareto_names = [s.name for s in stats if s.is_pareto_optimal]
    pareto_sorted = sorted(
        [s for s in stats if s.is_pareto_optimal], key=lambda s: s.avg_gpu_seconds
    )
    best_eff = pareto_sorted[0] if pareto_sorted else None

    return {
        "db_path": str(db_path),
        "pair_label": pair_label,
        "benchmark": benchmark,
        "strategies": [
            {
                "name": s.name,
                "tasks": s.tasks,
                "pass_count": s.pass_count,
                "accuracy": round(s.accuracy, 4),
                "avg_gpu_seconds": round(s.avg_gpu_seconds, 2),
                "avg_latency_s": round(s.avg_latency_s, 2),
                "avg_input_tokens": round(s.avg_input_tokens, 1),
                "avg_output_tokens": round(s.avg_output_tokens, 1),
                "avg_tool_calls": round(s.avg_tool_calls, 2)
                if s.avg_tool_calls is not None
                else None,
                "pareto_optimal": s.is_pareto_optimal,
            }
            for s in sorted(stats, key=lambda x: -x.accuracy)
        ],
        "pareto_frontier": pareto_names,
        "best_accuracy": {
            "strategy": best_acc.name,
            "accuracy": round(best_acc.accuracy, 4),
        }
        if best_acc
        else None,
        "best_efficiency": {
            "strategy": best_eff.name,
            "accuracy": round(best_eff.accuracy, 4),
            "gpu_seconds": round(best_eff.avg_gpu_seconds, 2),
        }
        if best_eff
        else None,
        "escalation": [
            {
                "strategy": e.strategy,
                "total": e.total,
                "escalated": e.escalated,
                "not_escalated": e.not_escalated,
                "escalation_rate": round(e.escalation_rate, 4),
                "acc_escalated": round(e.acc_escalated, 4)
                if e.acc_escalated is not None
                else None,
                "acc_not_escalated": round(e.acc_not_escalated, 4)
                if e.acc_not_escalated is not None
                else None,
            }
            for e in esc
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentCAP experiment analysis")
    p.add_argument("--db", required=True, help="Primary results database")
    p.add_argument("--output-dir", default="results/analysis/", help="Output directory")
    p.add_argument(
        "--pair-label",
        default=None,
        help="Label for model pair (auto-detect if omitted)",
    )
    p.add_argument(
        "--format",
        choices=["text", "latex", "both"],
        default="both",
        help="Table format",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from {db_path} ...")
    runs = load_runs(db_path)
    if not runs:
        print("ERROR: No runs found.")
        return
    print(f"  {len(runs)} runs, {len({r.strategy for r in runs})} strategies")

    pair_label = args.pair_label or _auto_pair(runs)
    benchmark = _benchmark(runs[0].experiment_name, db_path)
    print(f"  Pair: {pair_label}  |  Benchmark: {benchmark}")

    _setup_style()
    stats = compute_strategy_stats(runs)
    esc_stats = compute_escalation(runs)

    # Attach escalation rate to strategy stats
    esc_map = {e.strategy: e.escalation_rate for e in esc_stats}
    for s in stats:
        if s.name in esc_map:
            s.escalation_rate = esc_map[s.name]

    # Print tables
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    tt = text_table(stats)
    print(tt)

    print("\n" + "=" * 60)
    print("ESCALATION ANALYSIS")
    print("=" * 60)
    et = escalation_text(esc_stats)
    print(et)

    # Save tables
    if args.format in ("text", "both"):
        (out_dir / "strategy_table.txt").write_text(tt)
        (out_dir / "escalation_table.txt").write_text(et)
    if args.format in ("latex", "both"):
        (out_dir / "strategy_table.tex").write_text(latex_table(stats))

    # Generate figures
    print("\nGenerating figures ...")
    plot_pareto(stats, out_dir, pair_label)
    plot_strategy_bars(stats, out_dir, pair_label)
    plot_scatter(stats, out_dir, pair_label)
    plot_escalation(esc_stats, out_dir, pair_label)
    plot_tokens(stats, out_dir, pair_label)
    plot_cost_per_correct(stats, runs, out_dir, pair_label)
    plot_task_difficulty(runs, out_dir, pair_label)

    # Summary JSON
    summary = make_summary(db_path, pair_label, benchmark, stats, esc_stats)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("  Saved summary.json")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    pareto = [s for s in stats if s.is_pareto_optimal]
    print(f"  Pareto-optimal: {', '.join(s.name for s in pareto)}")
    best = max(stats, key=lambda s: s.accuracy)
    print(f"  Highest accuracy: {best.name} ({best.accuracy * 100:.1f}%)")
    if pareto:
        cheapest = min(pareto, key=lambda s: s.avg_gpu_seconds)
        print(
            f"  Most efficient: {cheapest.name} ({cheapest.accuracy * 100:.1f}% @ {cheapest.avg_gpu_seconds:.1f} GPU-s)"
        )
    print("\nDone!")


if __name__ == "__main__":
    main()
