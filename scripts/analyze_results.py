#!/usr/bin/env python3

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent_cap.analysis.pareto import ParetoPoint, compute_pareto_frontier


LATEX_NL = "\\\\"
GPU_FALLBACK_MIN_SECONDS = 1e-3
GPU_TO_LATENCY_RATIO_FLOOR = 0.05
ESCALATION_STRATEGIES = {"cascade", "adaptive-cascade"}


@dataclass
class RunRecord:
    experiment_name: str
    model_id: str
    strategy: str
    task_id: str
    task_success: Optional[bool]
    gpu_seconds: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    combination_detail: str
    tool_call_count: Optional[int]

    @property
    def latency_seconds(self) -> float:
        return max(0.0, self.latency_ms / 1000.0)

    @property
    def effective_gpu_seconds(self) -> float:
        gpu = max(0.0, self.gpu_seconds)
        latency_s = self.latency_seconds
        if latency_s <= 0:
            return gpu
        if gpu <= GPU_FALLBACK_MIN_SECONDS:
            return latency_s
        if gpu < latency_s * GPU_TO_LATENCY_RATIO_FLOOR:
            return latency_s
        return gpu


@dataclass
class EscalationStats:
    strategy: str
    escalated: int
    not_escalated: int
    escalation_rate: Optional[float]
    acc_escalated: Optional[float]
    acc_not_escalated: Optional[float]
    avg_small_confidence: Optional[float]


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
    avg_tool_calls: Optional[float]
    is_pareto_optimal: bool = False
    escalation_rate: Optional[float] = None
    acc_escalated: Optional[float] = None
    acc_not_escalated: Optional[float] = None
    avg_small_confidence: Optional[float] = None


@dataclass
class ExperimentAnalysis:
    db_path: Path
    experiment_name: str
    benchmark_label: str
    pair_label: str
    has_tool_call_count: bool
    strategies: List[StrategyStats]
    pareto_frontier: List[str]
    escalation_rows: List[EscalationStats]
    baselines: List[Dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=(
            "analyze_results.py --db DB [--db2 DB2] [--output-dir DIR] "
            "[--pair-label LABEL] [--format {text,latex,both}] [--baselines DB]"
        )
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--db2", default=None)
    parser.add_argument("--output-dir", default="results/analysis/")
    parser.add_argument("--pair-label", default=None)
    parser.add_argument("--format", choices=["text", "latex", "both"], default="both")
    parser.add_argument("--baselines", default=None)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def safe_ratio(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def to_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        token = str(value).strip().lower()
        if token in {"true", "yes", "y", "t"}:
            return True
        if token in {"false", "no", "n", "f"}:
            return False
    return None


def detect_columns(conn: sqlite3.Connection, table_name: str = "runs") -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [str(row[1]) for row in rows]


def list_experiments(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT DISTINCT experiment_name FROM runs ORDER BY experiment_name"
    ).fetchall()
    names = [str(row[0]) for row in rows if row and row[0] is not None]
    return names or ["unknown-experiment"]


def infer_benchmark_label(experiment_name: str, db_path: Path) -> str:
    probe = f"{experiment_name}|{db_path.name}".lower()
    if "bigcodebench" in probe:
        return "BigCodeBench"
    if "mcp-atlas" in probe or "mcpatlas" in probe:
        return "MCP-Atlas"
    if "gsm8k" in probe:
        return "GSM8K"
    if "gpqa" in probe:
        return "GPQA"
    return experiment_name


def extract_size_tag(model_name: str) -> Optional[str]:
    tail = model_name.split("/")[-1]
    match = re.search(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?B(?:-[A-Za-z0-9]+)?)", tail)
    if match:
        return match.group(1)
    return None


def extract_model_short(model_name: str) -> str:
    size = extract_size_tag(model_name)
    if size:
        return size
    return model_name.split("/")[-1]


def extract_model_size_value(model_name: str) -> float:
    size = extract_size_tag(model_name)
    if not size:
        return math.inf
    match = re.match(r"(\d+(?:\.\d+)?)B", size)
    if not match:
        return math.inf
    return safe_float(match.group(1), default=math.inf)


def auto_detect_pair_label(runs: Sequence[RunRecord]) -> str:
    model_ids = sorted({run.model_id for run in runs if run.model_id})
    for model_id in model_ids:
        if "+" in model_id:
            parts = [part.strip() for part in model_id.split("+") if part.strip()]
            if len(parts) >= 2:
                return f"{extract_model_short(parts[0])} vs {extract_model_short(parts[1])}"
    if len(model_ids) >= 2:
        ordered = sorted(model_ids, key=lambda m: (extract_model_size_value(m), m))
        return (
            f"{extract_model_short(ordered[0])} vs {extract_model_short(ordered[-1])}"
        )
    if model_ids:
        return extract_model_short(model_ids[0])
    return "unknown_pair"


def slugify(text: str) -> str:
    lowered = (text or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "analysis"


def latex_escape(text: str) -> str:
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(ch, ch) for ch in text)


def parse_detail_json(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_confidence_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    for token in re.findall(r"\b\d+\b", text):
        value = int(token)
        if 0 <= value <= 10:
            return float(value)
    return None


def infer_escalation_info(
    strategy: str,
    detail_raw: str,
) -> Tuple[Optional[bool], Optional[float]]:
    detail = parse_detail_json(detail_raw)
    if detail is None:
        return None, None

    escalated: Optional[bool] = None
    small_confidence: Optional[float] = None

    if isinstance(detail, dict):
        if "escalated" in detail:
            escalated = to_optional_bool(detail.get("escalated"))
        if escalated is None and "winner" in detail:
            winner = str(detail.get("winner", "")).strip().lower()
            if winner in {"small", "large"}:
                escalated = winner == "large"
        if "small_confidence" in detail:
            conf = safe_float(detail.get("small_confidence"), default=float("nan"))
            if not math.isnan(conf):
                small_confidence = conf

    if isinstance(detail, list):
        step_names: List[str] = []
        for item in detail:
            if not isinstance(item, dict):
                continue
            step_name = str(item.get("step_name", "")).strip().lower()
            if step_name:
                step_names.append(step_name)
            if (
                strategy == "adaptive-cascade"
                and small_confidence is None
                and step_name == "self_assess"
            ):
                small_confidence = parse_confidence_from_text(
                    str(item.get("output_text", ""))
                )
        if strategy == "cascade" and step_names:
            escalated = any("large" in name for name in step_names)
        if strategy == "adaptive-cascade" and step_names:
            escalated = any(
                ("escalate" in name) or ("large" in name) for name in step_names
            )

    return escalated, small_confidence


def load_runs(
    conn: sqlite3.Connection,
    experiment_name: str,
) -> Tuple[List[RunRecord], bool]:
    columns = detect_columns(conn)
    has_tool_calls = "tool_call_count" in columns
    has_detail = "combination_detail" in columns

    selected = [
        "experiment_name",
        "model_id",
        "combination_strategy",
        "task_id",
        "task_success",
        "gpu_seconds",
        "latency_e2e_ms",
        "input_tokens",
        "output_tokens",
    ]
    if has_detail:
        selected.append("combination_detail")
    if has_tool_calls:
        selected.append("tool_call_count")

    query = (
        f"SELECT {', '.join(selected)} FROM runs "
        "WHERE experiment_name = ? ORDER BY combination_strategy, task_id"
    )
    rows = conn.execute(query, (experiment_name,)).fetchall()

    out: List[RunRecord] = []
    for row in rows:
        item = dict(row)
        out.append(
            RunRecord(
                experiment_name=str(item.get("experiment_name", experiment_name)),
                model_id=str(item.get("model_id", "")),
                strategy=str(item.get("combination_strategy", "") or ""),
                task_id=str(item.get("task_id", "")),
                task_success=to_optional_bool(item.get("task_success")),
                gpu_seconds=safe_float(item.get("gpu_seconds"), default=0.0),
                latency_ms=safe_float(item.get("latency_e2e_ms"), default=0.0),
                input_tokens=safe_int(item.get("input_tokens"), default=0),
                output_tokens=safe_int(item.get("output_tokens"), default=0),
                combination_detail=str(item.get("combination_detail", "") or ""),
                tool_call_count=safe_int(item.get("tool_call_count"), default=0)
                if has_tool_calls
                else None,
            )
        )
    return out, has_tool_calls


def compute_escalation_stats(runs: Sequence[RunRecord]) -> List[EscalationStats]:
    grouped: Dict[str, List[RunRecord]] = {}
    for run in runs:
        if run.strategy in ESCALATION_STRATEGIES:
            grouped.setdefault(run.strategy, []).append(run)

    out: List[EscalationStats] = []
    for strategy in sorted(grouped):
        items = grouped[strategy]
        esc_idx: List[int] = []
        non_idx: List[int] = []
        conf_values: List[float] = []

        for idx, run in enumerate(items):
            escalated, conf = infer_escalation_info(strategy, run.combination_detail)
            if conf is not None:
                conf_values.append(conf)
            if escalated is None:
                continue
            if escalated:
                esc_idx.append(idx)
            else:
                non_idx.append(idx)

        known = len(esc_idx) + len(non_idx)
        esc_pass = sum(1 for idx in esc_idx if items[idx].task_success)
        non_pass = sum(1 for idx in non_idx if items[idx].task_success)
        out.append(
            EscalationStats(
                strategy=strategy,
                escalated=len(esc_idx),
                not_escalated=len(non_idx),
                escalation_rate=safe_ratio(len(esc_idx), known),
                acc_escalated=safe_ratio(esc_pass, len(esc_idx)),
                acc_not_escalated=safe_ratio(non_pass, len(non_idx)),
                avg_small_confidence=safe_mean(conf_values) if conf_values else None,
            )
        )

    return out


def compute_strategy_stats(
    runs: Sequence[RunRecord],
    has_tool_call_count: bool,
    escalation_rows: Sequence[EscalationStats],
) -> Tuple[List[StrategyStats], List[str]]:
    grouped: Dict[str, List[RunRecord]] = {}
    for run in runs:
        strategy = run.strategy.strip()
        if strategy:
            grouped.setdefault(strategy, []).append(run)

    if not grouped:
        return [], []

    escalation_map = {row.strategy: row for row in escalation_rows}
    rows: List[StrategyStats] = []

    for strategy, items in grouped.items():
        tasks = len(items)
        pass_count = sum(1 for run in items if run.task_success)
        accuracy = pass_count / tasks if tasks else 0.0
        avg_gpu = safe_mean([run.effective_gpu_seconds for run in items])
        avg_latency = safe_mean([run.latency_seconds for run in items])
        avg_input = safe_mean([float(run.input_tokens) for run in items])
        avg_output = safe_mean([float(run.output_tokens) for run in items])
        avg_tool_calls = None
        if has_tool_call_count:
            avg_tool_calls = safe_mean(
                [float(run.tool_call_count or 0) for run in items]
            )

        esc = escalation_map.get(strategy)
        rows.append(
            StrategyStats(
                name=strategy,
                tasks=tasks,
                pass_count=pass_count,
                accuracy=accuracy,
                avg_gpu_seconds=avg_gpu,
                avg_latency_s=avg_latency,
                avg_input_tokens=avg_input,
                avg_output_tokens=avg_output,
                avg_tool_calls=avg_tool_calls,
                escalation_rate=esc.escalation_rate if esc else None,
                acc_escalated=esc.acc_escalated if esc else None,
                acc_not_escalated=esc.acc_not_escalated if esc else None,
                avg_small_confidence=esc.avg_small_confidence if esc else None,
            )
        )

    rows.sort(key=lambda row: (-row.accuracy, row.avg_gpu_seconds, row.name))

    points = [
        ParetoPoint(
            config_id=row.name,
            quality=row.accuracy,
            gpu_seconds=row.avg_gpu_seconds,
            latency_ms=row.avg_latency_s * 1000.0,
            metadata={"tasks": row.tasks},
        )
        for row in rows
    ]
    frontier = compute_pareto_frontier(points)
    frontier_names = [point.config_id for point in frontier]
    frontier_set = set(frontier_names)

    for row in rows:
        row.is_pareto_optimal = row.name in frontier_set

    return rows, frontier_names


def load_baselines(db_path: Path) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT model_id, task_success, gpu_seconds, latency_e2e_ms FROM runs ORDER BY model_id"
        ).fetchall()
    finally:
        conn.close()

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        model_id = str(row["model_id"] or "")
        grouped.setdefault(model_id, []).append(dict(row))

    ordered = sorted(grouped.keys(), key=lambda m: (extract_model_size_value(m), m))
    out: List[Dict[str, Any]] = []

    for idx, model_id in enumerate(ordered):
        items = grouped[model_id]
        tasks = len(items)
        pass_count = 0
        gpu_vals: List[float] = []

        for item in items:
            success = to_optional_bool(item.get("task_success"))
            if success:
                pass_count += 1

            gpu = safe_float(item.get("gpu_seconds"), 0.0)
            latency_ms = safe_float(item.get("latency_e2e_ms"), 0.0)
            if gpu <= 0.0 or gpu < 1e-6:
                gpu = latency_ms / 1000.0 if latency_ms > 0 else 0.0
            gpu_vals.append(gpu)

        if len(ordered) == 1:
            label = "single-pass"
        elif idx == 0:
            label = "single-small"
        elif idx == len(ordered) - 1:
            label = "single-large"
        else:
            label = f"single-{extract_model_short(model_id)}"

        out.append(
            {
                "name": label,
                "model_id": model_id,
                "model_short": extract_model_short(model_id),
                "tasks": tasks,
                "pass_count": pass_count,
                "accuracy": (pass_count / tasks) if tasks else 0.0,
                "avg_gpu_seconds": safe_mean(gpu_vals),
            }
        )

    return out


def analyze_experiment(
    db_path: Path,
    experiment_name: str,
    pair_label_override: Optional[str],
    baselines: Optional[List[Dict[str, Any]]],
) -> ExperimentAnalysis:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        runs, has_tool_calls = load_runs(conn, experiment_name)
    finally:
        conn.close()

    if not runs:
        raise ValueError(
            f"No runs found for experiment '{experiment_name}' in {db_path}"
        )

    pair_label = pair_label_override or auto_detect_pair_label(runs)
    benchmark_label = infer_benchmark_label(experiment_name, db_path)
    escalation_rows = compute_escalation_stats(runs)
    strategy_rows, frontier = compute_strategy_stats(
        runs, has_tool_calls, escalation_rows
    )

    return ExperimentAnalysis(
        db_path=db_path,
        experiment_name=experiment_name,
        benchmark_label=benchmark_label,
        pair_label=pair_label,
        has_tool_call_count=has_tool_calls,
        strategies=strategy_rows,
        pareto_frontier=frontier,
        escalation_rows=escalation_rows,
        baselines=baselines or [],
    )


def format_text_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Optional[Sequence[str]] = None,
) -> str:
    if aligns is None:
        aligns = ["left"] * len(headers)

    widths: List[int] = []
    for i, header in enumerate(headers):
        max_row = max((len(str(row[i])) for row in rows), default=0)
        widths.append(max(len(str(header)), max_row))

    def pad(text: str, width: int, align: str) -> str:
        if align == "right":
            return text.rjust(width)
        if align == "center":
            return text.center(width)
        return text.ljust(width)

    header_line = " | ".join(
        pad(str(headers[i]), widths[i], aligns[i]) for i in range(len(headers))
    )
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))

    body_lines = []
    for row in rows:
        body_lines.append(
            " | ".join(
                pad(str(row[i]), widths[i], aligns[i]) for i in range(len(headers))
            )
        )

    return "\n".join([header_line, sep_line] + body_lines)


def format_latex_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Optional[Sequence[str]] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    if aligns is None:
        aligns = ["left"] * len(headers)

    align_map = {"left": "l", "right": "r", "center": "c"}
    col_spec = "".join(align_map.get(a, "l") for a in aligns)

    lines: List[str] = [r"\begin{table}[h]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{latex_escape(caption)}}}")
    if label:
        lines.append(rf"\label{{{latex_escape(label)}}}")

    lines.extend(
        [
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(latex_escape(str(h)) for h in headers) + r" \\",
            r"\midrule",
        ]
    )
    for row in rows:
        lines.append(" & ".join(latex_escape(str(c)) for c in row) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def render_strategy_text(analysis: ExperimentAnalysis) -> str:
    headers = [
        "Strategy",
        "Tasks",
        "Pass",
        "Accuracy",
        "Avg GPU-s",
        "Avg Latency",
        "Pareto",
    ]
    rows: List[List[str]] = []

    for row in analysis.strategies:
        rows.append(
            [
                row.name,
                str(row.tasks),
                str(row.pass_count),
                f"{row.accuracy * 100:.1f}%",
                f"{row.avg_gpu_seconds:.2f}",
                f"{row.avg_latency_s:.2f}",
                "yes" if row.is_pareto_optimal else "no",
            ]
        )

    aligns = ["left", "right", "right", "right", "right", "right", "center"]
    title = f"Strategy comparison ({analysis.benchmark_label}, {analysis.pair_label})"
    return title + "\n" + format_text_table(headers, rows, aligns)


def render_strategy_latex(analysis: ExperimentAnalysis) -> str:
    headers = [
        "Strategy",
        "Tasks",
        "Pass",
        "Accuracy",
        "Avg GPU-s",
        "Avg Latency(s)",
        "Pareto",
    ]
    rows: List[List[str]] = []

    for row in analysis.strategies:
        rows.append(
            [
                row.name,
                str(row.tasks),
                str(row.pass_count),
                f"{row.accuracy * 100:.1f}\\%",
                f"{row.avg_gpu_seconds:.2f}",
                f"{row.avg_latency_s:.2f}",
                "yes" if row.is_pareto_optimal else "no",
            ]
        )

    aligns = ["left", "right", "right", "right", "right", "right", "center"]
    caption = (
        f"Strategy comparison for {analysis.pair_label} on {analysis.benchmark_label}"
    )
    label = f"tab:{slugify(analysis.benchmark_label)}:{slugify(analysis.pair_label)}:strategy"
    return format_latex_table(headers, rows, aligns, caption, label)


def render_escalation_text(analysis: ExperimentAnalysis) -> str:
    headers = [
        "Strategy",
        "Escalated",
        "Not Escalated",
        "Escalation Rate",
        "Acc (Escalated)",
        "Acc (Not Escalated)",
    ]

    rows: List[List[str]] = []
    for row in analysis.escalation_rows:
        rows.append(
            [
                row.strategy,
                str(row.escalated),
                str(row.not_escalated),
                f"{(row.escalation_rate or 0.0) * 100:.1f}%"
                if row.escalation_rate is not None
                else "N/A",
                f"{(row.acc_escalated or 0.0) * 100:.1f}%"
                if row.acc_escalated is not None
                else "N/A",
                f"{(row.acc_not_escalated or 0.0) * 100:.1f}%"
                if row.acc_not_escalated is not None
                else "N/A",
            ]
        )

    if not rows:
        return "Escalation analysis\nNo cascade/adaptive-cascade rows found."

    aligns = ["left", "right", "right", "right", "right", "right"]
    return "Escalation analysis\n" + format_text_table(headers, rows, aligns)


def render_escalation_latex(analysis: ExperimentAnalysis) -> str:
    headers = [
        "Strategy",
        "Escalated",
        "Not Escalated",
        "Escalation Rate",
        "Acc (Escalated)",
        "Acc (Not Escalated)",
    ]

    rows: List[List[str]] = []
    for row in analysis.escalation_rows:
        rows.append(
            [
                row.strategy,
                str(row.escalated),
                str(row.not_escalated),
                f"{(row.escalation_rate or 0.0) * 100:.1f}\\%"
                if row.escalation_rate is not None
                else "N/A",
                f"{(row.acc_escalated or 0.0) * 100:.1f}\\%"
                if row.acc_escalated is not None
                else "N/A",
                f"{(row.acc_not_escalated or 0.0) * 100:.1f}\\%"
                if row.acc_not_escalated is not None
                else "N/A",
            ]
        )

    aligns = ["left", "right", "right", "right", "right", "right"]
    caption = f"Escalation statistics on {analysis.benchmark_label}"
    label = f"tab:{slugify(analysis.benchmark_label)}:{slugify(analysis.pair_label)}:escalation"
    return format_latex_table(headers, rows, aligns, caption, label)


def save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf", dpi=300)
    plt.close(fig)


def plot_pareto(analysis: ExperimentAnalysis, output_dir: Path) -> None:
    if not analysis.strategies:
        return

    names = [row.name for row in analysis.strategies]
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0.0, 1.0, max(len(names), 1)))

    fig, ax = plt.subplots()
    for idx, row in enumerate(analysis.strategies):
        ax.scatter(
            row.avg_gpu_seconds,
            row.accuracy * 100.0,
            marker="*" if row.is_pareto_optimal else "o",
            s=220 if row.is_pareto_optimal else 90,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            row.name,
            (row.avg_gpu_seconds, row.accuracy * 100.0),
            xytext=(6, 6),
            textcoords="offset points",
        )

    frontier_map = {row.name: row for row in analysis.strategies}
    frontier_rows = [
        frontier_map[name] for name in analysis.pareto_frontier if name in frontier_map
    ]
    frontier_rows.sort(key=lambda item: item.avg_gpu_seconds)

    if frontier_rows:
        ax.plot(
            [row.avg_gpu_seconds for row in frontier_rows],
            [row.accuracy * 100.0 for row in frontier_rows],
            linestyle="--",
            color="black",
            linewidth=1.5,
            zorder=2,
        )

    ax.set_xlabel("Avg GPU-seconds per task")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Pareto Frontier — {analysis.pair_label}")
    save_figure(fig, output_dir, "pareto_frontier")


def plot_strategy_comparison(analysis: ExperimentAnalysis, output_dir: Path) -> None:
    stats = analysis.strategies
    if not stats:
        return

    fig, ax1 = plt.subplots()
    x = np.arange(len(stats), dtype=float)
    width = 0.38

    accuracy = [row.accuracy * 100.0 for row in stats]
    gpu = [row.avg_gpu_seconds for row in stats]
    labels = [row.name.replace("-", "\n") for row in stats]

    bars1 = ax1.bar(x - width / 2.0, accuracy, width, color="#4C72B0")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2.0, gpu, width, color="#DD8452", alpha=0.85)
    ax2.set_ylabel("Avg GPU-s/task")

    if stats:
        ax1.legend(
            [bars1[0], bars2[0]], ["Accuracy (%)", "Avg GPU-s/task"], loc="upper left"
        )

    ax1.set_title(f"Strategy Comparison — {analysis.pair_label}")
    save_figure(fig, output_dir, "strategy_comparison")


def plot_cost_accuracy(analysis: ExperimentAnalysis, output_dir: Path) -> None:
    points: List[Tuple[str, float, float, str]] = []

    for row in analysis.strategies:
        points.append(
            (row.name, row.avg_gpu_seconds, row.accuracy * 100.0, "multi-agent")
        )

    for baseline in analysis.baselines:
        points.append(
            (
                str(baseline.get("name", "baseline")),
                safe_float(baseline.get("avg_gpu_seconds"), 0.0),
                safe_float(baseline.get("accuracy"), 0.0) * 100.0,
                "single-agent baseline",
            )
        )

    if not points:
        return

    fig, ax = plt.subplots()
    for name, x, y, kind in points:
        color = "#1f77b4" if kind == "multi-agent" else "#2ca02c"
        ax.scatter(
            x, y, s=120, color=color, edgecolor="black", linewidth=0.6, alpha=0.85
        )
        ax.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points")

    ax.set_xlabel("Avg GPU-seconds per task")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Cost Accuracy Scatter — {analysis.pair_label}")
    save_figure(fig, output_dir, "cost_accuracy_scatter")


def plot_escalation(analysis: ExperimentAnalysis, output_dir: Path) -> None:
    escalation_stats = analysis.escalation_rows
    fig, ax = plt.subplots()

    if not escalation_stats:
        ax.text(0.5, 0.5, "No cascade strategies found", ha="center", va="center")
        ax.set_axis_off()
        save_figure(fig, output_dir, "escalation_analysis")
        return

    x = np.arange(len(escalation_stats), dtype=float)
    width = 0.25
    labels = [item.strategy.replace("-", "\n") for item in escalation_stats]
    rate_values = [
        (item.escalation_rate * 100.0) if item.escalation_rate is not None else 0.0
        for item in escalation_stats
    ]
    esc_values = [
        (item.acc_escalated * 100.0) if item.acc_escalated is not None else 0.0
        for item in escalation_stats
    ]
    non_values = [
        (item.acc_not_escalated * 100.0) if item.acc_not_escalated is not None else 0.0
        for item in escalation_stats
    ]

    ax.bar(x - width, rate_values, width, color="#55A868", label="Escalation rate (%)")
    ax.bar(x, esc_values, width, color="#C44E52", label="Escalated acc (%)")
    ax.bar(x + width, non_values, width, color="#8172B2", label="Not escalated acc (%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage")
    ax.set_title(f"Escalation Analysis — {analysis.pair_label}")
    ax.legend(loc="best")

    save_figure(fig, output_dir, "escalation_analysis")


def plot_tokens(analysis: ExperimentAnalysis, output_dir: Path) -> None:
    stats = analysis.strategies
    if not stats:
        return

    fig, ax = plt.subplots()

    x = np.arange(len(stats), dtype=float)
    labels = [row.name.replace("-", "\n") for row in stats]
    input_vals = np.asarray([row.avg_input_tokens for row in stats], dtype=float)
    output_vals = np.asarray([row.avg_output_tokens for row in stats], dtype=float)

    ax.bar(x, input_vals, color="#17becf", alpha=0.9, label="Input tokens")
    ax.bar(
        x,
        output_vals,
        bottom=input_vals,
        color="#bcbd22",
        alpha=0.9,
        label="Output tokens",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average tokens per task")
    ax.set_title(f"Token Usage Breakdown — {analysis.pair_label}")
    ax.legend(loc="best")

    save_figure(fig, output_dir, "token_usage")


def rank_map(rows: Sequence[StrategyStats]) -> Dict[str, int]:
    ordered = sorted(
        rows, key=lambda row: (-row.accuracy, row.avg_gpu_seconds, row.name)
    )
    return {row.name: idx + 1 for idx, row in enumerate(ordered)}


def render_cross_text(
    primary: ExperimentAnalysis, secondary: ExperimentAnalysis
) -> str:
    left = {row.name: row for row in primary.strategies}
    right = {row.name: row for row in secondary.strategies}
    left_rank = rank_map(primary.strategies)
    right_rank = rank_map(secondary.strategies)
    all_names = sorted(set(left) | set(right))

    headers = ["Strategy", "A Acc%", "A GPU-s", "A Rank", "B Acc%", "B GPU-s", "B Rank"]
    rows: List[List[str]] = []
    for name in all_names:
        l = left.get(name)
        r = right.get(name)
        rows.append(
            [
                name,
                f"{l.accuracy * 100:.1f}" if l else "-",
                f"{l.avg_gpu_seconds:.2f}" if l else "-",
                str(left_rank.get(name, "-")),
                f"{r.accuracy * 100:.1f}" if r else "-",
                f"{r.avg_gpu_seconds:.2f}" if r else "-",
                str(right_rank.get(name, "-")),
            ]
        )

    title = (
        f"Cross benchmark: A={primary.benchmark_label}, B={secondary.benchmark_label}"
    )
    table = format_text_table(
        headers, rows, ["left", "right", "right", "right", "right", "right", "right"]
    )
    return title + "\n" + table


def render_cross_latex(
    primary: ExperimentAnalysis, secondary: ExperimentAnalysis
) -> str:
    left = {row.name: row for row in primary.strategies}
    right = {row.name: row for row in secondary.strategies}
    left_rank = rank_map(primary.strategies)
    right_rank = rank_map(secondary.strategies)
    all_names = sorted(set(left) | set(right))

    headers = [
        "Strategy",
        "A Acc(%)",
        "A GPU-s",
        "A Rank",
        "B Acc(%)",
        "B GPU-s",
        "B Rank",
    ]
    rows: List[List[str]] = []
    for name in all_names:
        l = left.get(name)
        r = right.get(name)
        rows.append(
            [
                name,
                f"{l.accuracy * 100:.1f}" if l else "-",
                f"{l.avg_gpu_seconds:.2f}" if l else "-",
                str(left_rank.get(name, "-")),
                f"{r.accuracy * 100:.1f}" if r else "-",
                f"{r.avg_gpu_seconds:.2f}" if r else "-",
                str(right_rank.get(name, "-")),
            ]
        )

    caption = f"Cross-benchmark comparison: {primary.benchmark_label} vs {secondary.benchmark_label}"
    label = f"tab:cross:{slugify(primary.benchmark_label)}:{slugify(secondary.benchmark_label)}"
    return format_latex_table(
        headers,
        rows,
        ["left", "right", "right", "right", "right", "right", "right"],
        caption,
        label,
    )


def plot_cross(
    primary: ExperimentAnalysis, secondary: ExperimentAnalysis, output_dir: Path
) -> None:
    fig, ax = plt.subplots()

    payload = [(primary, "#1f77b4", "o"), (secondary, "#d62728", "s")]
    for analysis, color, marker in payload:
        ax.scatter(
            [row.avg_gpu_seconds for row in analysis.strategies],
            [row.accuracy * 100.0 for row in analysis.strategies],
            c=color,
            marker=marker,
            alpha=0.8,
            label=analysis.benchmark_label,
        )

        frontier_map = {row.name: row for row in analysis.strategies}
        frontier_rows = [
            frontier_map[name]
            for name in analysis.pareto_frontier
            if name in frontier_map
        ]
        if frontier_rows:
            frontier_rows.sort(key=lambda item: item.avg_gpu_seconds)
            ax.plot(
                [row.avg_gpu_seconds for row in frontier_rows],
                [row.accuracy * 100.0 for row in frontier_rows],
                color=color,
                linestyle="--",
                linewidth=1.6,
            )

    ax.set_xlabel("Avg GPU-seconds per task")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Cross-Benchmark Pareto — {primary.pair_label}")
    ax.legend(loc="best")
    save_figure(fig, output_dir, "cross_benchmark_pareto")


def generate_summary_json(analysis: ExperimentAnalysis) -> Dict[str, Any]:
    if not analysis.strategies:
        return {
            "db_path": str(analysis.db_path),
            "pair_label": analysis.pair_label,
            "benchmark": analysis.benchmark_label,
            "strategies": [],
            "pareto_frontier": [],
            "best_accuracy": {"strategy": None, "accuracy": None},
            "best_efficiency": {
                "strategy": None,
                "accuracy": None,
                "gpu_seconds": None,
            },
            "escalation": [],
            "baselines": analysis.baselines,
        }

    best_accuracy = max(analysis.strategies, key=lambda row: row.accuracy)
    best_eff = max(
        analysis.strategies,
        key=lambda row: row.accuracy / max(row.avg_gpu_seconds, 1e-9),
    )

    return {
        "db_path": str(analysis.db_path),
        "pair_label": analysis.pair_label,
        "benchmark": analysis.benchmark_label,
        "experiment_name": analysis.experiment_name,
        "strategies": [
            {
                "name": row.name,
                "tasks": row.tasks,
                "pass": row.pass_count,
                "accuracy": row.accuracy,
                "avg_gpu_seconds": row.avg_gpu_seconds,
                "avg_latency_s": row.avg_latency_s,
                "avg_input_tokens": row.avg_input_tokens,
                "avg_output_tokens": row.avg_output_tokens,
                "avg_tool_calls": row.avg_tool_calls,
                "pareto_optimal": row.is_pareto_optimal,
                "escalation_rate": row.escalation_rate,
                "acc_escalated": row.acc_escalated,
                "acc_not_escalated": row.acc_not_escalated,
                "avg_small_confidence": row.avg_small_confidence,
            }
            for row in analysis.strategies
        ],
        "pareto_frontier": analysis.pareto_frontier,
        "best_accuracy": {
            "strategy": best_accuracy.name,
            "accuracy": best_accuracy.accuracy,
        },
        "best_efficiency": {
            "strategy": best_eff.name,
            "accuracy": best_eff.accuracy,
            "gpu_seconds": best_eff.avg_gpu_seconds,
        },
        "escalation": [
            {
                "strategy": row.strategy,
                "escalated": row.escalated,
                "not_escalated": row.not_escalated,
                "escalation_rate": row.escalation_rate,
                "acc_escalated": row.acc_escalated,
                "acc_not_escalated": row.acc_not_escalated,
                "avg_small_confidence": row.avg_small_confidence,
            }
            for row in analysis.escalation_rows
        ],
        "baselines": analysis.baselines,
    }


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_outputs(
    analysis: ExperimentAnalysis, output_dir: Path, table_format: str
) -> None:
    if table_format in {"text", "both"}:
        write_text(
            output_dir / "strategy_comparison.txt", render_strategy_text(analysis)
        )
        write_text(
            output_dir / "escalation_table.txt", render_escalation_text(analysis)
        )

    if table_format in {"latex", "both"}:
        write_text(
            output_dir / "strategy_comparison.tex", render_strategy_latex(analysis)
        )
        write_text(
            output_dir / "escalation_table.tex", render_escalation_latex(analysis)
        )

    plot_pareto(analysis, output_dir)
    plot_strategy_comparison(analysis, output_dir)
    plot_cost_accuracy(analysis, output_dir)
    plot_escalation(analysis, output_dir)
    plot_tokens(analysis, output_dir)

    write_json(output_dir / "summary.json", generate_summary_json(analysis))


def analyze_db(
    db_path: Path,
    pair_label_override: Optional[str],
    baselines: Optional[List[Dict[str, Any]]],
) -> List[ExperimentAnalysis]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        experiments = list_experiments(conn)
    finally:
        conn.close()

    out: List[ExperimentAnalysis] = []
    for experiment_name in experiments:
        out.append(
            analyze_experiment(
                db_path=db_path,
                experiment_name=experiment_name,
                pair_label_override=pair_label_override,
                baselines=baselines,
            )
        )
    return out


def print_primary_text(
    reports: Sequence[ExperimentAnalysis], table_format: str
) -> None:
    if table_format not in {"text", "both"}:
        return

    for report in reports:
        print(render_strategy_text(report))
        print()
        print(render_escalation_text(report))
        print()


def main() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    plt.rcParams.update(
        {
            "font.size": 12,
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox_inches": "tight",
        }
    )

    args = parse_args()
    primary_db = Path(args.db)
    output_dir = Path(args.output_dir)

    if not primary_db.exists():
        raise FileNotFoundError(f"Primary DB not found: {primary_db}")

    output_dir.mkdir(parents=True, exist_ok=True)

    baselines = load_baselines(Path(args.baselines)) if args.baselines else []
    primary_reports = analyze_db(primary_db, args.pair_label, baselines)
    for report in primary_reports:
        save_outputs(report, output_dir, args.format)

    print_primary_text(primary_reports, args.format)

    if args.db2:
        secondary_db = Path(args.db2)
        if not secondary_db.exists():
            raise FileNotFoundError(f"Secondary DB not found: {secondary_db}")

        secondary_reports = analyze_db(secondary_db, args.pair_label, None)
        primary = primary_reports[0]
        secondary = secondary_reports[0]

        if args.format in {"text", "both"}:
            write_text(
                output_dir / "cross_benchmark_comparison.txt",
                render_cross_text(primary, secondary),
            )
            print(render_cross_text(primary, secondary))
            print()

        if args.format in {"latex", "both"}:
            write_text(
                output_dir / "cross_benchmark_comparison.tex",
                render_cross_latex(primary, secondary),
            )

        plot_cross(primary, secondary, output_dir)
        write_json(
            output_dir / "cross_benchmark_summary.json",
            {
                "benchmark_a": generate_summary_json(primary),
                "benchmark_b": generate_summary_json(secondary),
            },
        )


main()
