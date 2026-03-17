#!/usr/bin/env python3
"""Generate NeurIPS-quality figures for the AgentCAP paper."""

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DB_PATH = Path(__file__).parent.parent / "results" / "hybrid_experiments.db"
OUT_DIR = Path(__file__).parent.parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "local-qwen4b": "#2196F3",
    "gpt54-qwen4b": "#FF9800",
    "gpt54-self": "#4CAF50",
    "local-qwen32b": "#9C27B0",
    "gpt54-qwen32b": "#F44336",
}

LABELS = {
    "local-qwen4b": "Qwen3-4B (local)",
    "gpt54-qwen4b": "GPT-5.4 → Qwen3-4B",
    "gpt54-self": "GPT-5.4 (self)",
    "local-qwen32b": "Qwen3-32B (local)",
    "gpt54-qwen32b": "GPT-5.4 → Qwen3-32B",
}

MARKERS = {
    "local-qwen4b": "o",
    "gpt54-qwen4b": "^",
    "gpt54-self": "s",
    "local-qwen32b": "o",
    "gpt54-qwen32b": "^",
}

ORDER = ["local-qwen4b", "gpt54-qwen4b", "gpt54-self", "local-qwen32b", "gpt54-qwen32b"]


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    data = {}
    for exp in ORDER:
        rows = conn.execute(
            "SELECT * FROM hybrid_runs WHERE experiment_name = ?", (exp,)
        ).fetchall()
        costs = sorted(r["total_cost_usd"] for r in rows)
        n = len(costs)
        data[exp] = {
            "n": n,
            "avg_cov": np.mean([r["coverage_score"] or 0 for r in rows]),
            "median_cost": costs[n // 2] if n else 0,
            "avg_cost": np.mean(costs),
            "avg_plan_cost": np.mean([r["plan_cost_usd"] for r in rows]),
            "avg_exec_cost": np.mean([r["exec_cost_usd"] for r in rows]),
            "tool_pct": sum(1 for r in rows if r["exec_tool_calls"] > 0) / n * 100,
            "avg_tc": np.mean([r["exec_tool_calls"] for r in rows]),
            "avg_lat": np.mean([r["total_latency_s"] for r in rows]),
            "pass75": sum(1 for r in rows if (r["coverage_score"] or 0) >= 0.75)
            / n
            * 100,
        }
    conn.close()
    return data


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def fig1_pareto(data):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    xs, ys = [], []
    for exp in ORDER:
        d = data[exp]
        x, y = d["median_cost"], d["avg_cov"]
        xs.append(x)
        ys.append(y)
        ax.scatter(
            x,
            y,
            c=COLORS[exp],
            marker=MARKERS[exp],
            s=120,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
        offset_x = 0.08 if exp != "gpt54-self" else -0.15
        offset_y = 0.02 if exp != "local-qwen32b" else -0.03
        ax.annotate(
            LABELS[exp],
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            ha="left",
        )

    # Pareto frontier: gpt54-self dominates on cost-accuracy
    pareto_exps = ["local-qwen4b", "gpt54-qwen4b", "gpt54-self"]
    px = [data[e]["median_cost"] for e in pareto_exps]
    py = [data[e]["avg_cov"] for e in pareto_exps]
    ax.plot(px, py, "k--", alpha=0.4, linewidth=1.0, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("Median Cost per Task (USD)", fontsize=12)
    ax.set_ylabel("Average Coverage Score", fontsize=12)
    ax.set_title("Cost–Accuracy Pareto Frontier", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.02, 0.50)
    save(fig, "fig1_pareto")


def fig2_cost_breakdown(data):
    sorted_exps = sorted(ORDER, key=lambda e: data[e]["avg_cost"])
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(sorted_exps))
    plan_costs = [data[e]["avg_plan_cost"] for e in sorted_exps]
    exec_costs = [data[e]["avg_exec_cost"] for e in sorted_exps]
    labels = [LABELS[e] for e in sorted_exps]

    ax.bar(
        x,
        plan_costs,
        0.6,
        label="Plan Phase",
        color="#90CAF9",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x,
        exec_costs,
        0.6,
        bottom=plan_costs,
        label="Exec Phase",
        color="#1565C0",
        edgecolor="black",
        linewidth=0.5,
    )

    for i, exp in enumerate(sorted_exps):
        total = data[exp]["avg_cost"]
        cov = data[exp]["avg_cov"]
        ax.text(
            i, total + 0.02, f"cov={cov:.2f}", ha="center", fontsize=8, style="italic"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Average Cost per Task (USD)", fontsize=12)
    ax.set_title(
        "Cost Breakdown: Planning vs Execution", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    save(fig, "fig2_cost_breakdown")


def fig3_planning_boost(data):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    groups = [
        ("Qwen3-4B", "local-qwen4b", "gpt54-qwen4b"),
        ("Qwen3-32B", "local-qwen32b", "gpt54-qwen32b"),
    ]

    x = np.arange(len(groups))
    width = 0.3

    for i, (label, local_exp, hybrid_exp) in enumerate(groups):
        local_cov = data[local_exp]["avg_cov"]
        hybrid_cov = data[hybrid_exp]["avg_cov"]
        boost = ((hybrid_cov - local_cov) / local_cov * 100) if local_cov > 0 else 0

        b1 = ax.bar(
            i - width / 2,
            local_cov,
            width,
            color="#BBDEFB",
            edgecolor="black",
            linewidth=0.5,
            label="Without Plan" if i == 0 else "",
        )
        b2 = ax.bar(
            i + width / 2,
            hybrid_cov,
            width,
            color="#FF9800",
            edgecolor="black",
            linewidth=0.5,
            label="With GPT-5.4 Plan" if i == 0 else "",
        )

        ax.text(
            i + width / 2,
            hybrid_cov + 0.008,
            f"+{boost:.0f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#E65100",
        )

        local_tool = data[local_exp]["tool_pct"]
        hybrid_tool = data[hybrid_exp]["tool_pct"]
        ax.text(
            i - width / 2,
            local_cov + 0.008,
            f"tool:{local_tool:.0f}%",
            ha="center",
            fontsize=7,
            color="#666",
        )
        ax.text(
            i + width / 2,
            hybrid_cov + 0.028,
            f"tool:{hybrid_tool:.0f}%",
            ha="center",
            fontsize=7,
            color="#666",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=12)
    ax.set_ylabel("Average Coverage Score", fontsize=12)
    ax.set_title(
        "Effect of Planning on Local Executor Quality", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 0.38)
    ax.tick_params(labelsize=10)
    save(fig, "fig3_planning_boost")


def fig4_latency_scatter(data):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for exp in ORDER:
        d = data[exp]
        inv_cost = 1.0 / max(d["median_cost"], 0.001)
        size = min(inv_cost * 8, 600)
        ax.scatter(
            d["avg_lat"],
            d["avg_cov"],
            c=COLORS[exp],
            marker=MARKERS[exp],
            s=max(size, 60),
            edgecolors="black",
            linewidths=0.5,
            alpha=0.85,
            zorder=5,
        )
        ax.annotate(
            LABELS[exp],
            (d["avg_lat"], d["avg_cov"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )

    ax.set_xlabel("Average Latency (seconds)", fontsize=12)
    ax.set_ylabel("Average Coverage Score", fontsize=12)
    ax.set_title(
        "Latency vs Accuracy (bubble size ∝ 1/cost)", fontsize=13, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.02, 0.50)
    ax.tick_params(labelsize=10)
    save(fig, "fig4_latency_accuracy")


def main():
    print("Loading data...")
    data = load_data()
    for exp in ORDER:
        d = data[exp]
        print(
            f"  {exp}: cov={d['avg_cov']:.3f} cost=${d['median_cost']:.4f} lat={d['avg_lat']:.0f}s"
        )

    print("\nGenerating figures...")
    fig1_pareto(data)
    fig2_cost_breakdown(data)
    fig3_planning_boost(data)
    fig4_latency_scatter(data)
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
