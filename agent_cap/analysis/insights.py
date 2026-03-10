from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Insight:
    category: str
    title: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "severity": self.severity,
        }


def find_model_substitutions(
    results: List[Dict[str, Any]],
    quality_threshold: float = 0.90,
) -> List[Insight]:
    insights: List[Insight] = []

    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        task = result.get("task_id", "unknown")
        by_task.setdefault(task, []).append(result)

    for task, task_results in by_task.items():
        if not task_results:
            continue

        best = max(task_results, key=lambda r: r.get("quality_score", 0))
        best_quality = best.get("quality_score", 0)
        if best_quality <= 0:
            continue

        target = best_quality * quality_threshold
        qualifying = [r for r in task_results if r.get("quality_score", 0) >= target]
        if not qualifying:
            continue

        cheapest = min(qualifying, key=lambda r: r.get("gpu_seconds", float("inf")))
        if cheapest.get("model_id") == best.get("model_id"):
            continue

        best_gpu_seconds = max(best.get("gpu_seconds", 1), 0.001)
        savings = 1 - (cheapest.get("gpu_seconds", 0) / best_gpu_seconds)

        if savings > 0.5:
            cheapest_quality = cheapest.get("quality_score", 0)
            cheapest_quant = cheapest.get("quantization", "")
            cheapest_skills = cheapest.get("skill_subset", "")
            insights.append(
                Insight(
                    category="substitution",
                    title=f"{cheapest['model_id']} replaces {best['model_id']} on {task}",
                    description=(
                        f"{cheapest['model_id']} (q={cheapest_quant}, skills={cheapest_skills}) "
                        f"achieves {cheapest_quality:.1f} quality "
                        f"({(cheapest_quality / best_quality) * 100:.0f}% of best) "
                        f"at {savings * 100:.0f}% lower compute cost"
                    ),
                    evidence={
                        "best": best,
                        "substitute": cheapest,
                        "savings_pct": savings * 100,
                    },
                )
            )

    return insights


def find_diminishing_returns(
    results: List[Dict[str, Any]],
    dimension: str = "num_retries",
) -> List[Insight]:
    insights: List[Insight] = []

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        key_parts = [
            f"model_id={result.get('model_id', '')}",
            f"quantization={result.get('quantization', '')}",
            f"skill_subset={result.get('skill_subset', '')}",
            f"task_id={result.get('task_id', '')}",
        ]
        key = "|".join(key_parts)
        groups.setdefault(key, []).append(result)

    for group in groups.values():
        if len(group) < 2:
            continue

        sorted_group = sorted(group, key=lambda r: r.get(dimension, 0))

        for index in range(1, len(sorted_group)):
            prev = sorted_group[index - 1]
            curr = sorted_group[index]

            delta_quality = curr.get("quality_score", 0) - prev.get("quality_score", 0)
            delta_cost = curr.get("gpu_seconds", 0) - prev.get("gpu_seconds", 0)

            if delta_cost > 0 and (delta_quality / delta_cost) < 0.005:
                mcv = delta_quality / delta_cost
                insights.append(
                    Insight(
                        category="diminishing_returns",
                        title=(
                            f"Diminishing returns: {dimension} "
                            f"{prev.get(dimension, '')} → {curr.get(dimension, '')}"
                        ),
                        description=(
                            f"Increasing {dimension} from {prev.get(dimension, '')} "
                            f"to {curr.get(dimension, '')} adds {delta_cost:.0f} gpu-seconds "
                            f"but only {delta_quality:.2f} quality points (MCV={mcv:.4f})"
                        ),
                        evidence={"prev": prev, "curr": curr, "mcv": mcv},
                        severity="warning",
                    )
                )

    return insights
