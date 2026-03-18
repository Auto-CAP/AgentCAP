from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ParetoPoint:
    config_id: str
    quality: float
    gpu_seconds: float
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def is_dominated(point: ParetoPoint, by: ParetoPoint) -> bool:
    return by.quality >= point.quality and by.gpu_seconds <= point.gpu_seconds and (
        by.quality > point.quality or by.gpu_seconds < point.gpu_seconds
    )


def compute_pareto_frontier(points: List[ParetoPoint]) -> List[ParetoPoint]:
    if not points:
        return []

    frontier: List[ParetoPoint] = []
    for point in points:
        dominated = any(
            is_dominated(point, other)
            for other in points
            if other is not point
        )
        if not dominated:
            frontier.append(point)

    return sorted(frontier, key=lambda p: p.gpu_seconds)


def compute_pareto_3d(points: List[ParetoPoint]) -> List[ParetoPoint]:
    if not points:
        return []

    frontier: List[ParetoPoint] = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            if (
                other.quality >= point.quality
                and other.gpu_seconds <= point.gpu_seconds
                and other.latency_ms <= point.latency_ms
                and (
                    other.quality > point.quality
                    or other.gpu_seconds < point.gpu_seconds
                    or other.latency_ms < point.latency_ms
                )
            ):
                dominated = True
                break

        if not dominated:
            frontier.append(point)

    return sorted(frontier, key=lambda p: p.gpu_seconds)
