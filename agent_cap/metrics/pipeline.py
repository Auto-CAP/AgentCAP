from dataclasses import dataclass
from typing import Dict, List


@dataclass
class StepQuality:
    step_index: int
    quality: float
    model_id: str = ""
    quantization: str = "fp16"


def compute_sdr(step_qualities: List[float]) -> float:
    n = len(step_qualities)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = sum(step_qualities) / n

    numerator = sum((i - x_mean) * (q - y_mean) for i, q in enumerate(step_qualities))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_epr(step_qualities: List[float], threshold: float = 3.0) -> float:
    if len(step_qualities) < 2:
        return 0.0

    bad_after_bad = 0
    bad_after_good = 0
    total_after_bad = 0
    total_after_good = 0

    for i in range(len(step_qualities) - 1):
        current_bad = step_qualities[i] < threshold
        next_bad = step_qualities[i + 1] < threshold

        if current_bad:
            total_after_bad += 1
            if next_bad:
                bad_after_bad += 1
        else:
            total_after_good += 1
            if next_bad:
                bad_after_good += 1

    p_bad_after_bad = bad_after_bad / total_after_bad if total_after_bad > 0 else 0.0
    p_bad_after_good = bad_after_good / total_after_good if total_after_good > 0 else 0.0

    return p_bad_after_bad - p_bad_after_good


def compute_epd(qualities_by_depth: Dict[int, float], threshold: float = 0.95) -> int:
    if not qualities_by_depth:
        return 0

    max_depth = max(qualities_by_depth.keys())

    for depth in sorted(qualities_by_depth.keys()):
        if qualities_by_depth[depth] < threshold:
            return depth - 1 if depth > 0 else 0

    return max_depth


@dataclass
class PipelineMetrics:
    model_id: str
    quantization: str
    sdr: float
    epr: float
    epd: int
    step_qualities: List[float]
    num_steps: int
    threshold_tau: float = 0.95

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id,
            "quantization": self.quantization,
            "sdr": self.sdr,
            "epr": self.epr,
            "epd": self.epd,
            "num_steps": self.num_steps,
            "threshold_tau": self.threshold_tau,
        }
