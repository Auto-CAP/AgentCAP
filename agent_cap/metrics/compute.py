from dataclasses import dataclass
from typing import Dict, Optional


def compute_slr(
    small_model_quality: float,
    small_model_params: float,
    equivalent_bare_params: float,
) -> float:
    _ = small_model_quality
    if small_model_params <= 0:
        return 0.0
    return equivalent_bare_params / small_model_params


def compute_mcv(
    quality_before: float,
    quality_after: float,
    gpu_seconds_before: float,
    gpu_seconds_after: float,
) -> float:
    delta_gpu = gpu_seconds_after - gpu_seconds_before
    if delta_gpu <= 0:
        return float("inf") if quality_after > quality_before else 0.0
    return (quality_after - quality_before) / delta_gpu


def compute_gar(total_inference_seconds: float, total_wall_clock_seconds: float) -> float:
    if total_wall_clock_seconds <= 0:
        return 0.0
    return total_inference_seconds / total_wall_clock_seconds


@dataclass
class ComputeMetrics:
    slr: Optional[float] = None
    mcv: Optional[float] = None
    gar: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {"slr": self.slr, "mcv": self.mcv, "gar": self.gar}
