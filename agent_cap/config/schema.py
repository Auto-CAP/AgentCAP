from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml


@dataclass
class ModelConfig:
    id: str
    arch: str = "dense"
    params_b: float = 0
    active_b: float = 0
    tp: int = 1
    tier: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            id=str(data["id"]),
            arch=str(data.get("arch", "dense")),
            params_b=float(data.get("params_b", 0)),
            active_b=float(data.get("active_b", 0)),
            tp=int(data.get("tp", 1)),
            tier=str(data.get("tier", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "arch": self.arch,
            "params_b": self.params_b,
            "active_b": self.active_b,
            "tp": self.tp,
            "tier": self.tier,
        }


@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    models: List[ModelConfig] = field(default_factory=list)
    quantizations: List[str] = field(default_factory=lambda: ["fp16"])
    skill_subsets: List[str] = field(default_factory=lambda: ["all"])
    num_retries: List[int] = field(default_factory=lambda: [0])
    temperatures: List[float] = field(default_factory=lambda: [0.0])
    agent_modes: List[str] = field(default_factory=lambda: ["single-pass"])
    serving_engine: str = "sglang"
    repetitions: int = 3
    max_tokens: int = 4096
    gpu_type: str = "H100-80G"
    num_gpus: int = 8
    task_filter: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        model_dicts = data.get("models", []) or []
        models = [ModelConfig.from_dict(m) for m in model_dicts]
        task_filter_raw = data.get("task_filter")
        task_filter = None
        if task_filter_raw is not None:
            task_filter = [str(t) for t in task_filter_raw]

        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            models=models,
            quantizations=[str(x) for x in (data.get("quantizations", ["fp16"]) or ["fp16"])],
            skill_subsets=[str(x) for x in (data.get("skill_subsets", ["all"]) or ["all"])],
            num_retries=[int(x) for x in (data.get("num_retries", [0]) or [0])],
            temperatures=[float(x) for x in (data.get("temperatures", [0.0]) or [0.0])],
            agent_modes=[str(x) for x in (data.get("agent_modes", ["single-pass"]) or ["single-pass"])],
            serving_engine=str(data.get("serving_engine", "sglang")),
            repetitions=int(data.get("repetitions", 3)),
            max_tokens=int(data.get("max_tokens", 4096)),
            gpu_type=str(data.get("gpu_type", "H100-80G")),
            num_gpus=int(data.get("num_gpus", 8)),
            task_filter=task_filter,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "models": [m.to_dict() for m in self.models],
            "quantizations": self.quantizations,
            "skill_subsets": self.skill_subsets,
            "num_retries": self.num_retries,
            "temperatures": self.temperatures,
            "agent_modes": self.agent_modes,
            "serving_engine": self.serving_engine,
            "repetitions": self.repetitions,
            "max_tokens": self.max_tokens,
            "gpu_type": self.gpu_type,
            "num_gpus": self.num_gpus,
            "task_filter": self.task_filter,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("Experiment config YAML must contain a mapping at top level")
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    @property
    def total_configs(self) -> int:
        return (
            len(self.models)
            * len(self.quantizations)
            * len(self.skill_subsets)
            * len(self.num_retries)
            * len(self.temperatures)
            * len(self.agent_modes)
        )

    def iter_configs(self) -> Iterator[Dict[str, Any]]:
        for model, quant, skills, retries, temp, mode in product(
            self.models,
            self.quantizations,
            self.skill_subsets,
            self.num_retries,
            self.temperatures,
            self.agent_modes,
        ):
            yield {
                "model": model,
                "quantization": quant,
                "skill_subset": skills,
                "num_retries": retries,
                "temperature": temp,
                "agent_mode": mode,
            }
