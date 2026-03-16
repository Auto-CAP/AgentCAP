"""Configuration schema for single-agent benchmarking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SingleAgentBenchConfig:
    """Configuration for a single-agent performance benchmark.

    Attributes:
        name: Human-readable experiment name.
        model_id: HuggingFace model identifier (e.g. ``unsloth/GPT-OSS-120b``).
        serving_engine: Inference engine (``"vllm"`` or ``"sglang"``).
        base_url: Base URL of the running inference server.
        dataset: Benchmark dataset name (passed to ``load_benchmark``).
        dataset_count: Number of tasks to sample from the dataset.
        batch_sizes: List of concurrent-request batch sizes to sweep.
        enable_tool_calls: Whether to run the *with-tool-calls* variant.
        tool_definitions: Optional OpenAI-format tool definitions.
        max_tokens: Maximum output tokens per request.
        temperature: Sampling temperature.
        repetitions: How many times to repeat each (batch_size, tool_mode) combo.
        gpu_monitor_interval: GPU polling interval in seconds.
        cpu_monitor_interval: CPU polling interval in seconds.
        tp: Tensor-parallel degree (for server management).
        gpu_type: GPU model string for bookkeeping.
        python_path: Path to the Python interpreter for launching servers.
        cuda_visible_devices: CUDA device list (comma-separated).
        output_dir: Directory for result artefacts.
    """

    name: str = "single-agent-bench"
    model_id: str = "unsloth/GPT-OSS-120b"
    serving_engine: str = "vllm"
    base_url: str = "http://localhost:30000"
    dataset: str = "swebench_pro"
    dataset_count: int = 50
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    enable_tool_calls: bool = True
    tool_definitions: Optional[List[Dict[str, Any]]] = None
    max_tokens: int = 4096
    temperature: float = 0.0
    repetitions: int = 1
    gpu_monitor_interval: float = 0.5
    cpu_monitor_interval: float = 0.5
    tp: int = 1
    gpu_type: str = "H100-80G"
    python_path: str = "python"
    cuda_visible_devices: str = ""
    output_dir: str = "results/single_agent"
    max_turns: int = 20
    workspace_dir: str = "/tmp/agent_cap_workspace"
    shell_timeout: int = 30

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleAgentBenchConfig":
        return cls(
            name=str(data.get("name", "single-agent-bench")),
            model_id=str(data.get("model_id", "unsloth/GPT-OSS-120b")),
            serving_engine=str(data.get("serving_engine", "vllm")),
            base_url=str(data.get("base_url", "http://localhost:30000")),
            dataset=str(data.get("dataset", "swebench_pro")),
            dataset_count=int(data.get("dataset_count", 50)),
            batch_sizes=[int(b) for b in (data.get("batch_sizes") or [1])],
            enable_tool_calls=bool(data.get("enable_tool_calls", True)),
            tool_definitions=data.get("tool_definitions"),
            max_tokens=int(data.get("max_tokens", 4096)),
            temperature=float(data.get("temperature", 0.0)),
            repetitions=int(data.get("repetitions", 1)),
            gpu_monitor_interval=float(data.get("gpu_monitor_interval", 0.5)),
            cpu_monitor_interval=float(data.get("cpu_monitor_interval", 0.5)),
            tp=int(data.get("tp", 1)),
            gpu_type=str(data.get("gpu_type", "H100-80G")),
            python_path=str(data.get("python_path", "python")),
            cuda_visible_devices=str(data.get("cuda_visible_devices", "")),
            output_dir=str(data.get("output_dir", "results/single_agent")),
            max_turns=int(data.get("max_turns", 20)),
            workspace_dir=str(data.get("workspace_dir", "/tmp/agent_cap_workspace")),
            shell_timeout=int(data.get("shell_timeout", 30)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "serving_engine": self.serving_engine,
            "base_url": self.base_url,
            "dataset": self.dataset,
            "dataset_count": self.dataset_count,
            "batch_sizes": self.batch_sizes,
            "enable_tool_calls": self.enable_tool_calls,
            "tool_definitions": self.tool_definitions,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetitions": self.repetitions,
            "gpu_monitor_interval": self.gpu_monitor_interval,
            "cpu_monitor_interval": self.cpu_monitor_interval,
            "tp": self.tp,
            "gpu_type": self.gpu_type,
            "python_path": self.python_path,
            "cuda_visible_devices": self.cuda_visible_devices,
            "output_dir": self.output_dir,
            "max_turns": self.max_turns,
            "workspace_dir": self.workspace_dir,
            "shell_timeout": self.shell_timeout,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SingleAgentBenchConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("Config YAML must contain a mapping at top level")
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
