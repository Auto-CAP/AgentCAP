"""
Agent-CAP: Benchmarking of Cost, Accuracy, and Performance for Agentic AI Systems
"""

from agent_cap.core.types import Step, Trace, StepType
from agent_cap.core.tracer import Tracer, tracer
from agent_cap.visualization.timeline import TimelineVisualizer

from agent_cap.config.schema import ExperimentConfig, ModelConfig
from agent_cap.server.manager import ModelServerManager, ServerConfig
from agent_cap.server.client import ChatClient, ChatResponse
from agent_cap.server.gpu_monitor import GPUMonitor
from agent_cap.db.store import ResultStore, RunResult
from agent_cap.runner.executor import ExperimentExecutor, TaskDef
from agent_cap.metrics.pipeline import compute_sdr, compute_epr, compute_epd, PipelineMetrics
from agent_cap.metrics.compute import compute_slr, compute_mcv, compute_gar
from agent_cap.analysis.pareto import compute_pareto_frontier, ParetoPoint
from agent_cap.analysis.insights import (
    find_model_substitutions,
    find_diminishing_returns,
    Insight,
)

__version__ = "0.2.0"
__all__ = [
    "Step",
    "Trace",
    "StepType",
    "Tracer",
    "tracer",
    "TimelineVisualizer",
    "ExperimentConfig",
    "ModelConfig",
    "ModelServerManager",
    "ServerConfig",
    "ChatClient",
    "ChatResponse",
    "GPUMonitor",
    "ExperimentExecutor",
    "TaskDef",
    "ResultStore",
    "RunResult",
    "compute_sdr",
    "compute_epr",
    "compute_epd",
    "PipelineMetrics",
    "compute_slr",
    "compute_mcv",
    "compute_gar",
    "compute_pareto_frontier",
    "ParetoPoint",
    "find_model_substitutions",
    "find_diminishing_returns",
    "Insight",
]
