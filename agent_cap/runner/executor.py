import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from agent_cap.config.schema import ExperimentConfig, ModelConfig
from agent_cap.db.store import ResultStore, RunResult
from agent_cap.evaluator import EvalConfig, evaluate
from agent_cap.server.client import ChatClient
from agent_cap.server.gpu_monitor import GPUMonitor
from agent_cap.server.manager import ModelServerManager, ServerConfig

logger = logging.getLogger("agent_cap.runner")


@dataclass
class TaskDef:
    id: str
    name: str
    messages: List[Dict[str, Any]]
    category: str = ""
    expected_skills: List[str] = field(default_factory=list)
    eval_config: Optional[Dict[str, Any]] = None


class ExperimentExecutor:
    def __init__(
        self,
        config: ExperimentConfig,
        tasks: List[TaskDef],
        store: ResultStore,
        skills_loader: Optional[Callable[[str], str]] = None,
    ):
        self.config = config
        self.tasks = tasks
        self.store = store
        self.skills_loader = skills_loader
        self._existing_run_ids: Optional[Set[str]] = None

    def run(self) -> None:
        total = self.config.total_configs * len(self.tasks) * self.config.repetitions
        logger.info("Starting experiment '%s': %d total runs", self.config.name, total)

        for (model_id, quantization), model_configs in self._group_by_model().items():
            first_cfg = model_configs[0]
            model = first_cfg["model"]
            server_config = ServerConfig(
                engine=self.config.serving_engine,
                model_id=model_id,
                quantization=quantization,
                tp=model.tp,
                python_path=self.config.python_path,
                env_vars={"CUDA_VISIBLE_DEVICES": self.config.cuda_visible_devices}
                if self.config.cuda_visible_devices
                else {},
                extra_flags=[],
            )

            logger.info(
                "Launching %s for %s (q=%s)",
                server_config.engine,
                model_id,
                quantization,
            )

            with ModelServerManager(server_config) as server:
                if not server.wait_until_ready():
                    logger.error("Server did not become ready for %s (q=%s)", model_id, quantization)
                    continue

                client = ChatClient(base_url=f"http://localhost:{server.config.port}")
                for cfg in model_configs:
                    self._run_config(client, cfg["model"], cfg)

        logger.info("Experiment '%s' complete.", self.config.name)

    def _group_by_model(self) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for cfg in self.config.iter_configs():
            key = (cfg["model"].id, cfg["quantization"])
            groups.setdefault(key, []).append(cfg)
        return groups

    def _run_config(self, client: ChatClient, model: ModelConfig, cfg: Dict[str, Any]) -> None:
        for task in self.tasks:
            for rep in range(self.config.repetitions):
                self._run_single(client, model, cfg, task, rep)

    def _run_single(
        self,
        client: ChatClient,
        model: ModelConfig,
        cfg: Dict[str, Any],
        task: TaskDef,
        rep: int,
    ) -> None:
        run_id = self._make_run_id(model, cfg, task, rep)
        if self._run_exists(run_id):
            logger.debug("Skipping %s (already exists)", run_id)
            return

        messages = list(task.messages)
        if self.skills_loader is not None:
            skill_message = self.skills_loader(cfg["skill_subset"])
            if skill_message:
                messages = [{"role": "system", "content": skill_message}] + messages

        monitor = GPUMonitor(interval=0.5)
        monitor.start()
        started_at = datetime.now().isoformat()

        try:
            response = client.chat(
                messages=messages,
                model=model.id,
                temperature=cfg["temperature"],
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            monitor.stop()
            logger.error("Run %s failed: %s", run_id, exc)
            return

        gpu_stats = monitor.stop()
        completed_at = datetime.now().isoformat()

        task_success = None
        quality_score = None
        if task.eval_config:
            eval_cfg = EvalConfig.from_dict(task.eval_config)
            eval_result = evaluate(response.content, eval_cfg)
            task_success = eval_result.task_success
            quality_score = eval_result.quality_score
            logger.info(
                "  Eval: %s (score=%.1f) %s",
                "PASS" if eval_result.task_success else "FAIL",
                eval_result.quality_score,
                eval_result.explanation[:80],
            )

        result = RunResult(
            id=run_id,
            experiment_name=self.config.name,
            model_id=model.id,
            model_params_b=model.params_b,
            model_arch=model.arch,
            serving_engine=self.config.serving_engine,
            quantization=cfg["quantization"],
            tensor_parallel=model.tp,
            gpu_type=self.config.gpu_type,
            skill_subset=cfg["skill_subset"],
            num_retries=cfg["num_retries"],
            temperature=cfg["temperature"],
            agent_mode=cfg["agent_mode"],
            task_id=task.id,
            task_name=task.name,
            repetition=rep,
            task_success=task_success,
            quality_score=quality_score,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_e2e_ms=response.latency_ms,
            avg_gpu_util_pct=gpu_stats.avg_gpu_util_pct,
            avg_power_w=gpu_stats.avg_power_w,
            peak_vram_mb=gpu_stats.peak_memory_used_mb,
            gpu_seconds=gpu_stats.duration_s * (gpu_stats.avg_gpu_util_pct / 100.0),
            output_text=response.content,
            trajectory_log=json.dumps(response.raw_response, ensure_ascii=False),
            started_at=started_at,
            completed_at=completed_at,
        )

        self.store.save_run(result)
        if self._existing_run_ids is not None:
            self._existing_run_ids.add(run_id)

        logger.info(
            "  %s: %d+%d tokens, %.0fms",
            run_id,
            response.input_tokens,
            response.output_tokens,
            response.latency_ms,
        )

    def _run_exists(self, run_id: str) -> bool:
        if self._existing_run_ids is None:
            runs = self.store.get_runs(experiment_name=self.config.name)
            self._existing_run_ids = {run.id for run in runs}
        return run_id in self._existing_run_ids

    def _make_run_id(
        self,
        model: ModelConfig,
        cfg: Dict[str, Any],
        task: TaskDef,
        rep: int,
    ) -> str:
        run_key = "|".join(
            [
                self.config.name,
                model.id,
                str(cfg["quantization"]),
                str(cfg["skill_subset"]),
                str(cfg["num_retries"]),
                str(cfg["temperature"]),
                str(cfg["agent_mode"]),
                task.id,
                str(rep),
            ]
        )
        return f"run-{uuid.uuid5(uuid.NAMESPACE_URL, run_key).hex}"
