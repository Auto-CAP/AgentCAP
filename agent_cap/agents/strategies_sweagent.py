"""SWE-agent subprocess strategy.

Wraps the third-party `sweagent` CLI as a Strategy so the unified
`python -m agent_cap.agents` entrypoint covers SWE-bench Lite/Pro the
same way it covers MCP-Atlas. Produces the same per-task schema as
SingleAgentStrategy (input/output/completion/reasoning/cached_tokens,
ttft_ms, tpot_ms, latency_ms, num_turns, tool_calls).

Reads task config from `task.metadata["_unified_task"].eval_config`
(populated by unified_runner's swe_bench_lite loader). Spawns one
`python -m sweagent run` subprocess per task, parses traj for the
final patch, then reads stream_stats.jsonl (written by the patched
sweagent/agent/models.py) for usage + timing.

Required `--agent sweagent=...` keys:
    name:            litellm model id, e.g. openai/unsloth/gpt-oss-120b
    base_url:        OpenAI-compatible endpoint
    api_key:         (defaults to "dummy")

Strategy-level config goes on `RunResult.extras["sweagent_config"]`:
    deployment:      docker | modal | local | k8s    (default: docker)
    sweagent_dir:    path to swe-agent checkout      (default: /tmp/swe_agent)
    image_repo:      ""=local image, jefzda/sweap-images=registry, etc.
    per_instance_call_limit:                          (default: 200)
    output_dir:      per-task scratch dir            (required)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from agent_cap.agents.agent import Agent
from agent_cap.agents.registry import register_strategy
from agent_cap.agents.strategies import Strategy
from agent_cap.agents.tools import ToolProvider
from agent_cap.agents.types import RunResult, Task, Usage


def _swebench_image(instance_id: str, deployment: str, image_repo: str) -> str:
    iid = instance_id.lower().replace("/", "__")
    if deployment == "modal":
        return f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest"
    if image_repo:
        return f"swebench/sweb.eval.x86_64.{iid}"
    return f"sweb.eval.x86_64.{iid}:latest"


@register_strategy("sweagent")
class SWEAgentStrategy(Strategy):
    """Subprocess-driven SWE-agent runner.

    Doesn't use the `Agent`/LLM machinery — sweagent has its own loop.
    `agents["agent"]` is still required so we can read endpoint config.
    """

    required_roles = ("agent",)

    async def run(
        self,
        task: Task,
        agents: Dict[str, Agent],
        tools: Optional[ToolProvider] = None,
    ) -> RunResult:
        self.validate(agents)
        agent = agents["agent"]
        endpoint = agent.spec.endpoint

        meta = task.metadata or {}
        unified = meta.get("_unified_task")
        eval_cfg = getattr(unified, "eval_config", None) or {}
        if not eval_cfg.get("instance_id"):
            raise ValueError(
                "sweagent strategy requires task.metadata['_unified_task'].eval_config "
                "with 'instance_id' (swe-bench-lite dataset)"
            )

        cfg = meta.get("sweagent_config") or {}
        deployment = cfg.get("deployment", "docker")
        sweagent_dir = Path(cfg.get("sweagent_dir", "/tmp/swe_agent"))
        image_repo = cfg.get("image_repo", "")
        call_limit = int(cfg.get("per_instance_call_limit", 200))
        out_root = Path(cfg.get("output_dir") or "/tmp/sweagent_out")
        task_dir = out_root / f"task_{task.task_id.replace('/', '_')}"
        task_dir.mkdir(parents=True, exist_ok=True)

        instance_id = eval_cfg["instance_id"]
        image = _swebench_image(instance_id, deployment, image_repo)
        if deployment == "modal" and not image.startswith("docker.io/"):
            image = f"docker.io/{image_repo or 'swebench'}:{image}"

        ps_file = task_dir / "problem.txt"
        prompt = task.user_prompt
        cut = prompt.find("\n========== VERIFICATION")
        if cut > 0:
            prompt = prompt[:cut].rstrip() + "\n"
        ps_file.write_text(prompt)

        traj_dir = task_dir / "sweagent_traj"
        traj_dir.mkdir(parents=True, exist_ok=True)

        deploy_args = [
            "--env.deployment.type", deployment,
            "--env.deployment.image", image,
            "--env.repo.type", "preexisting",
            "--env.repo.repo_name", "testbed",
        ]
        if deployment == "modal":
            deploy_args += [
                "--env.deployment.deployment_timeout", "14400",
                "--env.deployment.runtime_timeout", "900",
            ]

        cmd = [
            sys.executable, "-m", "sweagent", "run",
            "--config", str(sweagent_dir / "config" / "default.yaml"),
            "--agent.model.name", endpoint.name,
            "--agent.model.api_base", endpoint.base_url.rstrip("/"),
            "--agent.model.per_instance_cost_limit", "0",
            "--agent.model.total_cost_limit", "0",
            "--agent.model.per_instance_call_limit", str(call_limit),
            "--agent.model.completion_kwargs",
            '{"extra_body": {"stop_token_ids": [200012, 200002]}}',
            "--agent.templates.put_demos_in_history", "false",
            "--problem_statement.path", str(ps_file),
            "--output_dir", str(traj_dir),
        ] + deploy_args

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = endpoint.api_key or "dummy"
        stats_path = task_dir / "stream_stats.jsonl"
        env["SWEAGENT_STREAM_STATS_PATH"] = str(stats_path)

        t0 = time.perf_counter()
        # Run the blocking SWE-agent subprocess off the asyncio event loop.
        # The outer AgentCAP CLI uses asyncio.gather + Semaphore(concurrency);
        # calling subprocess.run() directly here serializes all tasks despite
        # --concurrency > 1.
        r = await asyncio.to_thread(
            subprocess.run,
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=int(cfg.get("subprocess_timeout", 1800)),
            cwd=str(sweagent_dir),
        )
        elapsed = time.perf_counter() - t0
        (task_dir / "sweagent_stdout.log").write_text(r.stdout or "")
        (task_dir / "sweagent_stderr.log").write_text(r.stderr or "")

        patch = ""
        tool_calls_count = 0
        for tf in sorted(traj_dir.rglob("*.traj"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                traj = json.loads(tf.read_text())
                p = traj.get("info", {}).get("submission") or traj.get("info", {}).get("model_patch") or ""
                hist = traj.get("history") or traj.get("trajectory") or []
                tool_calls_count = sum(len(m.get("tool_calls") or []) for m in hist)
                if p:
                    patch = p
                    break
            except Exception:
                continue
        if patch:
            (task_dir / "patch.diff").write_text(patch)

        prompt_total = visible_total = reasoning_total = cached_total = 0
        ttft_ms_first = 0.0
        decode_ms_sum = 0.0
        requests = 0
        if stats_path.exists():
            for line in stats_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_total += int(s.get("prompt_tokens") or 0)
                visible_total += int(s.get("completion_tokens") or 0)
                reasoning_total += int(s.get("reasoning_tokens") or 0)
                cached_total += int(s.get("cached_tokens") or 0)
                if requests == 0 and s.get("ttft_ms"):
                    ttft_ms_first = float(s["ttft_ms"])
                tpot = float(s.get("tpot_ms") or 0.0)
                out_tok = int(s.get("total_output_tokens") or 0)
                if tpot > 0 and out_tok > 0:
                    decode_ms_sum += tpot * out_tok
                requests += 1

        usage = Usage(
            input_tokens=prompt_total,
            output_tokens=visible_total + reasoning_total,
            completion_tokens=visible_total,
            reasoning_tokens=reasoning_total,
            cached_tokens=cached_total,
            requests=requests,
        )
        result = RunResult(
            task_id=task.task_id,
            strategy="sweagent",
            output_text=patch,
            e2e_latency_s=elapsed,
            per_role_usage={"agent": usage},
            errors=[] if patch or r.returncode == 0 else [f"sweagent rc={r.returncode}"],
        )
        result.extras["has_patch"] = bool(patch)
        result.extras["sweagent_rc"] = r.returncode
        result.extras["deployment"] = deployment
        result.extras["instance_id"] = instance_id
        result.extras["image"] = image
        result.extras["traj_file"] = str(traj_dir)
        result.extras["tool_calls"] = tool_calls_count
        result.extras["ttft_ms_first"] = ttft_ms_first
        if usage.output_tokens > 0:
            result.extras["tpot_ms"] = decode_ms_sum / usage.output_tokens
        else:
            result.extras["tpot_ms"] = 0.0
        return result
