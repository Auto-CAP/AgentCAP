"""Docker-based workspace for agentic SWE-bench.

Uses pre-built SWE-bench Docker images (sweb.eval.x86_64.{instance_id}).
All tool calls and tests run inside the container via ``docker exec``.

Build images first:
    python -c "
    from swebench.harness.docker_build import build_env_images
    import docker
    from datasets import load_dataset
    client = docker.from_env()
    ds = load_dataset('princeton-nlp/SWE-bench_oracle', split='test')
    build_env_images(client, list(ds), max_workers=4)
    "
"""

import json
import logging
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent_cap.single_agent.docker_env")


class DockerWorkspace:
    def __init__(self, eval_config: Dict[str, Any], **kwargs):
        self.instance_id = eval_config.get("instance_id", "unknown")
        self.repo = eval_config.get("repo", "")
        self.base_commit = eval_config.get("base_commit", "")
        self.test_patch = eval_config.get("test_patch", "")
        self.fail_to_pass = eval_config.get(
            "FAIL_TO_PASS", eval_config.get("fail_to_pass", "")
        )
        self.container_id: Optional[str] = None
        self.container_name = f"agentcap-{self.instance_id[:50]}"
        self.workdir = "/testbed"
        self.ready = False

    @property
    def workspace(self) -> str:
        return self.workdir

    def setup(self) -> bool:
        self._remove_existing()

        image = f"sweb.eval.x86_64.{self.instance_id}:latest"
        logger.info("[%s] Starting container from %s", self.instance_id[:30], image)

        try:
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    self.container_name,
                    "-w",
                    self.workdir,
                    image,
                    "sleep",
                    "infinity",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode != 0:
                logger.error("Container start failed: %s", proc.stderr[:300])
                return False

            self.container_id = proc.stdout.strip()
        except Exception as exc:
            logger.error("Docker run failed: %s", exc)
            return False

        if self.test_patch:
            self._docker_exec(
                f"echo {json.dumps(self.test_patch)} | "
                'python3 -c "import sys,json; '
                "open('/tmp/test.patch','w').write(json.loads(sys.stdin.read()))\" "
                "&& git apply /tmp/test.patch",
                timeout=30,
            )

        self.ready = True
        return True

    def get_git_diff(self) -> str:
        proc = self._docker_exec("git diff", timeout=10)
        return proc.stdout.strip() if proc and proc.returncode == 0 else ""

    def run_tests(self, timeout: int = 300) -> Dict[str, Any]:
        if not self.fail_to_pass:
            return {"passed": False, "reason": "no tests defined"}

        try:
            tests = (
                json.loads(self.fail_to_pass)
                if isinstance(self.fail_to_pass, str)
                else self.fail_to_pass
            )
        except json.JSONDecodeError:
            tests = [self.fail_to_pass]

        passed_count = 0
        details: List[Dict[str, Any]] = []

        for test_spec in tests:
            test_target = test_spec.split("::")[0].split(" | ")[0].strip()
            proc = self._docker_exec(
                f"python -m pytest {test_target} -x --tb=short -q",
                timeout=timeout,
            )
            ok = proc is not None and proc.returncode == 0
            if ok:
                passed_count += 1
            output = ""
            if proc:
                output = (proc.stdout + proc.stderr)[-500:]
            details.append({"test": test_spec[:100], "passed": ok, "output": output})

        return {
            "passed": passed_count == len(tests),
            "passed_count": passed_count,
            "total": len(tests),
            "details": details,
        }

    def cleanup(self):
        self._remove_existing()

    def _remove_existing(self):
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
            timeout=15,
        )
        self.container_id = None
        self.ready = False

    def _docker_exec(
        self, cmd: str, timeout: int = 30
    ) -> Optional[subprocess.CompletedProcess]:
        if not self.container_id:
            return None
        try:
            return subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    self.workdir,
                    self.container_id,
                    "bash",
                    "-c",
                    cmd,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None
