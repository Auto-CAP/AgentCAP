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
    def __init__(
        self, eval_config: Dict[str, Any], docker_hub_user: str = "", **kwargs
    ):
        self.instance_id = eval_config.get("instance_id", "unknown")
        self.repo = eval_config.get("repo", "")
        self.base_commit = eval_config.get("base_commit", "")
        self.test_patch = eval_config.get("test_patch", "")
        self.fail_to_pass = eval_config.get(
            "FAIL_TO_PASS", eval_config.get("fail_to_pass", "")
        )
        self.pass_to_pass = eval_config.get("pass_to_pass", "")
        self.before_repo_set_cmd = eval_config.get("before_repo_set_cmd", "")
        self.selected_test_files = eval_config.get("selected_test_files_to_run", "")
        self.dockerhub_tag = eval_config.get("dockerhub_tag", "")
        self.docker_hub_user = docker_hub_user
        self.container_id: Optional[str] = None
        self.container_name = f"agentcap-{self.instance_id[:50]}"
        self.workdir = "/app"
        self.ready = False

    @property
    def workspace(self) -> str:
        return self.workdir

    def setup(self) -> bool:
        self._remove_existing()

        if self.dockerhub_tag:
            image = f"jefzda/sweap-images:{self.dockerhub_tag}"
        elif self.docker_hub_user:
            image = f"{self.docker_hub_user}/sweb.eval.x86_64.{self.instance_id}:latest"
        else:
            image = f"sweb.eval.x86_64.{self.instance_id}:latest"

        logger.info("[%s] Pulling %s", self.instance_id[:30], image)
        pull_proc = subprocess.run(
            ["docker", "pull", image], capture_output=True, text=True, timeout=600
        )
        if pull_proc.returncode != 0:
            logger.warning(
                "[%s] Pull failed: %s", self.instance_id[:30], pull_proc.stderr[:200]
            )
            return False
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
                    "--entrypoint",
                    "/bin/bash",
                    image,
                    "-c",
                    "sleep infinity",
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

        self._docker_exec(
            "git init 2>/dev/null; "
            "git add -A 2>/dev/null; "
            "git -c user.email=bench@test -c user.name=bench commit -m baseline --allow-empty 2>/dev/null",
            timeout=30,
        )

        self.ready = True
        return True

    def get_git_diff(self) -> str:
        from agent_cap.backends.k8s_env import strip_binary_hunks
        self._docker_exec("git add -A", timeout=10)
        proc = self._docker_exec("git diff HEAD", timeout=10)
        if proc and proc.returncode == 0 and proc.stdout.strip():
            return strip_binary_hunks(proc.stdout.strip())
        proc = self._docker_exec("git diff --cached", timeout=10)
        if proc and proc.returncode == 0 and proc.stdout.strip():
            return strip_binary_hunks(proc.stdout.strip())
        proc = self._docker_exec("git diff", timeout=10)
        diff = proc.stdout.strip() if proc and proc.returncode == 0 else ""
        return strip_binary_hunks(diff)

    def run_tests(self, timeout: int = 300) -> Dict[str, Any]:
        """Run official evaluation — mirrors swe_bench_pro_eval.py exactly."""
        import ast, os
        from agent_cap.backends.k8s_env import _extract_env_cmds, SCRIPTS_DIR

        env_cmds = _extract_env_cmds(self.instance_id)

        before_cmd = ""
        if self.before_repo_set_cmd:
            before_cmd = self.before_repo_set_cmd.strip().split("\n")[-1]

        try:
            test_files = json.loads(self.selected_test_files) if self.selected_test_files else []
        except (json.JSONDecodeError, TypeError):
            test_files = []
        selected_str = ",".join(test_files)

        patch = self.get_git_diff()
        if not patch:
            return {"passed": False, "reason": "no patch to evaluate"}

        # Copy scripts into container
        scripts_base = os.path.join(SCRIPTS_DIR, self.instance_id)
        run_script = os.path.join(scripts_base, "run_script.sh")
        parser_py = os.path.join(scripts_base, "parser.py")

        if not os.path.exists(run_script) or not os.path.exists(parser_py):
            return {"passed": False, "reason": f"scripts missing for {self.instance_id}"}

        self._docker_exec("mkdir -p /workspace", timeout=5)

        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            f.write(patch)
            patch_tmp = f.name
        subprocess.run(["docker", "cp", patch_tmp, f"{self.container_id}:/workspace/patch.diff"],
                       capture_output=True, timeout=30)
        os.unlink(patch_tmp)
        subprocess.run(["docker", "cp", run_script, f"{self.container_id}:/workspace/run_script.sh"],
                       capture_output=True, timeout=30)
        subprocess.run(["docker", "cp", parser_py, f"{self.container_id}:/workspace/parser.py"],
                       capture_output=True, timeout=30)
        self._docker_exec("chmod +x /workspace/run_script.sh", timeout=5)

        # Build + run entry script (exact match of official)
        entry_script = f"""
{env_cmds}
cd /app
git reset --hard {self.base_commit}
git checkout {self.base_commit}
git apply -v /workspace/patch.diff
{before_cmd}
bash /workspace/run_script.sh {selected_str} > /workspace/stdout.log 2> /workspace/stderr.log
python3 /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
        self._docker_exec(entry_script, timeout=timeout)

        result_proc = self._docker_exec("cat /workspace/output.json", timeout=10)
        output_json = result_proc.stdout.strip() if result_proc and result_proc.returncode == 0 else ""

        try:
            parsed = json.loads(output_json) if output_json else {}
        except json.JSONDecodeError:
            parsed = {}

        passed_tests = {x["name"] for x in parsed.get("tests", []) if x["status"] == "PASSED"}

        try:
            f2p = set(json.loads(self.fail_to_pass)) if self.fail_to_pass else set()
        except (json.JSONDecodeError, TypeError):
            try:
                f2p = set(ast.literal_eval(self.fail_to_pass)) if self.fail_to_pass else set()
            except Exception:
                f2p = set()

        try:
            p2p = set(json.loads(self.pass_to_pass)) if self.pass_to_pass else set()
        except (json.JSONDecodeError, TypeError):
            try:
                p2p = set(ast.literal_eval(self.pass_to_pass)) if self.pass_to_pass else set()
            except Exception:
                p2p = set()

        resolved = (f2p | p2p) <= passed_tests if (f2p or p2p) else False

        return {
            "passed": resolved,
            "f2p_total": len(f2p),
            "p2p_total": len(p2p),
            "passed_count": len(passed_tests),
            "details": json.dumps(list(passed_tests)[:10]) if passed_tests else "[]",
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
