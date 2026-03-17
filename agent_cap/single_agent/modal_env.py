"""Modal-based workspace for agentic SWE-bench.

Same interface as DockerWorkspace but runs containers via Modal
(serverless). Works on RunPod and anywhere without local Docker.

Requires: pip install modal && modal setup
"""

import json
import logging
import subprocess
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent_cap.single_agent.modal_env")


class ModalWorkspace:
    """Run SWE-bench containers via Modal sandboxes."""

    def __init__(self, eval_config: Dict[str, Any], **kwargs):
        self.instance_id = eval_config.get("instance_id", "unknown")
        self.repo = eval_config.get("repo", "")
        self.base_commit = eval_config.get("base_commit", "")
        self.test_patch = eval_config.get("test_patch", "")
        self.fail_to_pass = eval_config.get(
            "FAIL_TO_PASS", eval_config.get("fail_to_pass", "")
        )
        self.dockerhub_tag = eval_config.get("dockerhub_tag", "")
        self.workdir = "/app"
        self.ready = False
        self._sandbox = None
        self._app = None

    @property
    def workspace(self) -> str:
        return self.workdir

    @property
    def container_id(self) -> Optional[str]:
        return self._sandbox.object_id if self._sandbox else None

    def setup(self) -> bool:
        try:
            import modal
        except ImportError:
            logger.error("modal not installed. Run: pip install modal && modal setup")
            return False

        if self.dockerhub_tag:
            image_ref = f"jefzda/sweap-images:{self.dockerhub_tag}"
        else:
            image_ref = f"sweb.eval.x86_64.{self.instance_id}:latest"

        logger.info(
            "[%s] Starting Modal sandbox from %s", self.instance_id[:30], image_ref
        )

        try:
            image = modal.Image.from_registry(image_ref)
            self._app = modal.App.lookup("agentcap-swebench", create_if_missing=True)
            self._sandbox = modal.Sandbox.create(
                image=image,
                app=self._app,
                workdir=self.workdir,
                timeout=600,
            )

            if self.test_patch:
                self._exec(
                    f"echo {json.dumps(self.test_patch)} | "
                    'python3 -c "import sys,json; '
                    "open('/tmp/test.patch','w').write(json.loads(sys.stdin.read()))\" "
                    "&& git apply /tmp/test.patch"
                )

            self.ready = True
            return True

        except Exception as exc:
            logger.error("[%s] Modal sandbox failed: %s", self.instance_id[:30], exc)
            return False

    def get_git_diff(self) -> str:
        result = self._exec("git diff")
        return result if result else ""

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
            output = self._exec(f"python -m pytest {test_target} -x --tb=short -q")
            ok = output is not None and "passed" in output.lower()
            if ok:
                passed_count += 1
            details.append(
                {
                    "test": test_spec[:100],
                    "passed": ok,
                    "output": (output or "")[-500:],
                }
            )

        return {
            "passed": passed_count == len(tests),
            "passed_count": passed_count,
            "total": len(tests),
            "details": details,
        }

    def cleanup(self):
        if self._sandbox:
            try:
                self._sandbox.terminate()
            except Exception:
                pass
            self._sandbox = None
        self.ready = False

    def _exec(self, cmd: str) -> Optional[str]:
        if not self._sandbox:
            return None
        try:
            process = self._sandbox.exec("bash", "-c", cmd)
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            return stdout + stderr
        except Exception as exc:
            return f"ERROR: {exc}"
