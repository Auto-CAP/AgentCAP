"""Modal-based workspace for agentic SWE-bench.

Same interface as DockerWorkspace but runs containers via Modal
(serverless). Works on RunPod and anywhere without local Docker.

Requires: pip install modal && modal setup
"""

import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
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
        self.workdir = "/testbed"
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

        # Always use public GHCR images (Epoch AI) with instance_id
        image_ref = f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{self.instance_id}:latest"

        logger.info(
            "[%s] Starting Modal sandbox from %s", self.instance_id[:30], image_ref
        )

        try:

            def _create_sandbox():
                img = modal.Image.from_registry(image_ref)
                app = modal.App.lookup("agentcap-swebench", create_if_missing=True)
                return modal.Sandbox.create(
                    image=img, app=app, workdir=self.workdir, timeout=86400
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                self._sandbox = pool.submit(_create_sandbox).result(timeout=120)

            # Init git first, then apply test_patch, then commit baseline
            self._exec(
                "git init 2>/dev/null; "
                "git add -A 2>/dev/null; "
                "git -c user.email=bench@test -c user.name=bench commit -m pre-test-patch --allow-empty 2>/dev/null"
            )

            if self.test_patch:
                self._exec(
                    f"echo {json.dumps(self.test_patch)} | "
                    'python3 -c "import sys,json; '
                    "open('/tmp/test.patch','w').write(json.loads(sys.stdin.read()))\" "
                    "&& cd /testbed && git apply --allow-empty /tmp/test.patch"
                )

            self._exec(
                "git add -A 2>/dev/null; "
                "git -c user.email=bench@test -c user.name=bench commit -m baseline --allow-empty 2>/dev/null"
            )

            self.ready = True
            return True

        except Exception as exc:
            logger.error("[%s] Modal sandbox failed: %s", self.instance_id[:30], exc)
            return False

    def get_git_diff(self) -> str:
        result = self._exec("git diff HEAD")
        return result.strip() if result and "fatal" not in result else ""

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

        # Build test command based on repo type
        test_args = " ".join(f'"{t}"' for t in tests)
        instance = self.instance_id.lower()

        if "django" in instance:
            # Django uses its own test runner
            modules = sorted(set(
                t.split("(")[-1].rstrip(")").rsplit(".", 1)[0]
                for t in tests if "(" in t
            ))
            if not modules:
                modules = [t.split("::")[0].replace("/", ".").replace(".py", "") for t in tests]
            test_directive = " ".join(modules)
            test_cmd = (
                "source /opt/miniconda3/bin/activate testbed 2>/dev/null; "
                f"cd {self.workdir} && python tests/runtests.py --settings=test_sqlite --parallel 1 {test_directive}"
            )
        elif "sympy" in instance:
            # sympy uses bin/test
            test_cmd = (
                "source /opt/miniconda3/bin/activate testbed 2>/dev/null; "
                f"cd {self.workdir} && PYTHONWARNINGS=\'ignore::UserWarning\' bin/test -C --verbose {test_args}"
            )
        elif "sphinx" in instance:
            # sphinx uses tox
            test_cmd = (
                "source /opt/miniconda3/bin/activate testbed 2>/dev/null; "
                f"cd {self.workdir} && tox --current-env -epy39 -v -- {test_args}"
            )
        else:
            # Default: pytest
            test_cmd = (
                "source /opt/miniconda3/bin/activate testbed 2>/dev/null; "
                f"cd {self.workdir} && python -m pytest -x --tb=short {test_args}"
            )

        try:
            process = self._sandbox.exec(
                "bash", "-c",
                f"{test_cmd} > /test_stdout.txt 2>&1; echo EXIT_CODE=$?",
            )
            process.wait()

            cat_process = self._sandbox.exec("cat", "/test_stdout.txt")
            cat_process.wait()
            output = cat_process.stdout.read()

            # Parse pytest output
            ok = "passed" in output and "failed" not in output and "error" not in output.lower().split("passed")[0]
            # More robust: check exit code
            if "EXIT_CODE=0" in output:
                ok = True
            elif "EXIT_CODE=" in output:
                ok = False

        except Exception as exc:
            output = str(exc)
            ok = False

        return {
            "passed": ok,
            "passed_count": 1 if ok else 0,
            "total": len(tests),
            "details": [
                {
                    "test": ", ".join(tests)[:100],
                    "passed": ok,
                    "output": output[-500:] if output else "",
                }
            ],
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
            process.wait()
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            return stdout + stderr
        except Exception as exc:
            return f"ERROR: {exc}"
