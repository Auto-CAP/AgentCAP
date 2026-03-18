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

            def _create_sandbox():
                img = modal.Image.from_registry(image_ref)
                app = modal.App.lookup("agentcap-swebench", create_if_missing=True)
                return modal.Sandbox.create(
                    image=img, app=app, workdir=self.workdir, timeout=86400
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                self._sandbox = pool.submit(_create_sandbox).result(timeout=120)

            if self.test_patch:
                self._exec(
                    f"echo {json.dumps(self.test_patch)} | "
                    'python3 -c "import sys,json; '
                    "open('/tmp/test.patch','w').write(json.loads(sys.stdin.read()))\" "
                    "&& git apply /tmp/test.patch"
                )

            self._exec(
                "git init 2>/dev/null; "
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

        test_files = [t.split(" | ")[0].strip() for t in tests]

        base_url = (
            "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/"
            f"run_scripts/{self.instance_id}"
        )
        self._exec(
            f"(curl -sL '{base_url}/run_script.sh' || wget -qO- '{base_url}/run_script.sh') > /run_script.sh 2>/dev/null; "
            f"(curl -sL '{base_url}/parser.py' || wget -qO- '{base_url}/parser.py') > /parser.py 2>/dev/null; "
            "chmod +x /run_script.sh"
        )

        try:
            process = self._sandbox.exec(
                "bash",
                "-c",
                f"cd {self.workdir} && bash /run_script.sh {','.join(test_files)} "
                "> /test_stdout.txt 2> /test_stderr.txt; echo $?",
            )
            process.wait()

            parse_process = self._sandbox.exec(
                "python3",
                "/parser.py",
                "/test_stdout.txt",
                "/test_stderr.txt",
                "/test_results.json",
            )
            parse_process.wait()

            cat_process = self._sandbox.exec("cat", "/test_results.json")
            cat_process.wait()
            results_json = cat_process.stdout.read()

            tail_process = self._sandbox.exec("tail", "-30", "/test_stdout.txt")
            tail_process.wait()
            output = tail_process.stdout.read()[-500:]

            ok = False
            try:
                parsed = json.loads(results_json)
                test_results = parsed.get("tests", [])
                fail_to_pass_names = set()
                for t in tests:
                    fail_to_pass_names.add(t.strip())
                    fail_to_pass_names.add(t.split(" | ")[0].strip())

                for tr in test_results:
                    if tr["status"] == "PASSED":
                        for ftp in fail_to_pass_names:
                            if ftp in tr["name"]:
                                ok = True
                                break
            except (json.JSONDecodeError, KeyError):
                pass

        except Exception as exc:
            output = str(exc)
            ok = False

        return {
            "passed": ok,
            "passed_count": 1 if ok else 0,
            "total": 1,
            "details": [
                {
                    "test": ",".join(test_files)[:100],
                    "passed": ok,
                    "output": output[-500:],
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
