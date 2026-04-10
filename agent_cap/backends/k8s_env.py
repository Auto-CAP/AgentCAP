"""Kubernetes sidecar-based workspace for agentic SWE-bench.

Same interface as DockerWorkspace but executes commands in a sidecar
container via an HTTP exec server running on port 9999.

The SWE-bench image runs as a sidecar with a tiny HTTP exec server.
The runner sends commands over localhost HTTP — no RBAC, no kubectl,
no privileged needed.
"""

import json
import logging
import os
import subprocess
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger("agent_cap.backends.k8s_env")


class K8sWorkspace:
    """Run SWE-bench tasks in a K8s sidecar container via HTTP exec."""

    def __init__(self, eval_config: Dict[str, Any], **kwargs):
        self.instance_id = eval_config.get("instance_id", "unknown")
        self.repo = eval_config.get("repo", "")
        self.base_commit = eval_config.get("base_commit", "")
        self.test_patch = eval_config.get("test_patch", "")
        self.fail_to_pass = eval_config.get(
            "FAIL_TO_PASS", eval_config.get("fail_to_pass", "")
        )
        self.workdir = "/app"
        self.ready = False
        self.container_id = None
        self.exec_url = os.environ.get("SWEBENCH_EXEC_URL", "http://localhost:9999/exec")

    @property
    def workspace(self) -> str:
        return self.workdir

    def setup(self) -> bool:
        probe = self._exec("echo ready", timeout=15)
        if not probe or probe.returncode != 0:
            logger.error("Cannot reach sidecar exec server at %s", self.exec_url)
            return False
        logger.info("Sidecar exec server OK")

        if self.test_patch:
            self._exec(
                f"echo {json.dumps(self.test_patch)} | "
                'python3 -c "import sys,json; '
                "open('/tmp/test.patch','w').write(json.loads(sys.stdin.read()))\" "
                "&& git apply /tmp/test.patch",
                timeout=30,
            )

        self._exec(
            "git init 2>/dev/null; "
            "git add -A 2>/dev/null; "
            "git -c user.email=bench@test -c user.name=bench "
            "commit -m baseline --allow-empty 2>/dev/null",
            timeout=30,
        )

        self.ready = True
        return True

    def get_git_diff(self) -> str:
        proc = self._exec("git diff HEAD", timeout=10)
        if proc and proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        proc = self._exec("git diff", timeout=10)
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

        # Run build.sh if present (installs deps, starts services like Redis)
        build_proc = self._exec(
            "test -f /build.sh && bash /build.sh 2>&1 || echo 'no build.sh'",
            timeout=120,
        )
        if build_proc:
            print(f"[run_tests] build.sh: rc={build_proc.returncode}", flush=True)

        # Run tests
        proc = self._exec("npm test 2>&1", timeout=timeout)
        output = (proc.stdout + proc.stderr)[-4000:] if proc else ""
        rc = proc.returncode if proc else 1

        return {
            "passed": rc == 0,
            "exit_code": rc,
            "total": len(tests),
            "test_output": output[-1000:],
        }

    def cleanup(self) -> None:
        self.ready = False

    def _exec_write_file(self, path: str, content: str) -> None:
        """Write content to a file in the sidecar, using base64 to avoid quoting."""
        import base64
        b64 = base64.b64encode(content.encode()).decode()
        self._exec(f"echo '{b64}' | base64 -d > {path}", timeout=10)

    def _exec(
        self, cmd: str, timeout: int = 30
    ) -> Optional[subprocess.CompletedProcess]:
        """Execute a command in the sidecar via HTTP exec server."""
        full_cmd = f"cd {self.workdir} && {cmd}"
        payload = json.dumps({"cmd": full_cmd, "timeout": timeout}).encode()
        req = urllib.request.Request(
            self.exec_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
                result = json.loads(resp.read())
            return subprocess.CompletedProcess(
                args=full_cmd,
                returncode=result.get("returncode", 1),
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
            )
        except Exception as exc:
            logger.error("HTTP exec failed: %s", exc)
            return None
