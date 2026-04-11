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
        self.before_repo_set_cmd = eval_config.get("before_repo_set_cmd", "")
        self.selected_test_files = eval_config.get("selected_test_files_to_run", "")
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
        # Same as DockerWorkspace: stage new files first, then diff
        self._exec("git add -A", timeout=10)
        proc = self._exec("git diff HEAD", timeout=10)
        if proc and proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        proc = self._exec("git diff --cached", timeout=10)
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

        # Use official run_script.sh + parser.py (same as DockerWorkspace)
        script_url = (
            f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/"
            f"run_scripts/{self.instance_id}/run_script.sh"
        )
        parser_url = (
            f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/"
            f"run_scripts/{self.instance_id}/parser.py"
        )

        self._exec(
            f"curl -sL '{script_url}' -o /run_script.sh && chmod +x /run_script.sh",
            timeout=30,
        )
        self._exec(f"curl -sL '{parser_url}' -o /parser.py", timeout=30)

        # Official evaluation flow (matches swe_bench_pro_eval.py exactly):
        # 1. Download run_script.sh and parser.py for this instance
        # 2. Run tests with selected_test_files, stdout/stderr to separate files
        # 3. Run parser.py <stdout> <stderr> <output.json>
        # 4. Read output.json to check results

        # Use selected_test_files_to_run (official) instead of fail_to_pass
        try:
            test_files = json.loads(self.selected_test_files) if self.selected_test_files else []
        except (json.JSONDecodeError, TypeError):
            test_files = []
        test_files_str = " ".join(test_files) if test_files else ",".join(
            t.split(" | ")[0].strip() for t in tests
        )

        self._exec(
            f"bash /run_script.sh {test_files_str} "
            f"> /workspace/stdout.log 2> /workspace/stderr.log",
            timeout=timeout,
        )

        self._exec(
            "python3 /parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json",
            timeout=30,
        )

        # Read output.json and check fail_to_pass tests
        result_proc = self._exec("cat /workspace/output.json", timeout=10)
        output_json = result_proc.stdout.strip() if result_proc and result_proc.returncode == 0 else ""

        ok = False
        details = ""
        try:
            parsed = json.loads(output_json) if output_json else {}
            test_results = parsed.get("tests", [])
            # Check if ALL fail_to_pass tests now pass
            fail_to_pass_names = set()
            for t in tests:
                # fail_to_pass format: "file | test_name" or just "test_name"
                fail_to_pass_names.add(t.strip())
            passed_names = {t["name"] for t in test_results if t.get("status") == "PASSED"}
            # A test passes if its name matches any fail_to_pass entry
            ok = all(
                any(fp in pn for pn in passed_names)
                for fp in fail_to_pass_names
            ) if fail_to_pass_names else False
            details = json.dumps(test_results[:10])
        except json.JSONDecodeError:
            details = output_json[:500]

        return {
            "passed": ok,
            "passed_count": 1 if ok else 0,
            "total": len(tests),
            "details": output[-1000:],
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
