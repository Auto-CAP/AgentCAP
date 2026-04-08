import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("agent_cap.backends.local_env")

_GIT = "/usr/bin/git" if os.path.exists("/usr/bin/git") else "git"


class LocalWorkspace:
    def __init__(
        self,
        eval_config: Dict[str, Any],
        base_dir: str = "/tmp/agentcap_workspaces",
        **kwargs,
    ):
        self.instance_id = eval_config.get("instance_id", "unknown")
        self.repo = eval_config.get("repo", "")
        self.base_commit = eval_config.get("base_commit", "")
        self.test_patch = eval_config.get("test_patch", "")
        self.fail_to_pass = eval_config.get(
            "FAIL_TO_PASS", eval_config.get("fail_to_pass", "")
        )
        self.base_dir = Path(base_dir)
        safe_id = self.instance_id.replace("/", "_").replace(" ", "_")[:100]
        self.workdir = str(self.base_dir / safe_id)
        self.ready = False

    @property
    def workspace(self) -> str:
        return self.workdir

    @property
    def container_id(self) -> Optional[str]:
        return None

    def setup(self) -> bool:
        workspace_path = Path(self.workdir)

        if workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)
        workspace_path.mkdir(parents=True, exist_ok=True)

        repo_url = f"https://github.com/{self.repo}.git"
        logger.info("[%s] Cloning %s", self.instance_id[:30], repo_url)
        proc = subprocess.run(
            [_GIT, "clone", "--depth", "100", repo_url, str(workspace_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            logger.warning(
                "[%s] Shallow clone failed, trying full clone", self.instance_id[:30]
            )
            shutil.rmtree(workspace_path, ignore_errors=True)
            workspace_path.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [_GIT, "clone", repo_url, str(workspace_path)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode != 0:
                logger.error(
                    "[%s] Clone failed: %s", self.instance_id[:30], proc.stderr[:300]
                )
                return False

        if self.base_commit:
            proc = subprocess.run(
                [_GIT, "checkout", self.base_commit],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(workspace_path),
            )
            if proc.returncode != 0:
                logger.warning(
                    "[%s] Checkout failed on shallow clone, trying full clone",
                    self.instance_id[:30],
                )
                shutil.rmtree(workspace_path, ignore_errors=True)
                workspace_path.mkdir(parents=True, exist_ok=True)
                proc = subprocess.run(
                    [_GIT, "clone", repo_url, str(workspace_path)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if proc.returncode != 0:
                    logger.error(
                        "[%s] Full clone failed: %s",
                        self.instance_id[:30],
                        proc.stderr[:300],
                    )
                    return False
                proc = subprocess.run(
                    [_GIT, "checkout", self.base_commit],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(workspace_path),
                )
                if proc.returncode != 0:
                    logger.error(
                        "[%s] Checkout failed: %s",
                        self.instance_id[:30],
                        proc.stderr[:200],
                    )
                    return False

        if self.test_patch and str(self.test_patch).strip():
            patch_file = workspace_path / ".test_patch.diff"
            patch_file.write_text(self.test_patch)
            subprocess.run(
                [_GIT, "apply", str(patch_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(workspace_path),
            )

        subprocess.run(
            [_GIT, "add", "-A"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(workspace_path),
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=bench@test",
                "-c",
                "user.name=bench",
                "commit",
                "-m",
                "baseline",
                "--allow-empty",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(workspace_path),
        )

        self.ready = True
        logger.info(
            "[%s] Local workspace ready at %s", self.instance_id[:30], self.workdir
        )
        return True

    def get_git_diff(self) -> str:
        proc = subprocess.run(
            [_GIT, "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=self.workdir,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        proc = subprocess.run(
            [_GIT, "diff"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=self.workdir,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

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

        test_files_str = ",".join(t.split(" | ")[0].strip() for t in tests)

        script_url = (
            f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/"
            f"run_scripts/{self.instance_id}/run_script.sh"
        )
        parser_url = (
            f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/"
            f"run_scripts/{self.instance_id}/parser.py"
        )

        subprocess.run(
            [
                "bash",
                "-c",
                f"curl -sL '{script_url}' -o /tmp/run_script.sh && chmod +x /tmp/run_script.sh",
            ],
            timeout=30,
        )
        subprocess.run(
            ["bash", "-c", f"curl -sL '{parser_url}' -o /tmp/parser.py"],
            timeout=30,
        )

        proc = subprocess.run(
            ["bash", "/tmp/run_script.sh", test_files_str],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.workdir,
        )
        output = (proc.stdout + proc.stderr)[-2000:]

        parse_proc = subprocess.run(
            [
                "python3",
                "/tmp/parser.py",
                "--log",
                output[-8000:],
                "--expected",
                json.dumps(tests),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.workdir,
        )
        parse_output = parse_proc.stdout if parse_proc else ""
        ok = "PASS" in parse_output.upper() if parse_output else False

        return {
            "passed": ok,
            "passed_count": 1 if ok else 0,
            "total": len(tests),
            "details": [
                {
                    "test": test_files_str[:100],
                    "passed": ok,
                    "output": output[-500:],
                }
            ],
        }

    def cleanup(self):
        workspace_path = Path(self.workdir)
        if workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)
        self.ready = False
