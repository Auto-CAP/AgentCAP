"""Local environment setup for SWE-bench without Docker.

Replicates what the SWE-bench Docker harness does:
1. git clone repo at base_commit
2. conda create environment with correct Python + dependencies
3. pip install the repo
4. Apply test patch
5. Run tests with the repo's test_cmd

Requires: conda, git
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent_cap.single_agent.local_env")

MAP_REPO_VERSION_TO_SPECS = None


def _load_specs():
    global MAP_REPO_VERSION_TO_SPECS
    if MAP_REPO_VERSION_TO_SPECS is not None:
        return
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS as _specs

        MAP_REPO_VERSION_TO_SPECS = _specs
    except ImportError:
        MAP_REPO_VERSION_TO_SPECS = {}
        logger.warning("swebench not installed — using fallback specs")


def _run(
    cmd: str, cwd: str = None, timeout: int = 300, env_name: str = None
) -> subprocess.CompletedProcess:
    if env_name:
        cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && {cmd}"
    return subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


class LocalWorkspace:
    """A local conda-based workspace for one SWE-bench instance."""

    def __init__(
        self, instance: Dict[str, Any], base_dir: str = "/tmp/agent_cap_workspaces"
    ):
        self.instance_id = instance.get("instance_id", "unknown")
        self.repo = instance.get("repo", "")
        self.base_commit = instance.get("base_commit", "")
        self.version = instance.get("version", "")
        self.test_patch = instance.get("test_patch", "")
        self.fail_to_pass = instance.get(
            "FAIL_TO_PASS", instance.get("fail_to_pass", "")
        )

        self.workspace = Path(base_dir) / self.instance_id.replace("/", "_")
        self.env_name = (
            f"testbed_{self.instance_id.replace('/', '_').replace('-', '_')[:30]}"
        )
        self.ready = False

    def setup(self) -> bool:
        _load_specs()

        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[%s] Cloning %s@%s",
            self.instance_id[:30],
            self.repo,
            self.base_commit[:12],
        )
        proc = _run(
            f"git clone --depth=50 https://github.com/{self.repo}.git {self.workspace}",
            timeout=120,
        )
        if proc.returncode != 0:
            logger.error("Clone failed: %s", proc.stderr[:300])
            return False

        _run(f"git reset --hard {self.base_commit}", cwd=str(self.workspace))
        _run("git remote remove origin", cwd=str(self.workspace))

        specs = {}
        if MAP_REPO_VERSION_TO_SPECS and self.repo in MAP_REPO_VERSION_TO_SPECS:
            version_specs = MAP_REPO_VERSION_TO_SPECS[self.repo]
            specs = version_specs.get(self.version, {})

        python_version = specs.get("python", "3.10")
        packages = specs.get("packages", "")
        install_cmd = specs.get("install", "pip install -e .")
        pip_packages = specs.get("pip_packages", [])

        logger.info(
            "[%s] Creating conda env (python=%s)", self.instance_id[:30], python_version
        )
        _run(f"conda create -n {self.env_name} python={python_version} -y", timeout=120)

        if packages and packages not in ("requirements.txt", "environment.yml"):
            _run(f"conda install -n {self.env_name} {packages} -y", timeout=120)

        if pip_packages:
            _run(
                f"pip install {' '.join(pip_packages)}",
                cwd=str(self.workspace),
                env_name=self.env_name,
                timeout=120,
            )

        logger.info("[%s] Installing repo: %s", self.instance_id[:30], install_cmd)
        proc = _run(
            install_cmd, cwd=str(self.workspace), env_name=self.env_name, timeout=300
        )
        if proc.returncode != 0:
            logger.warning("Install had errors: %s", proc.stderr[:300])

        _run("git config user.email setup@swebench.config", cwd=str(self.workspace))
        _run("git config user.name SWE-bench", cwd=str(self.workspace))
        _run("git commit --allow-empty -am SWE-bench-setup", cwd=str(self.workspace))

        self.ready = True
        return True

    def apply_test_patch(self) -> bool:
        if not self.test_patch:
            return True
        proc = _run(
            f"git apply -v -",
            cwd=str(self.workspace),
            timeout=30,
        )
        escaped = self.test_patch.replace("'", "'\\''")
        proc = subprocess.run(
            ["bash", "-c", f"cd {self.workspace} && echo '{escaped}' | git apply -v -"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.returncode == 0

    def get_git_diff(self) -> str:
        proc = _run("git diff", cwd=str(self.workspace))
        return proc.stdout.strip() if proc.returncode == 0 else ""

    def run_tests(self, timeout: int = 300) -> Dict[str, Any]:
        _load_specs()

        if not self.fail_to_pass:
            return {"passed": False, "reason": "no fail_to_pass tests"}

        try:
            tests = (
                json.loads(self.fail_to_pass)
                if isinstance(self.fail_to_pass, str)
                else self.fail_to_pass
            )
        except json.JSONDecodeError:
            tests = [self.fail_to_pass]

        specs = {}
        if MAP_REPO_VERSION_TO_SPECS and self.repo in MAP_REPO_VERSION_TO_SPECS:
            specs = MAP_REPO_VERSION_TO_SPECS[self.repo].get(self.version, {})
        test_cmd = specs.get("test_cmd", "python -m pytest")

        self.apply_test_patch()

        passed_count = 0
        details = []
        for test_spec in tests:
            test_target = test_spec.split("::")[0].split(" | ")[0].strip()
            cmd = f"{test_cmd} {test_target} -x --tb=short -q"
            proc = _run(
                cmd, cwd=str(self.workspace), env_name=self.env_name, timeout=timeout
            )
            ok = proc.returncode == 0
            if ok:
                passed_count += 1
            details.append(
                {
                    "test": test_spec[:100],
                    "passed": ok,
                    "output": (proc.stdout + proc.stderr)[-500:],
                }
            )

        return {
            "passed": passed_count == len(tests),
            "passed_count": passed_count,
            "total": len(tests),
            "details": details,
        }

    def cleanup(self):
        _run(f"conda env remove -n {self.env_name} -y", timeout=60)
        if self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)
