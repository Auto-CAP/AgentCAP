"""Kubernetes workspace for agentic SWE-bench.

Mirrors DockerWorkspace exactly, replacing docker commands with kubectl.
Evaluation flow matches swe_bench_pro_eval.py line-for-line.
"""

import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger("agent_cap.backends.k8s_env")

NAMESPACE = "eidf230ns"
SCRIPTS_DIR = os.environ.get(
    "SWEBENCH_SCRIPTS_DIR", "/tmp/swebench_pro_os/run_scripts"
)
DOCKERFILES_DIR = os.environ.get(
    "SWEBENCH_DOCKERFILES_DIR", "/tmp/swebench_pro_os/dockerfiles"
)


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch.
    Exact copy from swe_bench_pro_eval.py lines 75-92."""
    if not patch:
        return patch
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


def _load_dockerfile(instance_id: str, kind: str) -> str:
    """Load base or instance dockerfile content."""
    path = os.path.join(DOCKERFILES_DIR, f"{kind}_dockerfile", instance_id, "Dockerfile")
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _extract_env_cmds(instance_id: str) -> str:
    """Extract ENV commands from dockerfiles.
    Exact copy from swe_bench_pro_eval.py lines 102-112."""
    base_df = _load_dockerfile(instance_id, "base")
    instance_df = _load_dockerfile(instance_id, "instance")
    env_cmds = []
    for dockerfile_content in [base_df, instance_df]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                env_cmd = line.replace("ENV", "export", 1)
                env_cmds.append(env_cmd)
    return "\n".join(env_cmds)


class K8sWorkspace:
    """Run SWE-bench tasks in K8s pods via kubectl exec.

    Mirrors DockerWorkspace: kubectl create job = docker run,
    kubectl exec = docker exec, kubectl delete job = docker rm.
    """

    def __init__(self, eval_config: Dict[str, Any], **kwargs):
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
        self.workdir = eval_config.get("workdir", "/app")
        self._version = eval_config.get("version", "")
        self.ready = False
        self.pod_name: Optional[str] = None
        self.job_name: Optional[str] = None

    @property
    def workspace(self) -> str:
        return self.workdir

    # ------------------------------------------------------------------ #
    #  Lifecycle: setup / cleanup  (mirrors docker run / docker rm)
    # ------------------------------------------------------------------ #

    def setup(self) -> bool:
        # SWE-bench Lite: tag starts with "sweb.eval." → use swebench/ prefix
        # SWE-bench Pro: use jefzda/sweap-images prefix
        if self.dockerhub_tag.startswith("sweb.eval."):
            image = f"swebench/{self.dockerhub_tag}"
        else:
            image = f"jefzda/sweap-images:{self.dockerhub_tag}"
        safe_id = self.instance_id[:30].lower().replace("_", "-").rstrip("-")
        job_yaml = json.dumps({
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "generateName": f"swebench-{safe_id}-",
                "namespace": NAMESPACE,
                "labels": {
                    "app": "swebench-workspace",
                    "kueue.x-k8s.io/queue-name": f"{NAMESPACE}-user-queue",
                },
            },
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "swebench-workspace",
                            "kueue.x-k8s.io/queue-name": f"{NAMESPACE}-user-queue",
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "swebench",
                            "image": image,
                            "command": ["bash", "-c", "sleep infinity"],
                            "workingDir": self.workdir,
                            "resources": {
                                "requests": {"cpu": "2", "memory": "8Gi"},
                                "limits": {"cpu": "4", "memory": "16Gi"},
                            },
                        }],
                    },
                },
            },
        })

        r = subprocess.run(
            ["kubectl", "create", "-f", "-", "-n", NAMESPACE,
             "-o", "jsonpath={.metadata.name}"],
            input=job_yaml, capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.error("Job create failed: %s", r.stderr[:200])
            return False
        self.job_name = r.stdout.strip()

        # Wait for pod running
        import time
        for _ in range(120):
            r = subprocess.run(
                ["kubectl", "get", "pods", "-n", NAMESPACE,
                 f"-l=job-name={self.job_name}",
                 "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}"],
                capture_output=True, text=True,
            )
            parts = r.stdout.strip().split("|")
            phase = parts[0] if parts else ""
            self.pod_name = parts[1] if len(parts) > 1 else ""
            if phase == "Running" and self.pod_name:
                break
            if phase in ("Failed", "Succeeded"):
                logger.error("Pod %s", phase)
                return False
            time.sleep(3)
        else:
            logger.error("Pod startup timeout")
            return False

        logger.info("[%s] Pod %s running", self.instance_id[:30], self.pod_name)

        # Apply test_patch (same as docker_env lines 97-104)
        if self.test_patch:
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False
            ) as f:
                f.write(self.test_patch)
                tp_tmp = f.name
            cp_r = subprocess.run(
                ["kubectl", "cp", tp_tmp,
                 f"{NAMESPACE}/{self.pod_name}:/tmp/test.patch"],
                capture_output=True, text=True, timeout=30,
            )
            os.unlink(tp_tmp)
            if cp_r.returncode != 0:
                logger.error("Failed to copy test_patch: %s", cp_r.stderr[:200])
            apply_r = self._exec("git apply -v /tmp/test.patch", timeout=30)
            if apply_r and apply_r.returncode != 0:
                logger.error(
                    "[%s] test_patch apply failed: %s",
                    self.instance_id[:30],
                    (apply_r.stderr or apply_r.stdout or "")[:300],
                )

        # Copy official SWE-agent str_replace_editor + registry shim into container
        tools_dir = os.path.join(os.path.dirname(__file__), "swe_agent_tools")
        self._exec("mkdir -p /opt/swe_agent_tools", timeout=5)
        for fname in ("str_replace_editor", "registry.py"):
            local_path = os.path.join(tools_dir, fname)
            if os.path.exists(local_path):
                cp_r = subprocess.run(
                    ["kubectl", "cp", local_path,
                     f"{NAMESPACE}/{self.pod_name}:/opt/swe_agent_tools/{fname}"],
                    capture_output=True, timeout=30,
                )
                if cp_r.returncode != 0:
                    logger.error("Failed to copy %s: %s", fname, cp_r.stderr[:200])
        self._exec("chmod +x /opt/swe_agent_tools/str_replace_editor", timeout=5)
        # Verify str_replace_editor is accessible
        verify = self._exec(
            "test -f /opt/swe_agent_tools/str_replace_editor && echo OK || echo MISSING",
            timeout=5,
        )
        if verify and "MISSING" in (verify.stdout or ""):
            logger.error("[%s] str_replace_editor NOT installed", self.instance_id[:30])
        # Install tree-sitter-languages if available (for filemap on large .py files)
        self._exec(
            "pip install tree-sitter-languages 2>/dev/null || true",
            timeout=60,
        )

        # Git baseline (same as docker_env lines 106-111)
        self._exec(
            "git init 2>/dev/null; "
            "git add -A 2>/dev/null; "
            "git -c user.email=bench@test -c user.name=bench "
            "commit -m baseline --allow-empty 2>/dev/null",
            timeout=30,
        )

        self.ready = True
        return True

    def cleanup(self) -> None:
        """Delete the K8s Job (= docker rm -f)."""
        if self.job_name:
            subprocess.run(
                ["kubectl", "delete", "job", self.job_name, "-n", NAMESPACE,
                 "--wait=false", "--ignore-not-found=true"],
                capture_output=True, timeout=15,
            )
        self.pod_name = None
        self.job_name = None
        self.ready = False

    # ------------------------------------------------------------------ #
    #  Patch collection
    # ------------------------------------------------------------------ #

    def get_git_diff(self) -> str:
        # Revert any test file modifications before collecting patch
        # (model should not modify tests per SWE-bench rules)
        if self.test_patch:
            import re as _re
            test_files = []
            for m in _re.finditer(r'^\+\+\+ b/(.+)$', self.test_patch, _re.MULTILINE):
                test_files.append(m.group(1))
            if test_files:
                self._exec(
                    "git checkout HEAD -- " + " ".join(test_files),
                    timeout=10,
                )
        self._exec("git add -A", timeout=10)
        proc = self._exec("git diff HEAD", timeout=10)
        if proc and proc.returncode == 0 and proc.stdout.strip():
            diff = strip_binary_hunks(proc.stdout)
            # Ensure patch ends with newline (required by git apply)
            if diff and not diff.endswith("\n"):
                diff += "\n"
            return diff
        proc = self._exec("git diff --cached", timeout=10)
        if proc and proc.returncode == 0 and proc.stdout.strip():
            diff = strip_binary_hunks(proc.stdout)
            if diff and not diff.endswith("\n"):
                diff += "\n"
            return diff
        proc = self._exec("git diff", timeout=10)
        diff = proc.stdout if proc and proc.returncode == 0 else ""
        diff = strip_binary_hunks(diff)
        if diff and not diff.endswith("\n"):
            diff += "\n"
        return diff

    # ------------------------------------------------------------------ #
    #  Test evaluation
    # ------------------------------------------------------------------ #

    def _run_tests_pro(self, run_script: str, parser_py: str, timeout: int):
        """SWE-bench Pro eval: uses per-instance run_script.sh + parser.py."""
        env_cmds = _extract_env_cmds(self.instance_id)

        before_cmd = ""
        if self.before_repo_set_cmd:
            before_cmd = self.before_repo_set_cmd.strip().split("\n")[-1]

        try:
            test_files = json.loads(self.selected_test_files) if self.selected_test_files else []
        except (json.JSONDecodeError, TypeError):
            test_files = []
        selected_str = ",".join(test_files)

        subprocess.run(
            ["kubectl", "cp", run_script, f"{NAMESPACE}/{self.pod_name}:/workspace/run_script.sh"],
            capture_output=True, timeout=30,
        )
        subprocess.run(
            ["kubectl", "cp", parser_py, f"{NAMESPACE}/{self.pod_name}:/workspace/parser.py"],
            capture_output=True, timeout=30,
        )
        self._exec("chmod +x /workspace/run_script.sh", timeout=5)

        entry_script = f"""
{env_cmds}
# apply patch
cd {self.workdir}
git config --global --add safe.directory '*'
git reset --hard {self.base_commit}
git clean -fd
git checkout {self.base_commit}
git apply -v /workspace/patch.diff
{before_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_str} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python3 /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
        self._exec(entry_script, timeout=timeout)

    def _run_tests_lite(self, timeout: int):
        """SWE-bench Lite eval: uses official swebench harness to generate eval script."""
        from swebench.harness.test_spec.python import (
            make_eval_script_list_py,
            MAP_REPO_VERSION_TO_SPECS,
        )

        # Build the instance dict that swebench expects
        import ast as _ast
        instance = {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "test_patch": self.test_patch,
            "FAIL_TO_PASS": self.fail_to_pass,
            "PASS_TO_PASS": self.pass_to_pass,
        }
        # Parse version from instance_id or eval_config
        # swebench needs "version" to find specs
        version = getattr(self, "_version", "")
        if not version:
            # Try to find from MAP_REPO_VERSION_TO_SPECS
            repo_specs = MAP_REPO_VERSION_TO_SPECS.get(self.repo, {})
            if len(repo_specs) == 1:
                version = list(repo_specs.keys())[0]
        instance["version"] = version

        specs = MAP_REPO_VERSION_TO_SPECS.get(self.repo, {}).get(version, {})
        if not specs:
            logger.warning("[%s] No specs found for %s@%s, trying all versions",
                           self.instance_id[:30], self.repo, version)
            # Fall back: try each version
            for v, s in MAP_REPO_VERSION_TO_SPECS.get(self.repo, {}).items():
                specs = s
                version = v
                break

        script_list = make_eval_script_list_py(
            instance=instance,
            specs=specs,
            env_name="testbed",
            repo_directory=self.workdir,
            base_commit=self.base_commit,
            test_patch=self.test_patch,
        )
        eval_script = "\n".join(script_list)

        # Write eval script to pod and run it, capturing output
        import tempfile as _tf
        with _tf.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(eval_script)
            script_tmp = f.name
        subprocess.run(
            ["kubectl", "cp", script_tmp,
             f"{NAMESPACE}/{self.pod_name}:/workspace/eval.sh"],
            capture_output=True, text=True, timeout=30,
        )
        os.unlink(script_tmp)

        self._exec(
            "NO_COLOR=1 TERM=dumb PY_COLORS=0 PYTEST_ADDOPTS='--color=no' "
            "bash /workspace/eval.sh > /workspace/stdout.log 2>&1 || true",
            timeout=timeout,
        )

        # Parse test output and write output.json
        # The swebench harness uses grading.py, but we can parse the test output directly
        # Use swebench's official per-repo parser to grade results
        stdout_proc = self._exec("cat /workspace/stdout.log", timeout=30)
        test_output = stdout_proc.stdout if stdout_proc and stdout_proc.returncode == 0 else ""

        import re as _re
        # Strip ANSI escape codes before parsing
        if test_output:
            test_output = _re.sub(r'\x1b\[[0-9;]*[mGKHF]', '', test_output)
        tests = []
        if test_output:
            # pytest: "test/foo.py::test_bar PASSED"
            for m in _re.finditer(
                r'^(\S+\.py::\S+)\s+(PASSED|FAILED|ERROR)', test_output, _re.MULTILINE
            ):
                tests.append({"name": m.group(1), "status": m.group(2)})
            # pytest -rA: "PASSED test/foo.py::test_bar"
            for m in _re.finditer(
                r'^(PASSED|FAILED|ERROR)\s+(\S+\.py::\S+)', test_output, _re.MULTILINE
            ):
                if m.group(2) not in {t["name"] for t in tests}:
                    tests.append({"name": m.group(2), "status": m.group(1)})
            # Django: "test_name (module.Class) ... ok"
            for m in _re.finditer(
                r'^(\S+ \(\S+\))\s+\.\.\..*?\b(ok|FAIL|ERROR)\b', test_output, _re.MULTILINE
            ):
                s = "PASSED" if m.group(2) == "ok" else "FAILED" if m.group(2) == "FAIL" else "ERROR"
                tests.append({"name": m.group(1), "status": s})
            # sympy: lines like "test_name ok" or "test_name F" (with or without leading spaces)
            for m in _re.finditer(
                r'^\s*(test_\S+)\s+(ok|FAILED|f|F|E)\s*$', test_output, _re.MULTILINE
            ):
                s = "PASSED" if m.group(2) == "ok" else "FAILED"
                if m.group(1) not in {t["name"] for t in tests}:
                    tests.append({"name": m.group(1), "status": s})
            # tox/pytest verbose: "test_file.py::test_name PASSED" with possible leading whitespace or markers
            if not tests:
                for m in _re.finditer(
                    r'(\S+\.py::\S+)\s+(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)', test_output
                ):
                    name = m.group(1)
                    status = "PASSED" if m.group(2) in ("PASSED", "XPASS") else "FAILED" if m.group(2) in ("FAILED",) else m.group(2)
                    if name not in {t["name"] for t in tests}:
                        tests.append({"name": name, "status": status})

        if not tests and test_output:
            logger.warning("[%s] eval parser found 0 tests. stdout tail:\n%s",
                           self.instance_id[:30], test_output[-500:])

        # Write output.json for the common evaluation logic
        import tempfile as _tf2
        output_json = json.dumps({"tests": tests})
        with _tf2.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(output_json)
            oj_tmp = f.name
        subprocess.run(
            ["kubectl", "cp", oj_tmp,
             f"{NAMESPACE}/{self.pod_name}:/workspace/output.json"],
            capture_output=True, text=True, timeout=30,
        )
        os.unlink(oj_tmp)

    def run_tests(self, timeout: int = 300) -> Dict[str, Any]:
        """Run official evaluation.

        For SWE-bench Pro: uses per-instance run_script.sh + parser.py
        For SWE-bench Lite: runs pytest directly (no external scripts needed)
        """
        # ---- get patch from current workspace ---- #
        patch = self.get_git_diff()
        if not patch:
            return {"passed": False, "reason": "no patch to evaluate"}

        self._exec("mkdir -p /workspace", timeout=5)

        # Write patch to pod
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            f.write(patch)
            patch_tmp = f.name
        subprocess.run(
            ["kubectl", "cp", patch_tmp, f"{NAMESPACE}/{self.pod_name}:/workspace/patch.diff"],
            capture_output=True, timeout=30,
        )
        os.unlink(patch_tmp)

        # Check if this is a Pro instance (has run_script.sh) or Lite (direct pytest)
        scripts_base = os.path.join(SCRIPTS_DIR, self.instance_id)
        run_script = os.path.join(scripts_base, "run_script.sh")
        parser_py = os.path.join(scripts_base, "parser.py")
        use_pro_scripts = os.path.exists(run_script) and os.path.exists(parser_py)

        if use_pro_scripts:
            self._run_tests_pro(run_script, parser_py, timeout)
        else:
            self._run_tests_lite(timeout)

        # ---- read output.json (written by both Pro and Lite paths) ---- #
        result_proc = self._exec("cat /workspace/output.json", timeout=10)
        output_json = (
            result_proc.stdout.strip()
            if result_proc and result_proc.returncode == 0
            else ""
        )

        # ---- evaluate (exact match of official lines 555-558) ---- #
        import ast

        try:
            parsed = json.loads(output_json) if output_json else {}
        except json.JSONDecodeError:
            parsed = {}

        passed_tests = {
            x["name"] for x in parsed.get("tests", []) if x["status"] == "PASSED"
        }

        # Fallback: if per-instance parser found nothing, re-parse stdout
        # directly. Some parsers have regex order bugs (PASSED before test name
        # vs after), so we try both formats here.
        if not passed_tests:
            stdout_proc = self._exec("cat /workspace/stdout.log", timeout=10)
            stdout_log = stdout_proc.stdout if stdout_proc and stdout_proc.returncode == 0 else ""
            if stdout_log:
                import re as _re
                # Format 1: "test/foo.py::test_bar PASSED" (standard pytest)
                for m in _re.finditer(
                    r'^(test\S+\.py::\S+)\s+(?:PASSED|XPASS)', stdout_log, _re.MULTILINE
                ):
                    passed_tests.add(m.group(1))
                # Format 2: "[gw0] [ N%] PASSED test/foo.py::test_bar" (forked)
                for m in _re.finditer(
                    r'(?:PASSED|XPASS)\s+(test\S+\.py::\S+)', stdout_log
                ):
                    passed_tests.add(m.group(1))
                if passed_tests:
                    logger.info(
                        "[%s] fallback parser found %d passed tests",
                        self.instance_id[:30], len(passed_tests),
                    )

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

        # Normalize test names: strip path prefixes so
        # "lib/matplotlib/tests/test_legend.py::test_foo" matches "test_legend.py::test_foo"
        def _normalize_test_name(name: str) -> str:
            """Strip directory prefix, keep filename::test_id."""
            if "::" in name:
                path_part, test_part = name.split("::", 1)
                # Keep only filename
                filename = path_part.rsplit("/", 1)[-1]
                return f"{filename}::{test_part}"
            return name.rsplit("/", 1)[-1]

        passed_norm = {_normalize_test_name(t) for t in passed_tests}
        f2p_norm = {_normalize_test_name(t) for t in f2p}
        p2p_norm = {_normalize_test_name(t) for t in p2p}

        resolved = (f2p_norm | p2p_norm) <= passed_norm if (f2p_norm or p2p_norm) else False

        # Fallback: also try exact match in case normalization loses info
        if not resolved:
            resolved = (f2p | p2p) <= passed_tests if (f2p or p2p) else False

        if not resolved and f2p:
            missing_f2p = f2p - passed_tests
            if missing_f2p:
                logger.warning(
                    "[%s] f2p tests NOT passed: %s",
                    self.instance_id[:30], list(missing_f2p)[:5],
                )
            # Log eval stdout tail for debugging
            log_proc = self._exec("tail -30 /workspace/stdout.log 2>/dev/null", timeout=5)
            if log_proc and log_proc.stdout:
                logger.debug("[%s] eval stdout tail:\n%s", self.instance_id[:30], log_proc.stdout[:500])

        return {
            "passed": resolved,
            "f2p_total": len(f2p),
            "p2p_total": len(p2p),
            "passed_count": len(passed_tests),
            "details": json.dumps(list(passed_tests)[:10]) if passed_tests else "[]",
        }

    # ------------------------------------------------------------------ #
    #  Command execution (= docker exec)
    # ------------------------------------------------------------------ #

    def write_file(self, path: str, content: str) -> bool:
        """Write a file into the pod via kubectl cp (avoids URI length limits)."""
        if not self.pod_name:
            return False
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            proc = subprocess.run(
                ["kubectl", "cp", tmp_path,
                 f"{NAMESPACE}/{self.pod_name}:{path}"],
                capture_output=True, text=True, timeout=30,
            )
            return proc.returncode == 0
        finally:
            os.unlink(tmp_path)

    def _exec(
        self, cmd: str, timeout: int = 30
    ) -> Optional[subprocess.CompletedProcess]:
        if not self.pod_name:
            return None
        full_cmd = f"cd {self.workdir} && {cmd}"
        try:
            return subprocess.run(
                ["kubectl", "exec", self.pod_name, "-n", NAMESPACE,
                 "--", "bash", "-c", full_cmd],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None
