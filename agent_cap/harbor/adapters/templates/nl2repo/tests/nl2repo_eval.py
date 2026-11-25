"""NL2Repo verifier: reproduces upstream NL2RepoBench scoring inside Harbor.

Upstream (multimodal-art-projection/NL2RepoBench, openhands/post_processor.py):

1. copy the generated workspace;
2. delete every package-management file anywhere in it (same 15 names);
3. delete the task's test paths (test_files.json);
4. COPY it over /workspace of ghcr.io/multimodal-art-projection/nl2repobench/<task>:1.0,
   whose /workspace holds the hidden tests and the original pyproject.toml;
5. run each command of test_commands.json in /workspace, argv split with shlex,
   no shell, PYTHONPATH=/workspace;
6. for pytest commands, sum "(\\d+) passed", "(\\d+) failed" and "(\\d+) error"
   over every line of the command output; score = min(passed / test_case_count, 1).

Upstream runs commands through python-on-whales: a zero exit returns stdout, a
non-zero exit raises and the parsed text is the exception message, which carries
both stdout and stderr. This script reproduces that. The verifier image (see
tests/Dockerfile) moved the image's /workspace to /opt/nl2repo/reference at build
time; Harbor uploads the agent's /workspace here as an artifact.

Writes /logs/verifier/reward.json (numbers only) and /logs/verifier/grade.json.
Python 3.8+ standard library only (the evaluation images ship Python 3.10).
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

META_FILE = os.environ.get("NL2REPO_META", "/tests/meta.json")
REFERENCE_DIR = os.environ.get("NL2REPO_REFERENCE", "/opt/nl2repo/reference")
WORKSPACE = os.environ.get("NL2REPO_WORKSPACE", "/workspace")
STAGING = os.environ.get("NL2REPO_STAGING", "/tmp/nl2repo_submission")
VERIFIER_DIR = os.environ.get("VERIFIER_DIR", "/logs/verifier")
COMMAND_TIMEOUT_S = int(os.environ.get("NL2REPO_COMMAND_TIMEOUT_S", "1500"))

# openhands/post_processor.py: remove_package_files()
PACKAGE_FILES = [
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "tox.ini",
    "pytest.ini",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "environment.yml",
    "conda-env.yaml",
    "manifest.in",
    "MANIFEST.in",
]

# Files that change how pytest or the interpreter behaves without being tests.
# Upstream keeps them; the "hardened" score removes them (reported separately).
HARDENING_FILES = ["conftest.py", "sitecustomize.py", "usercustomize.py"]


def remove_package_files(root):
    removed = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name in PACKAGE_FILES:
                path = os.path.join(dirpath, name)
                try:
                    os.remove(path)
                    removed.append(os.path.relpath(path, root))
                except OSError:
                    pass
    return removed


def remove_test_files(root, test_files):
    removed = []
    for rel in test_files:
        target = os.path.join(root, rel)
        try:
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
                removed.append(rel)
            elif os.path.exists(target) or os.path.islink(target):
                os.remove(target)
                removed.append(rel)
        except OSError:
            pass
    return removed


def remove_hardening_files(root, keep_under):
    """Delete conftest/sitecustomize files and .pth files outside the hidden test paths."""
    keep = [os.path.normpath(os.path.join(root, k)) for k in keep_under]
    removed = []
    for dirpath, _dirs, files in os.walk(root):
        if any(os.path.normpath(dirpath).startswith(k) for k in keep):
            continue
        for name in files:
            if name in HARDENING_FILES or name.endswith(".pth"):
                path = os.path.join(dirpath, name)
                try:
                    os.remove(path)
                    removed.append(os.path.relpath(path, root))
                except OSError:
                    pass
    return removed


def copy_tree_over(src, dst):
    """Merge src into dst, overwriting files: the effect of Docker COPY src dst."""
    if not os.path.isdir(src):
        return
    for dirpath, dirs, files in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        target_dir = dst if rel == "." else os.path.join(dst, rel)
        if os.path.isfile(target_dir) or os.path.islink(target_dir):
            os.remove(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        for name in files:
            s = os.path.join(dirpath, name)
            d = os.path.join(target_dir, name)
            if os.path.isdir(d) and not os.path.islink(d):
                shutil.rmtree(d)
            elif os.path.lexists(d):
                os.remove(d)
            if os.path.islink(s):
                os.symlink(os.readlink(s), d)
            else:
                shutil.copy2(s, d)
        for name in list(dirs):
            s = os.path.join(dirpath, name)
            if os.path.islink(s):
                d = os.path.join(target_dir, name)
                if os.path.lexists(d):
                    if os.path.isdir(d) and not os.path.islink(d):
                        shutil.rmtree(d)
                    else:
                        os.remove(d)
                os.symlink(os.readlink(s), d)
                dirs.remove(name)


def reset_workspace(reference):
    if os.path.isdir(WORKSPACE):
        for name in os.listdir(WORKSPACE):
            path = os.path.join(WORKSPACE, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    else:
        os.makedirs(WORKSPACE)
    copy_tree_over(reference, WORKSPACE)


def run_command(command, junit_path=None):
    """Run one upstream test command; return (exit_code, text upstream would parse)."""
    argv = shlex.split(command)
    env = dict(os.environ)
    env["PYTHONPATH"] = WORKSPACE + ":" + env.get("PYTHONPATH", "")
    if junit_path and "pytest" in command.lower():
        env["PYTEST_ADDOPTS"] = (env.get("PYTEST_ADDOPTS", "") + " --junitxml=" + junit_path).strip()
    try:
        proc = subprocess.run(
            argv,
            cwd=WORKSPACE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", "replace")
        err = (exc.stderr or b"").decode("utf-8", "replace")
        return -1, "command timed out after %ss\n%s\n%s" % (COMMAND_TIMEOUT_S, out, err)
    except OSError as exc:
        return -1, "%s: %s" % (type(exc).__name__, exc)
    stdout = proc.stdout.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8", "replace")
    if proc.returncode == 0:
        return 0, stdout
    # Shape of python_on_whales.exceptions.DockerException's message.
    text = (
        "The command executed was `%s`.\nIt returned with code %d\n"
        "The content of stdout is '%s'\nThe content of stderr is '%s'\n"
        % (" ".join(argv), proc.returncode, stdout, stderr)
    )
    return proc.returncode, text


def analyze_pytest_results(command_results, total_test_cases):
    """openhands/post_processor.py: analyze_pytest_results(), unchanged logic."""
    results = {"passed": 0, "failed": 0, "errors": 0, "total": total_test_cases, "success_rate": 0.0}
    for result in command_results:
        if "pytest" in result["command"].lower():
            for line in result["output"].split("\n"):
                m = re.search(r"(\d+) passed", line)
                if m:
                    results["passed"] += int(m.group(1))
                m = re.search(r"(\d+) failed", line)
                if m:
                    results["failed"] += int(m.group(1))
                m = re.search(r"(\d+) error", line)
                if m:
                    results["errors"] += int(m.group(1))
    if results["total"] > 0:
        results["success_rate"] = min(results["passed"] / results["total"], 1)
    return results


def junit_counts(paths):
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    found = False
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        found = True
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.get(key, 0) or 0)
    if not found:
        return None
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    return counts


def evaluate(meta, submission, hardened):
    test_files = list(meta.get("test_files") or [])
    total = int(meta.get("test_case_count") or 0)
    shutil.rmtree(STAGING, ignore_errors=True)
    if os.path.isdir(submission):
        shutil.copytree(submission, STAGING, symlinks=True)
    else:
        os.makedirs(STAGING)
    removed_pkg = remove_package_files(STAGING)
    removed_tests = remove_test_files(STAGING, test_files)
    removed_hardening = remove_hardening_files(STAGING, test_files) if hardened else []
    reset_workspace(REFERENCE_DIR)
    copy_tree_over(STAGING, WORKSPACE)

    tag = "hardened" if hardened else "official"
    command_results, junit_paths = [], []
    for i, command in enumerate(meta.get("test_commands") or []):
        junit_path = os.path.join(VERIFIER_DIR, "junit-%s-%d.xml" % (tag, i))
        if os.path.exists(junit_path):
            os.remove(junit_path)
        code, output = run_command(command, junit_path)
        junit_paths.append(junit_path)
        command_results.append({"command": command, "exit_code": code, "output": output})
    parsed = analyze_pytest_results(command_results, total)
    return {
        "pytest_results": parsed,
        "junit": junit_counts(junit_paths),
        "removed_package_files": removed_pkg,
        "removed_test_files": removed_tests,
        "removed_hardening_files": removed_hardening,
        "commands": [
            {"command": r["command"], "exit_code": r["exit_code"], "output_tail": r["output"][-4000:]}
            for r in command_results
        ],
    }


def main():
    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    submission = os.path.join("/tmp", "nl2repo_agent_workspace")
    # Keep the agent's workspace aside: /workspace is rebuilt for each scoring pass.
    shutil.rmtree(submission, ignore_errors=True)
    if os.path.isdir(WORKSPACE):
        shutil.copytree(WORKSPACE, submission, symlinks=True)

    official = evaluate(meta, submission, hardened=False)
    passes = {"official": official}
    hardening_needed = bool(
        [
            p
            for p in _walk_names(submission)
            if os.path.basename(p) in HARDENING_FILES or p.endswith(".pth")
        ]
    )
    if hardening_needed and os.environ.get("NL2REPO_HARDENED_PASS", "1") == "1":
        passes["hardened"] = evaluate(meta, submission, hardened=True)

    total = int(meta.get("test_case_count") or 0)
    off = official["pytest_results"]
    hard = passes.get("hardened", official)["pytest_results"]
    junit = official["junit"] or {}
    reward = {
        "reward": float(off["success_rate"]),
        "passed": int(off["passed"]),
        "failed": int(off["failed"]),
        "errors": int(off["errors"]),
        "total": total,
        "pass_all": 1.0 if total > 0 and off["passed"] >= total else 0.0,
        "reward_hardened": float(hard["success_rate"]),
        "junit_passed": int(junit.get("passed", -1)) if junit else -1,
    }
    grade = {
        "evaluator": "nl2repo",
        "task": meta.get("task"),
        "upstream_commit": meta.get("upstream_commit"),
        "test_case_count": total,
        "score_definition": "passed / test_case_count, capped at 1 (upstream analyze_pytest_results)",
        "hardened_pass_run": "hardened" in passes,
        "passes": passes,
    }
    with open(os.path.join(VERIFIER_DIR, "grade.json"), "w", encoding="utf-8") as f:
        json.dump(grade, f, ensure_ascii=False, indent=1)
    with open(os.path.join(VERIFIER_DIR, "reward.json"), "w", encoding="utf-8") as f:
        json.dump(reward, f)
    print(json.dumps(reward))
    return 0


def _walk_names(root):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            yield os.path.join(dirpath, name)


if __name__ == "__main__":
    sys.exit(main())
