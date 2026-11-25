"""Tests for agent_cap.harbor (CAPAgent on Harbor).

Most tests need neither Harbor nor Docker. Tests that import Harbor are skipped
when it is not installed (it needs Python >= 3.12); the container test runs only
with AGENTCAP_DOCKER_TESTS=1 and a built image in AGENTCAP_IMO_TEST_IMAGE.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from harbor_fakes import DockerEnvironment, LocalEnvironment, docker_available  # noqa: E402

from agent_cap.harbor import RESULT_FILENAME, jobspec
from agent_cap.harbor.adapters import TEMPLATES, imo_answerbench, nl2repo
from agent_cap.harbor.export_teas import export_job, trial_row
from agent_cap.harbor.tools import HarborTools

HAS_HARBOR = importlib.util.find_spec("harbor") is not None
needs_harbor = pytest.mark.skipif(not HAS_HARBOR, reason="harbor not installed")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grade_imo = _load_module(TEMPLATES / "imo_answerbench" / "tests" / "grade_imo.py", "grade_imo")
nl2repo_eval = _load_module(TEMPLATES / "nl2repo" / "tests" / "nl2repo_eval.py", "nl2repo_eval")


# --------------------------------------------------------------------------- #
# IMO grader parity with AgentCAP's imo evaluator
# --------------------------------------------------------------------------- #

ANSWER_TEXTS = [
    "So the answer is \\boxed{3}.",
    "First \\boxed{1} then finally \\boxed{\\frac{1}{2}}",
    "The final answer is 42",
    "We get **7**.",
    "nothing to see here",
    "\\boxed {x^2 + 1}",
    "",
]


@pytest.mark.parametrize("text", ANSWER_TEXTS)
def test_grade_imo_extraction_matches_agentcap(text):
    from agent_cap.agents import evaluators

    assert grade_imo.scan_for_imo_answer(text) == evaluators._scan_for_imo_answer(text)


@pytest.mark.parametrize(
    "output,gold",
    [
        ("\\boxed{3}", "3"),
        ("\\boxed{0.5}", "\\frac{1}{2}"),
        ("\\boxed{4}", "3"),
        ("", "3"),
        ("no answer", "3"),
    ],
)
def test_grade_imo_decision_matches_agentcap(output, gold, monkeypatch):
    from agent_cap.agents.evaluators import IMOAnswerBenchEvaluator

    for var in ("OPENROUTER_API_KEY", "IMO_JUDGE_BASE_URL", "IMO_JUDGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    passed, details = grade_imo.evaluate(output, gold)
    ref = IMOAnswerBenchEvaluator().evaluate({"answer": gold}, output)
    assert passed == ref.passed
    assert details.get("method") == ref.details.get("method")
    assert details.get("predicted") == ref.details.get("predicted")


def test_grade_imo_main_writes_numeric_reward(tmp_path, monkeypatch):
    (tmp_path / "expected.json").write_text(json.dumps({"problem_id": "p1", "answer": "3"}))
    (tmp_path / "answer.txt").write_text("thus \\boxed{3}")
    monkeypatch.setenv("EXPECTED_FILE", str(tmp_path / "expected.json"))
    monkeypatch.setenv("ANSWER_FILE", str(tmp_path / "answer.txt"))
    monkeypatch.setenv("VERIFIER_DIR", str(tmp_path / "verifier"))
    module = _load_module(TEMPLATES / "imo_answerbench" / "tests" / "grade_imo.py", "grade_imo_main")
    assert module.main() == 0
    reward = json.loads((tmp_path / "verifier" / "reward.json").read_text())
    assert reward == {"reward": 1.0}
    grade = json.loads((tmp_path / "verifier" / "grade.json").read_text())
    assert grade["method"] == "math_verify" and grade["problem_id"] == "p1"


# --------------------------------------------------------------------------- #
# NL2Repo verifier
# --------------------------------------------------------------------------- #

def test_analyze_pytest_results_matches_upstream_rules():
    results = nl2repo_eval.analyze_pytest_results(
        [
            {"command": "pip install -e .", "output": "Successfully installed 3 passed"},
            {"command": "python -m pytest tests -v", "output": "== 7 passed, 2 failed, 1 error in 0.3s =="},
        ],
        10,
    )
    assert results["passed"] == 7 and results["failed"] == 2 and results["errors"] == 1
    assert results["success_rate"] == 0.7
    capped = nl2repo_eval.analyze_pytest_results([{"command": "pytest", "output": "12 passed"}], 10)
    assert capped["success_rate"] == 1


def _fake_nl2repo_task(root: Path):
    reference = root / "reference"
    (reference / "tests").mkdir(parents=True)
    (reference / "tests" / "test_mod.py").write_text(
        "from mypkg import add\n\n"
        "def test_add():\n    assert add(1, 2) == 3\n\n"
        "def test_add_neg():\n    assert add(-1, -2) == -3\n"
    )
    meta = {
        "task": "toy",
        "test_case_count": 2,
        "test_commands": [f"{sys.executable} -m pytest -p no:cacheprovider tests/test_mod.py"],
        "test_files": ["tests"],
    }
    (root / "meta.json").write_text(json.dumps(meta))
    return reference


def _run_nl2repo_eval(root: Path, workspace_files):
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    for rel, text in workspace_files.items():
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    env = dict(
        os.environ,
        NL2REPO_META=str(root / "meta.json"),
        NL2REPO_REFERENCE=str(root / "reference"),
        NL2REPO_WORKSPACE=str(workspace),
        NL2REPO_STAGING=str(root / "staging"),
        VERIFIER_DIR=str(root / "verifier"),
    )
    script = TEMPLATES / "nl2repo" / "tests" / "nl2repo_eval.py"
    subprocess.run([sys.executable, str(script)], env=env, check=True, capture_output=True)
    return (
        json.loads((root / "verifier" / "reward.json").read_text()),
        json.loads((root / "verifier" / "grade.json").read_text()),
    )


def test_nl2repo_eval_scores_and_strips_agent_files(tmp_path):
    _fake_nl2repo_task(tmp_path)
    reward, grade = _run_nl2repo_eval(
        tmp_path,
        {
            "mypkg/__init__.py": "def add(a, b):\n    return a + b\n",
            # agent's own tests and packaging are removed, as upstream does
            "tests/test_mod.py": "def test_fake():\n    assert True\n",
            "setup.py": "raise SystemExit('should have been removed')\n",
        },
    )
    assert reward["passed"] == 2 and reward["reward"] == 1.0 and reward["pass_all"] == 1.0
    official = grade["passes"]["official"]
    assert "setup.py" in official["removed_package_files"]
    assert "tests" in official["removed_test_files"]
    assert reward["junit_passed"] == 2


def test_nl2repo_hardened_score_removes_conftest_override(tmp_path):
    _fake_nl2repo_task(tmp_path)
    cheat = (
        "import pytest\n\n"
        "@pytest.hookimpl(hookwrapper=True)\n"
        "def pytest_runtest_makereport(item, call):\n"
        "    outcome = yield\n"
        "    rep = outcome.get_result()\n"
        "    rep.outcome = 'passed'\n"
    )
    reward, grade = _run_nl2repo_eval(
        tmp_path,
        {"mypkg/__init__.py": "def add(a, b):\n    return 0\n", "conftest.py": cheat},
    )
    assert grade["hardened_pass_run"] is True
    assert reward["reward_hardened"] == 0.0
    assert "conftest.py" in grade["passes"]["hardened"]["removed_hardening_files"]


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #

def test_imo_adapter_writes_parity_prompt_and_valid_toml(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"Problem ID": "imo-bench-algebra-001", "Problem": "Find x if x+1=4.", "Short Answer": "3", "Category": "Algebra"},
                {"Problem ID": "skip", "Problem": "   "},
                {"Problem ID": "imo-bench-nt-002", "Problem": 'Say "hi" \\ there', "Short Answer": "\\frac{1}{2}", "Category": "NT"},
            ]
        )
    )
    dirs = imo_answerbench.generate(tmp_path / "out", source=str(rows), limit=0)
    assert [d.name for d in dirs] == ["imo-bench-algebra-001", "imo-bench-nt-002"]
    task = dirs[0]
    assert (task / "instruction.md").read_text() == (
        "Find x if x+1=4.\n\nSolve this step by step. Place your final numerical answer inside \\boxed{}"
    )
    cfg = tomllib.loads((task / "task.toml").read_text())
    assert cfg["metadata"]["system_prompt"] == imo_answerbench.SYSTEM_PROMPT
    assert cfg["metadata"]["dataset_index"] == 0
    assert json.loads((dirs[1] / "tests" / "expected.json").read_text())["answer"] == "\\frac{1}{2}"
    tomllib.loads((dirs[1] / "task.toml").read_text())  # quotes/backslashes escaped
    assert os.access(task / "tests" / "test.sh", os.X_OK)
    limited = imo_answerbench.generate(tmp_path / "lim", source=str(rows), limit=1)
    assert len(limited) == 1


def _fake_nl2repo_checkout(root: Path) -> Path:
    task = root / "test_files" / "toy-pkg"
    task.mkdir(parents=True)
    (task / "start.md").write_text("# toy\nImplement add().\n")
    (task / "test_case_count.txt").write_text("5\n")
    (task / "test_commands.json").write_text(json.dumps(["pip install -e .", "python -m pytest tests -v"]))
    (task / "test_files.json").write_text(json.dumps(["tests"]))
    return root


def test_nl2repo_adapter_layout(tmp_path):
    source = _fake_nl2repo_checkout(tmp_path / "src")
    dirs = nl2repo.generate(source, tmp_path / "out")
    task = dirs[0]
    assert (task / "instruction.md").read_text() == nl2repo.INSTRUCTION
    assert (task / "environment" / "start.md").read_text().startswith("# toy")
    assert "ghcr.io/multimodal-art-projection/nl2repobench/toy-pkg:1.0" in (task / "tests" / "Dockerfile").read_text()
    meta = json.loads((task / "tests" / "meta.json").read_text())
    assert meta["test_case_count"] == 5 and meta["test_files"] == ["tests"]
    cfg = tomllib.loads((task / "task.toml").read_text())
    assert cfg["verifier"]["environment_mode"] == "separate"
    assert {"source": "/workspace", "exclude": [".venv", "venv", "__pycache__", "*.pyc", "node_modules", ".git"]} in cfg["artifacts"]
    # hidden tests never go into the agent's build context
    assert not any("test_" in p.name for p in (task / "environment").iterdir())


@needs_harbor
def test_generated_tasks_validate_with_harbor(tmp_path):
    from harbor.models.task.config import TaskConfig

    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([{"Problem ID": "p", "Problem": "1+1?", "Short Answer": "2"}]))
    imo_dirs = imo_answerbench.generate(tmp_path / "imo", source=str(rows))
    nl_dirs = nl2repo.generate(_fake_nl2repo_checkout(tmp_path / "src"), tmp_path / "nl")
    for d in imo_dirs + nl_dirs:
        TaskConfig.model_validate(tomllib.loads((d / "task.toml").read_text()))


# --------------------------------------------------------------------------- #
# Job specs
# --------------------------------------------------------------------------- #

TEAS_AGENT_YAML = """
strategy: single
dataset: imo-answerbench
num_tasks: 25
max_turns: 128
tool_backend: math-python
evaluator: imo
agents:
  agent:
    name: gpt-oss
    base_url: http://127.0.0.1:8000
    api_key: EMPTY
    protocol: harmony
    engine: sglang
    max_tokens: 131072
    temperature: 1.0
judge:
  name: gemini-3.1-flash-lite
  base_url: https://generativelanguage.googleapis.com/v1beta/openai
  api_key: ${GEMINI_API_KEY}
"""


def test_teas_agent_yaml_maps_to_capagent_job(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(TEAS_AGENT_YAML)
    teas_kwargs, verifier_env = jobspec.load_teas_agent_yaml(path)
    roles = teas_kwargs.pop("roles")
    kwargs = jobspec.capagent_kwargs("imo-answerbench", roles, teas_kwargs)
    assert kwargs["tool_backend"] == "math-python" and kwargs["max_turns"] == 128
    assert kwargs["roles"]["agent"]["protocol"] == "harmony"
    assert kwargs["system_prompt"] == imo_answerbench.SYSTEM_PROMPT
    assert verifier_env["IMO_JUDGE_API_KEY"] == "${GEMINI_API_KEY}"  # not expanded into files
    cfg = jobspec.build_job_config(
        dataset_path=tmp_path / "ds", agent_kwargs=kwargs, job_name="j", n_concurrent=4
    )
    assert cfg["agents"][0]["import_path"] == "agent_cap.harbor.agent:CAPAgent"
    assert cfg["n_concurrent_trials"] == 4
    if HAS_HARBOR:
        jobspec.validate_job_config(cfg)


def test_parse_role_keeps_env_references():
    role, fields = jobspec.parse_role("planner=name=m,base_url=http://h/v1,api_key=${KEY}")
    assert role == "planner" and fields["api_key"] == "${KEY}"


# --------------------------------------------------------------------------- #
# Tools over an environment
# --------------------------------------------------------------------------- #

def test_shell_tools_run_through_environment(tmp_path):
    env = LocalEnvironment(Path("/"))
    workdir = tmp_path / "workspace"

    async def go():
        tools = HarborTools(env, "shell", workdir=str(workdir))
        await tools.prepare()
        names = [t["function"]["name"] for t in await tools.list_tools()]
        created = await tools.call(
            "str_replace_editor",
            {"command": "create", "path": str(workdir / "a.py"), "file_text": "x = 1\n"},
        )
        edited = await tools.call(
            "str_replace_editor",
            {"command": "str_replace", "path": str(workdir / "a.py"), "old_str": "x = 1", "new_str": "x = 2"},
        )
        out = await tools.call("bash", {"command": "cat a.py && cd sub 2>/dev/null; pwd"})
        return names, created, edited, out

    names, created, edited, out = asyncio.run(go())
    assert names == ["bash", "str_replace_editor", "submit"]
    # Official SWE-agent editor messages.
    assert created.startswith("File created successfully")
    assert "has been edited" in edited and "x = 2" in edited
    assert "x = 2" in out and str(workdir) in out


@pytest.mark.skipif(importlib.util.find_spec("jupyter_client") is None, reason="jupyter_client missing")
def test_python_tool_matches_math_python_backend(tmp_path):
    env = LocalEnvironment(Path("/"))

    async def go():
        tools = HarborTools(
            env,
            "math-python",
            workdir=str(tmp_path),
            python_bin=sys.executable,
            kernel_dir=str(tmp_path / "kernel"),
            python_exec_timeout=10,
        )
        await tools.prepare()
        try:
            return [
                await tools.call("python", {"code": "x = 6 * 7\nx"}),
                await tools.call("python", {"code": "x + 1"}),
                await tools.call("python", {"code": "y = 1"}),
                await tools.call("python", {"code": ""}),
            ]
        finally:
            await tools.teardown()

    outs = asyncio.run(go())
    assert outs[0] == "42\n" and outs[1] == "43\n"
    assert outs[2] == "[WARN] No output. Use print() to see results."
    assert outs[3].startswith("[ERROR] Tool argument 'code' must be a non-empty string")


@pytest.mark.skipif(
    os.environ.get("AGENTCAP_DOCKER_TESTS") != "1" or not docker_available(),
    reason="set AGENTCAP_DOCKER_TESTS=1 (and AGENTCAP_IMO_TEST_IMAGE) to run container tests",
)
def test_python_tool_in_imo_task_container():
    image = os.environ.get("AGENTCAP_IMO_TEST_IMAGE", "capagent-test-imo")
    env = DockerEnvironment(image).start()

    async def go():
        tools = HarborTools(env, "math-python", python_exec_timeout=5)
        await tools.prepare()
        try:
            first = await tools.call("python", {"code": "x = 6 * 7\nx"})
            timeout = await tools.call("python", {"code": "import time\ntime.sleep(8)"})
            after = await tools.call("python", {"code": "x"})
            await tools.write_file("/workspace/answer.txt", "\\boxed{3}")
            written = (await env.exec("cat /workspace/answer.txt")).stdout
            return first, timeout, after, written
        finally:
            await tools.teardown()

    try:
        first, timeout, after, written = asyncio.run(go())
    finally:
        env.stop()
    assert first == "42\n"
    assert timeout.startswith("[ERROR] Execution timed out")
    assert "NameError" in after  # kernel restarted, state cleared, as on the host
    assert written == "\\boxed{3}"


# --------------------------------------------------------------------------- #
# CAPAgent and export
# --------------------------------------------------------------------------- #

@needs_harbor
def test_capagent_writes_agentcap_row(tmp_path):
    from harbor.models.agent.context import AgentContext

    from agent_cap.harbor.agent import CAPAgent

    logs = tmp_path / "jobs" / "job" / "toy-task__abc123" / "agent"
    agent = CAPAgent(
        logs_dir=logs,
        strategy="single",
        tool_backend="shell",
        workdir=str(tmp_path / "workspace"),
        roles={"agent": {"name": "mock-model"}},
        system_prompt="Be brief.",
        answer_file=str(tmp_path / "workspace" / "answer.txt"),
        max_turns=3,
    )
    env = LocalEnvironment(Path("/"))
    context = AgentContext()

    async def go():
        await agent.setup(env)
        await agent.run("Do the thing.", env, context)

    asyncio.run(go())
    row = json.loads((logs / RESULT_FILENAME).read_text())
    assert row["task_id"] == "toy-task"
    assert row["total_usage"]["requests"] == 2 and len(row["turn_stats"]) == 2
    assert row["agentcard"]["strategy"] == "single"
    assert context.n_input_tokens == row["total_usage"]["input_tokens"]
    assert (tmp_path / "workspace" / "answer.txt").read_text().startswith("ANSWER: mock reply")


def _fake_trial(job: Path, name: str, task: str, reward, row=None):
    trial = job / name
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    result = {
        "task_name": task,
        "trial_name": name,
        "started_at": "2026-11-25T00:00:00+00:00",
        "finished_at": "2026-11-25T00:01:00+00:00",
        "verifier_result": {"rewards": reward} if reward is not None else None,
        "exception_info": None,
    }
    (trial / "result.json").write_text(json.dumps(result))
    if row is not None:
        (trial / "agent" / RESULT_FILENAME).write_text(json.dumps(row))
    return trial


def _agent_row(n_requests: int):
    return {
        "task_id": "x",
        "strategy": "single",
        "output_text": "\\boxed{3}",
        "e2e_latency_s": 12.5,
        "per_role_usage": {"agent": {"input_tokens": 100, "output_tokens": 20, "requests": n_requests}},
        "total_usage": {"input_tokens": 100, "output_tokens": 20, "completion_tokens": 20,
                        "reasoning_tokens": 0, "cached_tokens": 10, "requests": n_requests},
        "errors": [],
        "num_turns": n_requests,
        "tool_calls": 1,
        "turn_stats": [
            {"input_tokens": 50, "output_tokens": 10, "completion_tokens": 10, "reasoning_tokens": 0,
             "cached_tokens": 5, "latency_s": 1.0, "ttft_s": 0.2, "decode_s": 0.5, "num_tool_calls": 1}
            for _ in range(n_requests)
        ],
        "agentcard": {"agent": "capagent", "strategy": "single", "roles": {"agent": "gpt-oss"}},
    }


def test_export_job_writes_teas_files(tmp_path):
    job = tmp_path / "job"
    _fake_trial(job, "p1__aaa", "imo-answerbench/p1", {"reward": 1.0}, _agent_row(2))
    _fake_trial(job, "p2__bbb", "imo-answerbench/p2", {"reward": 0.0}, _agent_row(1))
    _fake_trial(job, "p3__ccc", "imo-answerbench/p3", None, None)  # agent crashed, no reward
    metrics_path, results_path = export_job(job, tmp_path / "out", dataset="imo-answerbench", timestamp="t0")
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert [r["task_id"] for r in rows] == ["p1", "p2", "p3"]
    assert [r["eval_passed"] for r in rows] == [True, False, False]
    metrics = json.loads(metrics_path.read_text())
    assert metrics["quality"] == {"acc": 0.3333, "total_examples": 3, "passed": 1}
    assert metrics["agentic"]["total_requests"] == 3
    assert metrics["performance"]["ttft"] == 0.2
    detailed = (tmp_path / "out" / "detailed-results_imo-answerbench_t0.jsonl").read_text().splitlines()
    assert len(detailed) == 3
    metadata = json.loads((tmp_path / "out" / "metadata_imo-answerbench_t0.json").read_text())
    assert metadata["harbor"]["agent"] == "capagent" and metadata["harbor"]["n_trials"] == 3


def test_trial_row_uses_pass_all_for_partial_scores(tmp_path):
    trial = _fake_trial(tmp_path, "math-verify__x", "nl2repo/math-verify",
                        {"reward": 0.9, "pass_all": 0.0, "passed": 90, "total": 100}, _agent_row(1))
    row = trial_row(trial)
    assert row["eval_score"] == 0.9 and row["eval_passed"] is False
    assert row["task_id"] == "math-verify" and row["harbor_task"] == "nl2repo/math-verify"
