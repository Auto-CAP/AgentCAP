from dataclasses import dataclass
from unittest.mock import mock_open, patch

import pytest

from agent_cap.agents.evaluators_swebench import SWEBenchEvaluator
from agent_cap.agents.swebench_modal_compat import (
    add_legacy_editable_fallback,
    patch_test_spec,
)


def test_retries_only_editable_installs_with_legacy_pip():
    editable = "python -m pip install -e .[test] --verbose"
    patched = add_legacy_editable_fallback(editable)

    assert patched.count(editable) == 2
    assert "python -m pip install 'pip<25'" in patched
    assert add_legacy_editable_fallback("python -m pip install pytest") == (
        "python -m pip install pytest"
    )


@dataclass
class _FakeTestSpec:
    repo_script_list: list[str]


def test_copies_test_spec_instead_of_mutating_it():
    original = _FakeTestSpec(
        ["git clone example", "python3.11 -m pip install --editable=.[dev]"]
    )
    patched = patch_test_spec(original)

    assert patched is not original
    assert patched.repo_script_list[0] == original.repo_script_list[0]
    assert patched.repo_script_list[1] != original.repo_script_list[1]
    assert original.repo_script_list[1] == "python3.11 -m pip install --editable=.[dev]"


def test_evaluator_rejects_unknown_runtime():
    with pytest.raises(ValueError, match="expected 'docker' or 'modal'"):
        SWEBenchEvaluator(runtime="podman")


def test_evaluator_uses_modal_compat_runner(tmp_path):
    evaluator = SWEBenchEvaluator(runtime="modal", max_workers=3)
    evaluator._buffer["django__django-123"] = "diff --git a/a b/a"

    with (
        patch("builtins.open", mock_open()),
        patch("subprocess.run") as run,
    ):
        evaluator.finalize(tmp_path)

    args, kwargs = run.call_args
    command = args[0]
    assert command[1:3] == ["-m", "agent_cap.agents.swebench_modal_compat"]
    assert command[-2:] == ["--modal", "True"]
    assert kwargs["cwd"] == tmp_path
