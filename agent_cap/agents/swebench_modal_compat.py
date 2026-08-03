"""Run the upstream SWE-bench Modal harness with legacy editable-install support.

Some SWE-bench repositories use setuptools backends that predate PEP 660.
Modern pip versions reject ``pip install -e`` for those repositories instead
of falling back to ``setup.py develop``.  SWE-bench builds Modal images from a
fresh base image, so pinning pip in AgentCAP's own environment cannot affect
the pip version that performs the repository install.

This module is a drop-in entry point for
``swebench.harness.run_evaluation``.  It leaves successful editable installs
alone and retries only a failed editable install after installing ``pip<25``
inside that task's Modal image.
"""
from __future__ import annotations

import runpy
import shlex
from dataclasses import replace
from importlib.util import find_spec
from typing import Any, Iterable, List, Optional


def _editable_install_python(command: str) -> Optional[str]:
    """Return the Python executable used by an editable pip install command."""

    try:
        tokens = shlex.split(command)
    except ValueError:
        return None

    for index in range(len(tokens) - 3):
        python = tokens[index]
        if not (
            python == "python"
            or python == "python3"
            or python.startswith("python3.")
        ):
            continue
        if tokens[index + 1 : index + 4] != ["-m", "pip", "install"]:
            continue
        install_args = tokens[index + 4 :]
        if "-e" in install_args or "--editable" in install_args or any(
            arg.startswith("--editable=") for arg in install_args
        ):
            return python
    return None


def add_legacy_editable_fallback(command: str) -> str:
    """Retry a failed editable install with a pip that supports legacy backends."""

    python = _editable_install_python(command)
    if python is None:
        return command
    return (
        f"{command} || {{\n"
        "  echo 'AgentCAP: editable install failed; retrying with pip<25' >&2\n"
        f"  {python} -m pip install 'pip<25'\n"
        f"  {command}\n"
        "}"
    )


def patch_repo_script_list(commands: Iterable[str]) -> List[str]:
    """Return SWE-bench repository setup commands with the scoped fallback."""

    return [add_legacy_editable_fallback(command) for command in commands]


def patch_test_spec(test_spec: Any) -> Any:
    """Copy a TestSpec and patch its repository setup without mutating shared data."""

    commands = patch_repo_script_list(test_spec.repo_script_list)
    if commands == test_spec.repo_script_list:
        return test_spec
    return replace(test_spec, repo_script_list=commands)


def install_modal_compatibility() -> None:
    """Patch the Modal image builder before the upstream CLI loads test specs."""

    from swebench.harness.modal_eval.run_evaluation_modal import ModalSandboxRuntime

    current = ModalSandboxRuntime.get_instance_image
    if getattr(current, "_agentcap_legacy_editable_compat", False):
        return

    def get_instance_image(test_spec: Any) -> Any:
        return current(patch_test_spec(test_spec))

    get_instance_image._agentcap_legacy_editable_compat = True  # type: ignore[attr-defined]
    ModalSandboxRuntime.get_instance_image = staticmethod(get_instance_image)


def main() -> None:
    install_modal_compatibility()
    runner = find_spec("swebench.harness.run_evaluation")
    if runner is None or runner.origin is None:
        raise RuntimeError("cannot locate swebench.harness.run_evaluation")
    runpy.run_path(runner.origin, run_name="__main__")


if __name__ == "__main__":
    main()
