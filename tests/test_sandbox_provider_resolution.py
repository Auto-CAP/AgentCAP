"""Unit tests for the provider-resolution seam in sandbox_providers.py
(workstream A1: dotted-path resolution for --sandbox-provider/--exec-provider).

Covers get_sandbox_provider / get_exec_provider: http(s) URL handling
(sandbox only), dotted-path "package.module:ClassName" resolution, registry
fallback, and that a bad path raises a clear ValueError naming the failing
module/class. The dotted-path cases monkeypatch importlib.import_module so
they can prove resolution against the real contract path
(teasbench.sandbox.k8s:InClusterK8sProvider, see docs/REMOVING_K8S_FROM_AGENTCAP.md
and IMPLEMENTATION_SPEC.md section 2) without TEASBench actually being
installed here. No kubectl, no cluster, no network.
"""
from unittest import mock

import agent_cap.agents.sandbox_providers as sandbox_providers


# ---------------------------------------------------------------------------
# get_sandbox_provider
# ---------------------------------------------------------------------------

def test_sandbox_http_url_unchanged():
    p = sandbox_providers.get_sandbox_provider("http://broker.example:8080/sandbox")
    assert isinstance(p, sandbox_providers.HttpSandboxProvider)
    assert p.base_url == "http://broker.example:8080/sandbox"


def test_sandbox_https_url_unchanged():
    p = sandbox_providers.get_sandbox_provider("https://broker.example/sandbox")
    assert isinstance(p, sandbox_providers.HttpSandboxProvider)


def test_sandbox_registry_lookup_unchanged():
    p = sandbox_providers.get_sandbox_provider("k8s")
    assert isinstance(p, sandbox_providers.K8sSandboxProvider)


def test_sandbox_unknown_registry_name_raises_valueerror():
    try:
        sandbox_providers.get_sandbox_provider("no-such-provider")
        assert False, "expected ValueError"
    except ValueError as exc:
        msg = str(exc)
        assert "no-such-provider" in msg
        assert "k8s" in msg
        assert "dotted path" in msg


def test_sandbox_dotted_path_resolves_class_and_kwargs():
    """Proves the exact §2 contract path resolves: teasbench.sandbox.k8s
    isn't installed here, so importlib.import_module is monkeypatched to
    return a stand-in module exposing the target class."""

    class InClusterK8sProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = mock.MagicMock()
    fake_module.InClusterK8sProvider = InClusterK8sProvider

    with mock.patch.object(sandbox_providers.importlib, "import_module",
                            return_value=fake_module) as m:
        p = sandbox_providers.get_sandbox_provider(
            "teasbench.sandbox.k8s:InClusterK8sProvider", namespace="eidf230ns")
        m.assert_called_once_with("teasbench.sandbox.k8s")

    assert isinstance(p, InClusterK8sProvider)
    assert p.kwargs == {"namespace": "eidf230ns"}


def test_sandbox_dotted_path_bad_module_raises_clear_error():
    with mock.patch.object(sandbox_providers.importlib, "import_module",
                            side_effect=ModuleNotFoundError("No module named 'nope'")):
        try:
            sandbox_providers.get_sandbox_provider("nope.module:Whatever")
            assert False, "expected ValueError"
        except ValueError as exc:
            msg = str(exc)
            assert "nope.module:Whatever" in msg
            assert "nope.module" in msg


def test_sandbox_dotted_path_bad_class_raises_clear_error():
    fake_module = mock.MagicMock(spec=[])  # no attributes at all
    with mock.patch.object(sandbox_providers.importlib, "import_module",
                            return_value=fake_module):
        try:
            sandbox_providers.get_sandbox_provider("teasbench.sandbox.k8s:NoSuchClass")
            assert False, "expected ValueError"
        except ValueError as exc:
            msg = str(exc)
            assert "teasbench.sandbox.k8s:NoSuchClass" in msg
            assert "NoSuchClass" in msg


# ---------------------------------------------------------------------------
# get_exec_provider (same resolution minus the http(s) case)
# ---------------------------------------------------------------------------

def test_exec_registry_lookup_unchanged():
    p = sandbox_providers.get_exec_provider("k8s")
    assert isinstance(p, sandbox_providers.K8sExecProvider)


def test_exec_unknown_registry_name_raises_valueerror():
    try:
        sandbox_providers.get_exec_provider("no-such-provider")
        assert False, "expected ValueError"
    except ValueError as exc:
        msg = str(exc)
        assert "no-such-provider" in msg
        assert "k8s" in msg
        assert "dotted path" in msg


def test_exec_dotted_path_resolves_class_and_kwargs():
    class InClusterK8sProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = mock.MagicMock()
    fake_module.InClusterK8sProvider = InClusterK8sProvider

    with mock.patch.object(sandbox_providers.importlib, "import_module",
                            return_value=fake_module) as m:
        p = sandbox_providers.get_exec_provider(
            "teasbench.sandbox.k8s:InClusterK8sProvider", namespace="eidf230ns")
        m.assert_called_once_with("teasbench.sandbox.k8s")

    assert isinstance(p, InClusterK8sProvider)
    assert p.kwargs == {"namespace": "eidf230ns"}


def test_exec_dotted_path_bad_module_raises_clear_error():
    with mock.patch.object(sandbox_providers.importlib, "import_module",
                            side_effect=ModuleNotFoundError("No module named 'nope'")):
        try:
            sandbox_providers.get_exec_provider("nope.module:Whatever")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "nope.module:Whatever" in str(exc)


# ---------------------------------------------------------------------------
# ExecHandle / ExecProvider shape, and K8sExecProvider wiring
# (no kubectl: K8sExecContainer.start/stop are monkeypatched)
# ---------------------------------------------------------------------------

def test_exec_handle_and_provider_define_expected_methods():
    assert hasattr(sandbox_providers.ExecHandle, "cp")
    assert hasattr(sandbox_providers.ExecHandle, "exec")
    assert hasattr(sandbox_providers.ExecHandle, "stop")
    assert hasattr(sandbox_providers.ExecProvider, "acquire_exec")
    assert hasattr(sandbox_providers.ExecProvider, "release_exec")


def test_k8s_exec_provider_acquire_release_wiring():
    """acquire_exec should construct+start a K8sExecContainer and return it;
    release_exec should stop it. Mirrors K8sSandboxProvider.acquire/release,
    with start()/stop() stubbed out so no kubectl is invoked."""
    calls = []
    with mock.patch.object(sandbox_providers.K8sExecContainer, "start",
                            lambda self, *a, **k: calls.append("start")), \
         mock.patch.object(sandbox_providers.K8sExecContainer, "stop",
                            lambda self: calls.append("stop")):
        provider = sandbox_providers.K8sExecProvider(namespace="eidf230ns")
        box = provider.acquire_exec("docker.io/swebench/sweb.eval.x86_64.foo:latest", "foo")
        assert isinstance(box, sandbox_providers.K8sExecContainer)
        assert box.namespace == "eidf230ns"
        assert calls == ["start"]

        provider.release_exec(box)
        assert calls == ["start", "stop"]


def test_k8s_exec_provider_cleans_up_job_on_failed_start():
    """If start() fails after the Job was created, acquire_exec must still
    call stop() (to delete the orphaned Job) before re-raising — the same
    cleanup-then-raise pattern as K8sSandboxProvider.acquire."""
    calls = []

    def failing_start(self, *a, **k):
        calls.append("start")
        raise RuntimeError("pod never became Running")

    with mock.patch.object(sandbox_providers.K8sExecContainer, "start", failing_start), \
         mock.patch.object(sandbox_providers.K8sExecContainer, "stop",
                            lambda self: calls.append("stop")):
        provider = sandbox_providers.K8sExecProvider(namespace="eidf230ns")
        try:
            provider.acquire_exec("docker.io/swebench/sweb.eval.x86_64.foo:latest", "foo")
            assert False, "expected RuntimeError to propagate"
        except RuntimeError as exc:
            assert "pod never became Running" in str(exc)
        assert calls == ["start", "stop"]


def test_k8s_exec_provider_default_namespace_from_env():
    import os

    old = os.environ.get("SWEBENCH_K8S_NAMESPACE")
    os.environ["SWEBENCH_K8S_NAMESPACE"] = "some-other-ns"
    try:
        provider = sandbox_providers.K8sExecProvider()
        assert provider.namespace == "some-other-ns"
    finally:
        if old is None:
            os.environ.pop("SWEBENCH_K8S_NAMESPACE", None)
        else:
            os.environ["SWEBENCH_K8S_NAMESPACE"] = old


if __name__ == "__main__":
    # No pytest in this environment's venv; run directly for a self-check.
    # (A real pytest, if present, collects the test_* functions as usual.)
    tests = [(n, f) for n, f in sorted(globals().items())
              if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - test runner, want to catch all
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
