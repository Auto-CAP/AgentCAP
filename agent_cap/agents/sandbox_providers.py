"""Sandbox / container providers — the deployment side of the AgentCAP↔TEAS
interface.

AgentCAP (benchmark semantics) consumes only endpoints:
  - an OpenAI-compatible LLM URL (passed via --base-url), and
  - per task, a sandbox endpoint {host, port, auth_token} speaking the
    swe-rex protocol (SWE-agent's ``remote`` deployment type), and
  - per eval, an exec-container handle (upload files / run commands) built
    from the official swebench instance image.

HOW those endpoints come to exist (k8s pods, local docker, modal, vast.ai…)
is a deployment-scenario concern. This module holds the provider interface
plus the k8s implementation used on EIDF; implementations are selected by
name via :func:`get_sandbox_provider` and are expected to migrate to
TEASBench, which owns deployment scenarios.

The k8s code here is a verbatim extraction of what previously lived inline
in ``strategies_sweagent.py`` (swe-rex sidecars) and
``evaluators_swebench.py`` (eval pods): per-task Job from the official
swebench instance image + ``kubectl port-forward`` on an OS-assigned local
port.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_K8S_PORT_LOCK = threading.Lock()
_K8S_AUTH_TOKEN = os.environ.get("SWEBENCH_K8S_AUTH_TOKEN", "agentcap-swerex")


def _k8s_next_port() -> int:
    """OS-assigned free port. A fixed counter base collides with stale
    kubectl port-forwards left by a previous crashed run — SWE-agent then
    talks to the OLD sidecar and dies with SessionExistsError."""
    with _K8S_PORT_LOCK:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _kubectl(namespace: str, *args: str, input_text: Optional[str] = None,
             timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "-n", namespace, *args],
        input=input_text, capture_output=True, text=True, timeout=timeout,
    )


@dataclass
class SandboxEndpoint:
    """A running swe-rex server the agent can attach to (deployment-agnostic)."""
    host: str
    port: int
    auth_token: str
    handle: Any = field(default=None, repr=False)  # provider-private


class SandboxProvider:
    """Provides swe-rex sandbox endpoints from container images.

    Must be parameterized by image: SWE-bench uses one official image per
    instance, so a pre-warmed fixed pool cannot work.
    """

    def acquire(self, image: str, label: str = "") -> SandboxEndpoint:
        raise NotImplementedError

    def release(self, endpoint: SandboxEndpoint) -> None:
        raise NotImplementedError


class _K8sSidecar:
    """One swe-rex sidecar pod running the official swebench instance image."""

    def __init__(self, namespace: str, image: str, instance_id: str):
        self.namespace = namespace
        self.image = image
        self.instance_id = instance_id
        self.job_name: Optional[str] = None
        self.pod_name: Optional[str] = None
        self.local_port: Optional[int] = None
        self.pf_proc: Optional[subprocess.Popen] = None
        self._stopped = False
        self._pf_keeper: Optional[threading.Thread] = None

    def start(self, pod_timeout_s: int = 1200, swerex_timeout_s: int = 600) -> None:
        queue = f"{self.namespace}-user-queue"
        job = {
            "apiVersion": "batch/v1", "kind": "Job",
            "metadata": {
                "generateName": "swe-rex-",
                "namespace": self.namespace,
                "labels": {"app": "sweagent-sidecar",
                           "kueue.x-k8s.io/queue-name": queue},
            },
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 600,
                "activeDeadlineSeconds": 6 * 3600,
                "template": {
                    "metadata": {"labels": {"app": "sweagent-sidecar"}},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "swebench",
                            "image": self.image,
                            "command": ["/bin/bash", "-c"],
                            "args": [
                                "set -e; "
                                "git config --global --add safe.directory '*'; "
                                "python3 -m pip install --quiet --no-input 'swe-rex>=1.4.0' && "
                                f"exec python3 -m swerex --port 9999 --auth-token {_K8S_AUTH_TOKEN}"
                            ],
                            "ports": [{"containerPort": 9999}],
                            "env": [{"name": "PIP_BREAK_SYSTEM_PACKAGES", "value": "1"}],
                            "resources": {
                                "requests": {"cpu": "1", "memory": "4Gi"},
                                "limits": {"cpu": "2", "memory": "24Gi"},
                            },
                        }],
                    },
                },
            },
        }
        r = _kubectl(self.namespace, "create", "-f", "-",
                     "-o", "jsonpath={.metadata.name}", input_text=json.dumps(job))
        if r.returncode != 0:
            raise RuntimeError(f"sidecar job create failed: {r.stderr[:300]}")
        self.job_name = r.stdout.strip()

        deadline = time.time() + pod_timeout_s
        while time.time() < deadline:
            r = _kubectl(self.namespace, "get", "pods", f"-l=job-name={self.job_name}",
                         "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}")
            parts = r.stdout.strip().split("|")
            phase, pod = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
            if phase == "Running" and pod:
                self.pod_name = pod
                break
            if phase == "Failed":
                raise RuntimeError(f"sidecar pod failed for {self.instance_id}")
            time.sleep(5)
        else:
            raise RuntimeError(f"sidecar pod not Running after {pod_timeout_s}s "
                               f"({self.instance_id})")

        self.local_port = _k8s_next_port()
        # start_new_session: detach the tunnel from the caller's process
        # group so a terminal/session teardown can't kill it mid-task
        # (observed: session flap killed in-flight sidecar tunnels ->
        # "Cannot connect to host 127.0.0.1:<port>" -> task rc=1).
        self.pf_proc = subprocess.Popen(
            ["kubectl", "-n", self.namespace, "port-forward",
             f"pod/{self.pod_name}", f"{self.local_port}:9999"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.time() + swerex_timeout_s
        while time.time() < deadline:
            if self.pf_proc.poll() is not None:
                # port-forward died (e.g. pod still pip-installing) — restart it
                self.pf_proc = subprocess.Popen(
                    ["kubectl", "-n", self.namespace, "port-forward",
                     f"pod/{self.pod_name}", f"{self.local_port}:9999"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(3)
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.local_port}/is_alive",
                    headers={"X-API-Key": _K8S_AUTH_TOKEN},
                )
                urllib.request.urlopen(req, timeout=5)
                # Babysit the tunnel for the task's whole lifetime — kubectl
                # port-forward occasionally drops mid-task, which otherwise
                # kills the agent with "Cannot connect to 127.0.0.1:<port>".
                self._pf_keeper = threading.Thread(
                    target=self._keep_pf_alive, daemon=True)
                self._pf_keeper.start()
                return
            except Exception:
                time.sleep(3)
        raise RuntimeError(f"swerex not alive after {swerex_timeout_s}s ({self.instance_id})")

    def _keep_pf_alive(self) -> None:
        while not self._stopped:
            proc = self.pf_proc
            if proc is not None and proc.poll() is not None and not self._stopped:
                try:
                    self.pf_proc = subprocess.Popen(
                        ["kubectl", "-n", self.namespace, "port-forward",
                         f"pod/{self.pod_name}", f"{self.local_port}:9999"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                except Exception:
                    pass
            time.sleep(2)

    def stop(self) -> None:
        self._stopped = True
        if self.pf_proc is not None:
            self.pf_proc.kill()
            try:
                self.pf_proc.wait(timeout=5)
            except Exception:
                pass
        if self.job_name:
            try:
                _kubectl(self.namespace, "delete", "job", self.job_name,
                         "--wait=false", "--ignore-not-found=true")
            except Exception:
                pass


class K8sSandboxProvider(SandboxProvider):
    """EIDF k8s implementation: sidecar Job + port-forward per acquire()."""

    def __init__(self, namespace: Optional[str] = None):
        self.namespace = (namespace
                          or os.environ.get("SWEBENCH_K8S_NAMESPACE", "eidf230ns"))

    def acquire(self, image: str, label: str = "") -> SandboxEndpoint:
        sidecar = _K8sSidecar(self.namespace, image, label)
        try:
            sidecar.start()
        except Exception:
            sidecar.stop()
            raise
        return SandboxEndpoint(
            host="http://127.0.0.1",
            port=sidecar.local_port,
            auth_token=_K8S_AUTH_TOKEN,
            handle=sidecar,
        )

    def release(self, endpoint: SandboxEndpoint) -> None:
        if endpoint is not None and isinstance(endpoint.handle, _K8sSidecar):
            endpoint.handle.stop()


class HttpSandboxProvider(SandboxProvider):
    """External broker (e.g. TEASBench) speaking a 2-route HTTP contract:

        POST <base>/acquire  {"image": str, "label": str}
            -> 200 {"host": str, "port": int, "auth_token": str,
                    "handle": <opaque, echoed back on release>}
        POST <base>/release  {"handle": <opaque>}
            -> 200

    The returned {host, port, auth_token} must be a live swe-rex server
    reachable from where AgentCAP runs. Everything else (which cluster,
    which substrate, pooling, placement) is the broker's business.
    """

    def __init__(self, base_url: str, timeout_s: int = 1800):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/{route}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode() or "{}"
        return json.loads(body)

    def acquire(self, image: str, label: str = "") -> SandboxEndpoint:
        r = self._post("acquire", {"image": image, "label": label})
        return SandboxEndpoint(
            host=r["host"],
            port=int(r["port"]),
            auth_token=r.get("auth_token", ""),
            handle=r.get("handle"),
        )

    def release(self, endpoint: SandboxEndpoint) -> None:
        if endpoint is None:
            return
        try:
            self._post("release", {"handle": endpoint.handle})
        except Exception:
            pass


_PROVIDERS = {
    "k8s": K8sSandboxProvider,
}


def get_sandbox_provider(name: str, **kwargs: Any) -> SandboxProvider:
    """Resolve a provider by registry name, or by http(s) URL (external
    broker, e.g. TEASBench)."""
    if name.startswith(("http://", "https://")):
        return HttpSandboxProvider(name)
    try:
        cls = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown sandbox provider '{name}'. "
                         f"Available: {sorted(_PROVIDERS)} or an http(s) "
                         f"broker URL") from None
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Eval-side exec containers (upload files / run commands in an instance
# image). Used by the swebench-k8s evaluator; same extraction rationale.
# ---------------------------------------------------------------------------

class K8sExecContainer:
    """A pod from an instance image supporting cp/exec, for harness eval."""

    def __init__(self, namespace: str, image: str):
        self.namespace = namespace
        self.image = image
        self.job_name: str = ""
        self.pod_name: str = ""

    def start(self, pod_timeout_s: int = 1200) -> None:
        queue = f"{self.namespace}-user-queue"
        job = {
            "apiVersion": "batch/v1", "kind": "Job",
            "metadata": {"generateName": "swe-eval-", "namespace": self.namespace,
                         "labels": {"app": "swebench-eval",
                                    "kueue.x-k8s.io/queue-name": queue}},
            "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": 600,
                     "activeDeadlineSeconds": 3 * 3600,
                     "template": {"metadata": {"labels": {"app": "swebench-eval"}},
                                  "spec": {"restartPolicy": "Never", "containers": [{
                                      "name": "eval", "image": self.image,
                                      "command": ["sleep", "10800"],
                                      "resources": {
                                          "requests": {"cpu": "1", "memory": "4Gi"},
                                          "limits": {"cpu": "2", "memory": "24Gi"}},
                                  }]}}},
        }
        r = _kubectl(self.namespace, "create", "-f", "-",
                     "-o", "jsonpath={.metadata.name}", input_text=json.dumps(job))
        if r.returncode != 0:
            raise RuntimeError(f"job create: {r.stderr[:200]}")
        self.job_name = r.stdout.strip()
        deadline = time.time() + pod_timeout_s
        while time.time() < deadline:
            r = _kubectl(self.namespace, "get", "pods", f"-l=job-name={self.job_name}",
                         "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}")
            parts = r.stdout.strip().split("|")
            phase, pod = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
            if phase == "Running" and pod:
                self.pod_name = pod
                return
            if phase == "Failed":
                raise RuntimeError("eval pod failed")
            time.sleep(5)
        raise RuntimeError("eval pod timeout")

    def cp(self, local_path: str, remote_path: str, timeout: int = 300) -> None:
        _kubectl(self.namespace, "cp", local_path,
                 f"{self.pod_name}:{remote_path}", timeout=timeout)

    def exec(self, command: str, timeout: int = 300) -> subprocess.CompletedProcess:
        return _kubectl(self.namespace, "exec", self.pod_name, "--",
                        "bash", "-c", command, timeout=timeout)

    def stop(self) -> None:
        if self.job_name:
            try:
                _kubectl(self.namespace, "delete", "job", self.job_name,
                         "--wait=false", "--ignore-not-found=true")
            except Exception:
                pass
