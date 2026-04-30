#!/usr/bin/env python3
"""Run SWE-agent on SWE-bench tasks across deployment backends.

Supports four sandbox modes via --deployment:
  - k8s     : per-task swerex sidecar pod (existing behavior; needs kubectl + cluster)
  - docker  : per-task local DockerDeployment (needs docker daemon)
  - local   : per-task LocalDeployment, runs in a tempdir (no isolation; sandbox host only)
  - modal   : per-task ModalDeployment, runs in Modal cloud (needs `modal token new`)

vLLM/SGLang serving the model is orthogonal — pass --vllm-url whatever endpoint
you have (k8s port-forward, native server, modal, etc.).

Usage examples:

  # k8s (original)
  python run_sweagent.py --deployment k8s --vllm-job vllm-gptoss-h100-xxxx \
      --num-tasks 100 --concurrency 10

  # docker
  python run_sweagent.py --deployment docker --vllm-url http://localhost:30002/v1 \
      --num-tasks 100 --concurrency 4

  # local (sandbox host)
  python run_sweagent.py --deployment local --vllm-url http://gpu-host:30002/v1 \
      --num-tasks 100 --concurrency 2

  # modal (no docker / no k8s)
  python run_sweagent.py --deployment modal --vllm-url http://gpu-host:30002/v1 \
      --num-tasks 100 --concurrency 20
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from datasets import load_dataset

print_lock = Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Deployment backends
# ---------------------------------------------------------------------------

class K8sDeploy:
    """Original sidecar-pod path."""

    def __init__(self, namespace="eidf230ns", image_repo="jefzda/sweap-images"):
        self.namespace = namespace
        self.image_repo = image_repo

    def prepare(self, task_idx, instance_id, dockerhub_tag):
        job_yaml = json.dumps({
            "apiVersion": "batch/v1", "kind": "Job",
            "metadata": {
                "generateName": f"swe-rex-{task_idx:03d}-",
                "namespace": self.namespace,
                "labels": {"app": "sweagent-sidecar",
                           "kueue.x-k8s.io/queue-name": f"{self.namespace}-user-queue"},
            },
            "spec": {"backoffLimit": 0, "template": {
                "metadata": {"labels": {"app": "sweagent-sidecar",
                                        "kueue.x-k8s.io/queue-name": f"{self.namespace}-user-queue"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "swebench",
                        "image": f"{self.image_repo}:{dockerhub_tag}",
                        "command": ["bash", "-c"],
                        "args": [
                            "pip install --break-system-packages 'swe-rex>=1.4.0' 2>&1 | tail -1 && "
                            "git config --global --add safe.directory '*' && "
                            "python3 -m swerex --port 9999 --auth-token token123"
                        ],
                        "ports": [{"containerPort": 9999}],
                        "workingDir": "/app",
                        "env": [{"name": "PIP_BREAK_SYSTEM_PACKAGES", "value": "1"}],
                        "resources": {"requests": {"cpu": "1", "memory": "4Gi"},
                                      "limits": {"cpu": "2", "memory": "8Gi"}},
                    }],
                },
            }},
        })
        r = subprocess.run(["kubectl", "create", "-f", "-", "-n", self.namespace,
                            "-o", "jsonpath={.metadata.name}"],
                           input=job_yaml, capture_output=True, text=True)
        if r.returncode != 0:
            return None
        job_name = r.stdout.strip()

        # wait for pod
        for _ in range(120):
            rr = subprocess.run(["kubectl", "get", "pods", "-n", self.namespace,
                                 f"-l=job-name={job_name}",
                                 "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}"],
                                capture_output=True, text=True)
            parts = rr.stdout.strip().split("|")
            phase, pod_name = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
            if phase == "Running" and pod_name:
                break
            time.sleep(3)
        else:
            return None

        local_port = 18800 + task_idx
        pf_proc = subprocess.Popen(
            ["kubectl", "port-forward", pod_name, f"{local_port}:9999", "-n", self.namespace],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        for _ in range(120):
            try:
                req = urllib.request.Request(f"http://localhost:{local_port}/is_alive",
                                             headers={"X-API-Key": "token123"})
                urllib.request.urlopen(req, timeout=5)
                return {"job_name": job_name, "pf_proc": pf_proc, "port": local_port}
            except Exception:
                time.sleep(3)
        pf_proc.kill()
        return None

    def sweagent_args(self, ctx):
        return [
            "--env.deployment.type", "remote",
            "--env.deployment.host", "http://127.0.0.1",
            "--env.deployment.port", str(ctx["port"]),
            "--env.deployment.auth_token", "token123",
            "--env.repo.type", "preexisting",
            "--env.repo.repo_name", "app",
        ]

    def cleanup(self, ctx):
        if not ctx:
            return
        if ctx.get("pf_proc"):
            ctx["pf_proc"].kill()
            try: ctx["pf_proc"].wait(timeout=5)
            except: pass
        if ctx.get("job_name"):
            subprocess.run(["kubectl", "delete", "job", ctx["job_name"],
                            "-n", self.namespace, "--wait=false", "--ignore-not-found=true"],
                           capture_output=True)


class DockerDeploy:
    """Per-task DockerDeployment via sweagent CLI; needs docker daemon."""

    def __init__(self, image_repo="jefzda/sweap-images"):
        self.image_repo = image_repo

    def prepare(self, task_idx, instance_id, dockerhub_tag):
        # nothing to spin up — sweagent CLI starts/stops the container
        return {"image": f"{self.image_repo}:{dockerhub_tag}"}

    def sweagent_args(self, ctx):
        return [
            "--env.deployment.type", "docker",
            "--env.deployment.image", ctx["image"],
            "--env.repo.type", "preexisting",
            "--env.repo.repo_name", "app",
        ]

    def cleanup(self, ctx):
        pass


class LocalDeploy:
    """LocalDeployment — runs in current shell. Caller is responsible for sandbox."""

    def __init__(self, work_root=None):
        self.work_root = Path(work_root or tempfile.gettempdir()) / "sweagent_local"
        self.work_root.mkdir(parents=True, exist_ok=True)

    def prepare(self, task_idx, instance_id, dockerhub_tag):
        # Make a per-task working directory; sweagent will clone repo here.
        wd = self.work_root / f"task_{task_idx:03d}_{instance_id.replace('/', '_')[:40]}"
        if wd.exists():
            shutil.rmtree(wd, ignore_errors=True)
        wd.mkdir(parents=True)
        return {"workdir": str(wd)}

    def sweagent_args(self, ctx):
        return [
            "--env.deployment.type", "local",
            "--env.repo.type", "preexisting",
            "--env.repo.repo_name", "app",
            "--env.deployment.cwd", ctx["workdir"],
        ]

    def cleanup(self, ctx):
        if not ctx:
            return
        wd = ctx.get("workdir")
        if wd and Path(wd).exists():
            shutil.rmtree(wd, ignore_errors=True)


class ModalDeploy:
    """ModalDeployment — runs container in Modal cloud."""

    def __init__(self, image_repo="jefzda/sweap-images"):
        self.image_repo = image_repo

    def prepare(self, task_idx, instance_id, dockerhub_tag):
        return {"image": f"docker.io/{self.image_repo}:{dockerhub_tag}"}

    def sweagent_args(self, ctx):
        return [
            "--env.deployment.type", "modal",
            "--env.deployment.image", ctx["image"],
            "--env.repo.type", "preexisting",
            "--env.repo.repo_name", "app",
        ]

    def cleanup(self, ctx):
        pass


DEPLOYS = {
    "k8s": K8sDeploy,
    "docker": DockerDeploy,
    "local": LocalDeploy,
    "modal": ModalDeploy,
}


# ---------------------------------------------------------------------------
# vLLM/SGLang endpoint plumbing
# ---------------------------------------------------------------------------

class K8sPortForward:
    """Keep `localhost:port -> kube job` alive; restart if it dies."""
    def __init__(self, job_name, port=30002, namespace="eidf230ns"):
        self.job_name = job_name
        self.port = port
        self.ns = namespace
        self._proc = None
        self._lock = Lock()

    def url(self):
        return f"http://localhost:{self.port}/v1"

    def ensure_alive(self):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = subprocess.Popen(
                    ["kubectl", "port-forward", f"job/{self.job_name}",
                     f"{self.port}:{self.port}", "-n", self.ns],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)
            for _ in range(5):
                try:
                    urllib.request.urlopen(f"http://localhost:{self.port}/v1/models", timeout=5)
                    return True
                except Exception:
                    time.sleep(2)
            return False

    def stop(self):
        if self._proc:
            self._proc.kill()
            self._proc.wait()


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

def run_one_task(task_idx, instance_id, dockerhub_tag, problem_statement,
                 vllm_url, output_dir, sweagent_dir, deploy, pf_mgr=None):
    log(f"[task {task_idx}] START {instance_id[:60]}")
    if pf_mgr:
        pf_mgr.ensure_alive()

    ctx = deploy.prepare(task_idx, instance_id, dockerhub_tag)
    if ctx is None:
        return {"index": task_idx, "instance_id": instance_id, "status": "deploy_failed"}

    try:
        task_output = output_dir / f"task_{task_idx:03d}"
        task_output.mkdir(parents=True, exist_ok=True)
        ps_file = task_output / "problem.txt"
        ps_file.write_text(problem_statement)

        cmd = [
            sys.executable, "-m", "sweagent", "run",
            "--config", str(sweagent_dir / "config" / "bash_only.yaml"),
            "--agent.model.name", "hosted_vllm/openai/gpt-oss-120b",
            "--agent.model.api_base", vllm_url,
            "--agent.model.per_instance_cost_limit", "0",
            "--agent.model.total_cost_limit", "0",
            "--agent.model.per_instance_call_limit", "50",
            "--agent.templates.put_demos_in_history", "false",
            "--problem_statement.path", str(ps_file),
        ] + deploy.sweagent_args(ctx)

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "dummy"

        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=1800, cwd=str(sweagent_dir))

        traj_files = list(Path(sweagent_dir / "trajectories").rglob("*.traj"))
        patch = ""
        for tf in sorted(traj_files, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                traj = json.loads(tf.read_text())
                p = traj.get("info", {}).get("submission") or traj.get("info", {}).get("model_patch") or ""
                if p:
                    patch = p
                    (task_output / "trajectory.traj").write_text(tf.read_text())
                    break
            except Exception:
                continue

        if patch:
            (task_output / "patch.diff").write_text(patch)
            log(f"[task {task_idx}] DONE patch={len(patch)} chars")
        else:
            log(f"[task {task_idx}] DONE no patch (rc={r.returncode})")
        return {"index": task_idx, "instance_id": instance_id,
                "status": "ok", "has_patch": bool(patch)}

    except subprocess.TimeoutExpired:
        log(f"[task {task_idx}] TIMEOUT")
        return {"index": task_idx, "instance_id": instance_id, "status": "timeout"}
    except Exception as exc:
        log(f"[task {task_idx}] ERROR: {exc}")
        return {"index": task_idx, "instance_id": instance_id, "status": f"error: {exc}"}
    finally:
        deploy.cleanup(ctx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", choices=list(DEPLOYS.keys()), required=True)
    ap.add_argument("--dataset", default="swe-bench-lite",
                    choices=["swe-bench-lite", "swe-bench-pro"])
    ap.add_argument("--num-tasks", type=int, default=100)
    ap.add_argument("--task-offset", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--sweagent-dir", default="/tmp/swe_agent")
    # vLLM endpoint (mutually exclusive)
    ap.add_argument("--vllm-url", help="Direct HTTP URL, e.g. http://host:30002/v1")
    ap.add_argument("--vllm-job", help="K8s Job name; will port-forward")
    ap.add_argument("--vllm-port", type=int, default=30002)
    ap.add_argument("--namespace", default="eidf230ns")
    ap.add_argument("--image-repo", default="jefzda/sweap-images")
    ap.add_argument("--local-work-root", default=None,
                    help="Where local deployment puts per-task workdirs")
    args = ap.parse_args()

    if not (args.vllm_url or args.vllm_job):
        ap.error("Either --vllm-url or --vllm-job must be provided")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweagent_dir = Path(args.sweagent_dir)

    log(f"Loading dataset {args.dataset}...")
    if args.dataset == "swe-bench-lite":
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        # SWE-bench_Lite tasks lack dockerhub_tag; the standard image is
        # swebench/sweb.eval.x86_64.<instance_id_lower_with_underscores>:latest
        def _tag(ex):
            iid = ex["instance_id"].lower().replace("/", "__")
            return f"swebench/sweb.eval.x86_64.{iid}"
        get_image = _tag
    else:
        ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
        get_image = lambda ex: ex["dockerhub_tag"]

    pf_mgr = None
    if args.vllm_job:
        pf_mgr = K8sPortForward(args.vllm_job, args.vllm_port, args.namespace)
        pf_mgr.ensure_alive()
        vllm_url = f"http://localhost:{args.vllm_port}/v1"
    else:
        vllm_url = args.vllm_url

    DeployCls = DEPLOYS[args.deployment]
    if args.deployment == "local":
        deploy = DeployCls(work_root=args.local_work_root)
    elif args.deployment in ("docker", "modal"):
        deploy = DeployCls(image_repo=args.image_repo)
    elif args.deployment == "k8s":
        deploy = DeployCls(namespace=args.namespace, image_repo=args.image_repo)
    else:
        deploy = DeployCls()

    task_range = range(args.task_offset, min(args.task_offset + args.num_tasks, len(ds)))
    log(f"Running {len(task_range)} tasks  deployment={args.deployment}  concurrency={args.concurrency}")

    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {}
            for i in task_range:
                ex = ds[i]
                f = pool.submit(
                    run_one_task, i, ex["instance_id"], get_image(ex),
                    ex.get("problem_statement", ex.get("problem_text", "")),
                    vllm_url, output_dir, sweagent_dir, deploy, pf_mgr,
                )
                futures[f] = i
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                done = len(results)
                patches = sum(1 for x in results if x.get("has_patch"))
                log(f"[progress] {done}/{len(task_range)} done, {patches} patches — "
                    f"{r['instance_id'][:50]}: {r['status']}")
    finally:
        if pf_mgr:
            pf_mgr.stop()

    results.sort(key=lambda x: x["index"])
    with open(output_dir / "batch_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    ok = sum(1 for r in results if r["status"] == "ok")
    patches = sum(1 for r in results if r.get("has_patch"))
    log(f"\nDone! {ok}/{len(results)} ok, {patches} patches. Summary: {output_dir}/batch_summary.json")


if __name__ == "__main__":
    main()
