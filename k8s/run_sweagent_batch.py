#!/usr/bin/env python3
"""Run SWE-bench Pro with SWE-agent + swerex sidecars + shared vLLM.

For each task:
  1. Create K8s sidecar Job with correct Docker image + swerex server
  2. Port-forward sidecar to local port
  3. Run `sweagent run` connecting to sidecar (remote) + vLLM (port-forward)
  4. Collect patch from trajectory
  5. Delete sidecar

Usage:
    # 1. Start vLLM: kubectl create -f k8s/vllm-service.yaml
    # 2. Port-forward vLLM: kubectl port-forward job/<name> 30002:30002 -n eidf230ns &
    # 3. Run:
    python -u run_sweagent_batch.py --vllm-job <name> --num-tasks 100 --concurrency 10
"""
import argparse
import json
import os
import subprocess
import sys
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


def kubectl(*args):
    return subprocess.run(["kubectl"] + list(args), capture_output=True, text=True)


class PortForwardManager:
    def __init__(self, job_name, remote_port=30002, local_port=30002, namespace="eidf230ns"):
        self.job_name = job_name
        self.remote_port = remote_port
        self.local_port = local_port
        self.ns = namespace
        self._proc = None
        self._lock = Lock()

    def ensure_alive(self):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                log(f"[port-forward] (re)starting localhost:{self.local_port} -> {self.job_name}:{self.remote_port}")
                self._proc = subprocess.Popen(
                    ["kubectl", "port-forward", f"job/{self.job_name}",
                     f"{self.local_port}:{self.remote_port}", "-n", self.ns],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                time.sleep(2)
            for _ in range(5):
                try:
                    urllib.request.urlopen(f"http://localhost:{self.local_port}/v1/models", timeout=5)
                    return True
                except Exception:
                    time.sleep(2)
                    if self._proc.poll() is not None:
                        self._proc = subprocess.Popen(
                            ["kubectl", "port-forward", f"job/{self.job_name}",
                             f"{self.local_port}:{self.remote_port}", "-n", self.ns],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                        time.sleep(2)
            return False

    def stop(self):
        if self._proc:
            self._proc.kill()
            self._proc.wait()


def create_sidecar(task_idx, dockerhub_tag):
    """Create swerex sidecar. Returns (job_name, local_port, pf_proc)."""
    job_yaml = json.dumps({
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "generateName": f"swe-rex-{task_idx:03d}-",
            "namespace": "eidf230ns",
            "labels": {"app": "sweagent-sidecar", "task-index": str(task_idx),
                        "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue"},
        },
        "spec": {"backoffLimit": 0, "template": {
            "metadata": {"labels": {"app": "sweagent-sidecar", "task-index": str(task_idx),
                                     "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue"}},
            "spec": {
                "restartPolicy": "Never",
                "containers": [{
                    "name": "swebench",
                    "image": f"jefzda/sweap-images:{dockerhub_tag}",
                    "command": ["bash", "-c"],
                    "args": [
                        "pip install --break-system-packages 'swe-rex>=1.4.0' 2>&1 | tail -1 && "
                        "git config --global --add safe.directory '*' && "
                        "python3 -m swerex --port 9999 --auth-token token123"
                    ],
                    "ports": [{"containerPort": 9999}],
                    "workingDir": "/app",
                    "env": [{"name": "PIP_BREAK_SYSTEM_PACKAGES", "value": "1"},
                            {"name": "PIP_INDEX_URL", "value": "https://pypi.org/simple/"}],
                    "resources": {"requests": {"cpu": "1", "memory": "4Gi"},
                                  "limits": {"cpu": "2", "memory": "8Gi"}},
                }],
            },
        }},
    })

    r = subprocess.run(
        ["kubectl", "create", "-f", "-", "-n", "eidf230ns", "-o", "jsonpath={.metadata.name}"],
        input=job_yaml, capture_output=True, text=True,
    )
    if r.returncode != 0:
        log(f"[task {task_idx}] sidecar create failed: {r.stderr[:200]}")
        return None, None, None
    job_name = r.stdout.strip()

    # Wait for pod running
    for _ in range(120):
        r = kubectl("get", "pods", "-n", "eidf230ns", f"-l=job-name={job_name}",
                     "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}")
        parts = r.stdout.strip().split("|")
        phase, pod_name = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
        if phase == "Running" and pod_name:
            break
        if phase in ("Failed", "Succeeded"):
            return job_name, None, None
        time.sleep(3)
    else:
        return job_name, None, None

    # Wait for swerex server (pip install takes time)
    local_port = 18800 + task_idx
    pf_proc = subprocess.Popen(
        ["kubectl", "port-forward", pod_name, f"{local_port}:9999", "-n", "eidf230ns"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    for _ in range(120):
        try:
            req = urllib.request.Request(
                f"http://localhost:{local_port}/is_alive",
                headers={"X-API-Key": "token123"},
            )
            urllib.request.urlopen(req, timeout=5)
            log(f"[task {task_idx}] swerex ready localhost:{local_port}")
            return job_name, local_port, pf_proc
        except Exception:
            time.sleep(3)

    log(f"[task {task_idx}] swerex not ready")
    pf_proc.kill()
    return job_name, None, None


def delete_sidecar(job_name, pf_proc=None):
    if pf_proc:
        pf_proc.kill()
        try: pf_proc.wait(timeout=5)
        except: pass
    if job_name:
        kubectl("delete", "job", job_name, "-n", "eidf230ns",
                "--wait=false", "--ignore-not-found=true")


def run_one_task(task_idx, instance_id, dockerhub_tag, problem_statement,
                 vllm_port, output_dir, sweagent_dir, pf_mgr):
    """Run SWE-agent on one task."""
    log(f"[task {task_idx}] START {instance_id[:60]}")

    if pf_mgr:
        pf_mgr.ensure_alive()

    job_name, sidecar_port, pf_proc = create_sidecar(task_idx, dockerhub_tag)
    if not sidecar_port:
        delete_sidecar(job_name, pf_proc)
        return {"index": task_idx, "instance_id": instance_id, "status": "sidecar_failed"}

    try:
        task_output = output_dir / f"task_{task_idx:03d}"
        task_output.mkdir(parents=True, exist_ok=True)

        # Write problem statement to file (avoid shell quoting)
        ps_file = task_output / "problem.txt"
        ps_file.write_text(problem_statement)

        cmd = [
            sys.executable, "-m", "sweagent", "run",
            "--config", str(sweagent_dir / "config" / "bash_only.yaml"),
            "--agent.model.name", "hosted_vllm/openai/gpt-oss-120b",
            "--agent.model.api_base", f"http://localhost:{vllm_port}/v1",
            "--agent.model.per_instance_cost_limit", "0",
            "--agent.model.total_cost_limit", "0",
            "--agent.model.per_instance_call_limit", "50",
            "--agent.templates.put_demos_in_history", "false",
            "--env.deployment.type", "remote",
            "--env.deployment.host", "http://127.0.0.1",
            "--env.deployment.port", str(sidecar_port),
            "--env.deployment.auth_token", "token123",
            "--env.repo.type", "preexisting",
            "--env.repo.repo_name", "app",
            "--problem_statement.path", str(ps_file),
        ]

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "dummy"

        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=1800, cwd=str(sweagent_dir))

        # Find trajectory
        traj_files = list(Path(sweagent_dir / "trajectories").rglob("*.traj"))
        patch = ""
        for tf in sorted(traj_files, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                traj = json.loads(tf.read_text())
                p = traj.get("info", {}).get("submission", "") or traj.get("info", {}).get("model_patch", "")
                if p:
                    patch = p
                    # Copy traj to our output dir
                    (task_output / "trajectory.traj").write_text(tf.read_text())
                    break
            except Exception:
                continue

        if patch:
            (task_output / "patch.diff").write_text(patch)
            log(f"[task {task_idx}] DONE patch={len(patch)} chars")
        else:
            log(f"[task {task_idx}] DONE no patch")

        return {"index": task_idx, "instance_id": instance_id,
                "status": "ok", "has_patch": bool(patch)}

    except subprocess.TimeoutExpired:
        log(f"[task {task_idx}] TIMEOUT")
        return {"index": task_idx, "instance_id": instance_id, "status": "timeout"}
    except Exception as exc:
        log(f"[task {task_idx}] ERROR: {exc}")
        return {"index": task_idx, "instance_id": instance_id, "status": f"error: {exc}"}
    finally:
        delete_sidecar(job_name, pf_proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-job", required=True)
    parser.add_argument("--vllm-port", type=int, default=30002)
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/sweagent_100")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--sweagent-dir", type=str, default="/tmp/swe_agent")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweagent_dir = Path(args.sweagent_dir)

    log("Loading SWE-bench Pro dataset...")
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")

    pf_mgr = PortForwardManager(args.vllm_job, local_port=args.vllm_port)
    pf_mgr.ensure_alive()

    task_range = range(args.task_offset, min(args.task_offset + args.num_tasks, len(ds)))
    log(f"Running {len(task_range)} tasks with concurrency={args.concurrency}")

    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {}
            for i in task_range:
                ex = ds[i]
                f = pool.submit(
                    run_one_task, i, ex["instance_id"], ex["dockerhub_tag"],
                    ex["problem_statement"], args.vllm_port, output_dir,
                    sweagent_dir, pf_mgr,
                )
                futures[f] = i

            for f in as_completed(futures):
                result = f.result()
                results.append(result)
                done = len(results)
                patches = sum(1 for r in results if r.get("has_patch"))
                log(f"[progress] {done}/{len(task_range)} done, {patches} patches — "
                    f"{result['instance_id'][:50]}: {result['status']}")
    finally:
        pf_mgr.stop()

    results.sort(key=lambda x: x["index"])
    with open(output_dir / "batch_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    ok = sum(1 for r in results if r["status"] == "ok")
    patches = sum(1 for r in results if r.get("has_patch"))
    log(f"\nDone! {ok}/{len(results)} ok, {patches} patches. Summary: {output_dir}/batch_summary.json")


if __name__ == "__main__":
    main()
