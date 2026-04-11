#!/usr/bin/env python3
"""Orchestrate SWE-bench Pro batch run from login node.

Architecture:
  - vLLM runs as a K8s Job, port-forwarded to localhost:30002
  - For each task, create a sidecar Job (correct Docker image),
    port-forward its exec server, run unified_runner, clean up.
  - Multiple tasks run concurrently via ThreadPoolExecutor.

Usage:
    # 1. Start vLLM:  kubectl create -f k8s/vllm-service.yaml
    # 2. Port-forward: kubectl port-forward job/<name> 30002:30002 -n eidf230ns &
    # 3. Run:
    python -u run_swebench_batch.py --num-tasks 100 --concurrency 10
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

EXEC_SERVER_SCRIPT = r'''
import http.server, json, subprocess, socketserver
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))
        cmd, timeout = body.get("cmd","echo noop"), body.get("timeout",30)
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd="/app")
            r = {"returncode":p.returncode,"stdout":p.stdout[-8000:],"stderr":p.stderr[-4000:]}
        except subprocess.TimeoutExpired:
            r = {"returncode":124,"stdout":"","stderr":"timeout"}
        except Exception as e:
            r = {"returncode":1,"stdout":"","stderr":str(e)}
        self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(json.dumps(r).encode())
    def log_message(self, *a): pass
with socketserver.TCPServer(("",9999),H) as s:
    print("exec:9999",flush=True); s.serve_forever()
'''

print_lock = Lock()

def log(msg):
    with print_lock:
        print(msg, flush=True)


def kubectl(*args, check=True):
    r = subprocess.run(["kubectl"] + list(args), capture_output=True, text=True)
    if check and r.returncode != 0:
        log(f"kubectl error: {r.stderr[:200]}")
    return r


def create_sidecar(task_idx, dockerhub_tag):
    """Create sidecar Job, port-forward, return (job_name, local_port, pf_proc)."""
    job_yaml = json.dumps({
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "generateName": f"swe-side-{task_idx:03d}-",
            "namespace": "eidf230ns",
            "labels": {
                "app": "swebench-sidecar",
                "task-index": str(task_idx),
                "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": {
                    "app": "swebench-sidecar",
                    "task-index": str(task_idx),
                    "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue",
                }},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "swebench",
                        "image": f"jefzda/sweap-images:{dockerhub_tag}",
                        "command": ["python3", "-c", EXEC_SERVER_SCRIPT],
                        "ports": [{"containerPort": 9999}],
                        "workingDir": "/app",
                        "resources": {
                            "requests": {"cpu": "1", "memory": "4Gi"},
                            "limits": {"cpu": "2", "memory": "8Gi"},
                        },
                    }],
                },
            },
        },
    })

    r = subprocess.run(
        ["kubectl", "create", "-f", "-", "-n", "eidf230ns", "-o", "jsonpath={.metadata.name}"],
        input=job_yaml, capture_output=True, text=True,
    )
    if r.returncode != 0:
        log(f"[task {task_idx}] failed to create sidecar: {r.stderr[:200]}")
        return None, None, None
    job_name = r.stdout.strip()
    log(f"[task {task_idx}] sidecar job: {job_name}")

    # Wait for pod running
    for _ in range(90):
        r = kubectl("get", "pods", "-n", "eidf230ns", f"-l=job-name={job_name}",
                     "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}", check=False)
        parts = r.stdout.strip().split("|")
        phase, pod_name = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
        if phase == "Running" and pod_name:
            break
        if phase in ("Failed", "Succeeded"):
            log(f"[task {task_idx}] sidecar {phase}")
            return job_name, None, None
        time.sleep(3)
    else:
        log(f"[task {task_idx}] sidecar timeout")
        return job_name, None, None

    # Port-forward to unique local port
    local_port = 19900 + task_idx
    pf_proc = subprocess.Popen(
        ["kubectl", "port-forward", pod_name, f"{local_port}:9999", "-n", "eidf230ns"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    # Wait for exec server
    for _ in range(30):
        try:
            req = urllib.request.Request(
                f"http://localhost:{local_port}/exec",
                data=json.dumps({"cmd": "echo ok", "timeout": 5}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            log(f"[task {task_idx}] sidecar ready localhost:{local_port}")
            return job_name, local_port, pf_proc
        except Exception:
            time.sleep(2)

    log(f"[task {task_idx}] exec server unreachable")
    pf_proc.kill()
    return job_name, None, None


def delete_sidecar(job_name, pf_proc=None):
    if pf_proc:
        pf_proc.kill()
        try:
            pf_proc.wait(timeout=5)
        except Exception:
            pass
    if job_name:
        kubectl("delete", "job", job_name, "-n", "eidf230ns",
                "--wait=false", "--ignore-not-found=true", check=False)


def run_one_task(task_idx, instance_id, dockerhub_tag, vllm_url, output_dir):
    """Run a single task end-to-end. Returns result dict."""
    log(f"[task {task_idx}] START {instance_id}")

    job_name, local_port, pf_proc = create_sidecar(task_idx, dockerhub_tag)
    if not local_port:
        delete_sidecar(job_name, pf_proc)
        return {"index": task_idx, "instance_id": instance_id, "status": "sidecar_failed"}

    try:
        env = os.environ.copy()
        env["SWEBENCH_EXEC_URL"] = f"http://localhost:{local_port}/exec"
        cmd = [
            sys.executable, "-m", "agent_cap.runner.unified_runner",
            "--model-name", "openai/gpt-oss-120b",
            "--dataset", "swe-bench-pro",
            "--backend", "swebench-k8s",
            "--serving-engine", "vllm",
            "--base-url", vllm_url,
            "--max-turns", "50",
            "--num-tasks", "1",
            "--task-offset", str(task_idx),
            "--output-dir", str(output_dir / f"task_{task_idx:03d}"),
        ]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        log(f"[task {task_idx}] DONE rc={r.returncode}")
        return {"index": task_idx, "instance_id": instance_id,
                "status": "ok" if r.returncode == 0 else f"exit_{r.returncode}"}
    except subprocess.TimeoutExpired:
        log(f"[task {task_idx}] TIMEOUT (30min)")
        return {"index": task_idx, "instance_id": instance_id, "status": "timeout"}
    except Exception as exc:
        log(f"[task {task_idx}] ERROR: {exc}")
        return {"index": task_idx, "instance_id": instance_id, "status": f"error: {exc}"}
    finally:
        delete_sidecar(job_name, pf_proc)


def wait_for_vllm(url, timeout=900):
    log(f"Waiting for vLLM at {url} ...")
    for i in range(timeout // 5):
        try:
            urllib.request.urlopen(f"{url}/models", timeout=5)
            log("vLLM ready!")
            return True
        except Exception:
            if i % 12 == 0:
                log(f"  still waiting ({i*5}s)...")
            time.sleep(5)
    log("vLLM failed!")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://localhost:30002/v1")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/swebench_100")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading SWE-bench Pro dataset...")
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")

    if not wait_for_vllm(args.vllm_url):
        sys.exit(1)

    task_range = range(args.task_offset, min(args.task_offset + args.num_tasks, len(ds)))
    log(f"Running {len(task_range)} tasks with concurrency={args.concurrency}")

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for i in task_range:
            ex = ds[i]
            f = pool.submit(
                run_one_task, i, ex["instance_id"], ex["dockerhub_tag"],
                args.vllm_url, output_dir,
            )
            futures[f] = i

        for f in as_completed(futures):
            result = f.result()
            results.append(result)
            done = len(results)
            log(f"[progress] {done}/{len(task_range)} done — {result['instance_id']}: {result['status']}")

    results.sort(key=lambda x: x["index"])
    with open(output_dir / "batch_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    ok = sum(1 for r in results if r["status"] == "ok")
    log(f"\nDone! {ok}/{len(results)} tasks succeeded. Summary: {output_dir}/batch_summary.json")


if __name__ == "__main__":
    main()
