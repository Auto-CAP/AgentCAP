#!/usr/bin/env python3
"""Sidecar lifecycle manager - runs on login node.

Creates sidecar pods for SWE-bench tasks, maintains N concurrent,
writes task_queue.jsonl for the K8s orchestrator to consume.

Usage:
    python -u sidecar_manager.py --num-tasks 100 --concurrency 10
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_dataset


def kubectl(*args):
    return subprocess.run(["kubectl"] + list(args), capture_output=True, text=True)


def create_sidecar(task_idx, dockerhub_tag):
    """Create sidecar job. Returns (job_name, pod_ip) once ready."""
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
        print(f"[sidecar {task_idx}] create failed: {r.stderr[:200]}", flush=True)
        return None, None
    job_name = r.stdout.strip()

    # Wait for pod running + get IP
    for _ in range(120):
        r = kubectl("get", "pods", "-n", "eidf230ns", f"-l=job-name={job_name}",
                     "-o", "jsonpath={.items[0].status.phase}|{.items[0].status.podIP}")
        parts = r.stdout.strip().split("|")
        phase = parts[0] if parts else ""
        pod_ip = parts[1] if len(parts) > 1 else ""
        if phase == "Running" and pod_ip:
            print(f"[sidecar {task_idx}] running at {pod_ip}", flush=True)
            return job_name, pod_ip
        if phase in ("Failed", "Succeeded"):
            print(f"[sidecar {task_idx}] {phase}", flush=True)
            return job_name, None
        time.sleep(3)

    print(f"[sidecar {task_idx}] timeout waiting for pod", flush=True)
    return job_name, None


def delete_sidecar(job_name):
    if job_name:
        kubectl("delete", "job", job_name, "-n", "eidf230ns",
                "--wait=false", "--ignore-not-found=true")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--task-queue-file", default="/tmp/task_queue.jsonl",
                        help="Output JSONL for orchestrator")
    parser.add_argument("--results-dir", default="results/sweagent_h200",
                        help="Watch for DONE markers here")
    args = parser.parse_args()

    print("Loading SWE-bench Pro dataset...", flush=True)
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")

    task_range = list(range(args.task_offset, min(args.task_offset + args.num_tasks, len(ds))))
    results_dir = Path(args.results_dir)

    # Track active sidecars: {task_idx: job_name}
    active = {}
    queue = list(task_range)  # tasks waiting to be scheduled
    completed = set()

    # Write tasks to PVC via kubectl exec (orchestrator reads from PVC)
    pvc_queue_file = "/workspace/task_queue.jsonl"
    # Find a pod with PVC mounted for writing
    def write_to_pvc(line):
        """Append a line to task_queue.jsonl on PVC."""
        import base64
        b64 = base64.b64encode(line.encode()).decode()
        # Use any pod with the PVC - vLLM or orchestrator
        pods = kubectl("get", "pods", "-n", "eidf230ns",
                       "-l", "app in (vllm-gptoss-h200, sweagent-orchestrator)",
                       "--field-selector=status.phase=Running",
                       "-o", "jsonpath={.items[0].metadata.name}")
        pod = pods.stdout.strip()
        if pod:
            kubectl("exec", pod, "-n", "eidf230ns", "--",
                    "bash", "-c", f"echo '{b64}' | base64 -d >> {pvc_queue_file}")

    print(f"Managing {len(task_range)} tasks, concurrency={args.concurrency}", flush=True)

    while len(completed) < len(task_range):
        # Fill up to concurrency
        while len(active) < args.concurrency and queue:
            task_idx = queue.pop(0)
            ex = ds[task_idx]
            job_name, pod_ip = create_sidecar(task_idx, ex["dockerhub_tag"])

            if pod_ip:
                active[task_idx] = job_name
                # Write to task queue on PVC for orchestrator
                line = json.dumps({
                    "idx": task_idx,
                    "instance_id": ex["instance_id"],
                    "pod_ip": pod_ip,
                    "problem_statement": ex["problem_statement"],
                }) + "\n"
                write_to_pvc(line)
            else:
                # Sidecar failed to start, mark as done
                completed.add(task_idx)
                delete_sidecar(job_name)
                print(f"[manager] task {task_idx} sidecar failed, skipping", flush=True)

        # Check for completed tasks (DONE markers on PVC)
        for task_idx in list(active.keys()):
            r = kubectl("exec", "-n", "eidf230ns",
                        kubectl("get", "pods", "-n", "eidf230ns",
                                "-l", "app in (vllm-gptoss-h200, sweagent-orchestrator)",
                                "--field-selector=status.phase=Running",
                                "-o", "jsonpath={.items[0].metadata.name}").stdout.strip(),
                        "--", "test", "-f", f"/workspace/{results_dir}/task_{task_idx:03d}/DONE")
            if r.returncode == 0:
                completed.add(task_idx)
                delete_sidecar(active.pop(task_idx))
                print(f"[manager] task {task_idx} completed, {len(completed)}/{len(task_range)} done, "
                      f"{len(active)} active", flush=True)

        if active:
            time.sleep(10)

    qf.close()
    print(f"\nAll {len(task_range)} tasks processed.", flush=True)


if __name__ == "__main__":
    main()
