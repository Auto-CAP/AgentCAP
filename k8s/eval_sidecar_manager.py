#!/usr/bin/env python3
"""Eval sidecar manager - creates sidecars for patch evaluation."""
import json, subprocess, time, base64, sys
from datasets import load_dataset

def kubectl(*args):
    return subprocess.run(["kubectl"] + list(args), capture_output=True, text=True)

def write_to_pvc(line):
    pods = kubectl("get", "pods", "-n", "eidf230ns", "-l", "app=vllm-gptoss-h200",
                   "--field-selector=status.phase=Running", "-o", "jsonpath={.items[0].metadata.name}")
    pod = pods.stdout.strip()
    if pod:
        b64 = base64.b64encode(line.encode()).decode()
        kubectl("exec", pod, "-n", "eidf230ns", "--", "bash", "-c",
                f"echo '{b64}' | base64 -d >> /workspace/eval_queue.jsonl")

def create_eval_sidecar(i, dockerhub_tag):
    job_yaml = json.dumps({
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"generateName": f"swe-eval-{i:03d}-", "namespace": "eidf230ns",
            "labels": {"app": "sweagent-eval-side", "task-index": str(i),
                        "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue"}},
        "spec": {"backoffLimit": 0, "template": {
            "metadata": {"labels": {"app": "sweagent-eval-side", "task-index": str(i),
                                     "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue"}},
            "spec": {
                "restartPolicy": "Never",
                "containers": [{
                    "name": "eval",
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
    r = subprocess.run(["kubectl", "create", "-f", "-", "-n", "eidf230ns", "-o", "jsonpath={.metadata.name}"],
                       input=job_yaml, capture_output=True, text=True)
    if r.returncode != 0:
        return None, None
    job_name = r.stdout.strip()

    for _ in range(120):
        r = kubectl("get", "pods", "-n", "eidf230ns", f"-l=job-name={job_name}",
                     "-o", "jsonpath={.items[0].status.phase}|{.items[0].status.podIP}")
        parts = r.stdout.strip().split("|")
        phase = parts[0] if parts else ""
        ip = parts[1] if len(parts) > 1 else ""
        if phase == "Running" and ip:
            return job_name, ip
        if phase in ("Failed", "Succeeded"):
            return job_name, None
        time.sleep(3)
    return job_name, None

def main():
    patches = json.load(open(sys.argv[1]))
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    ds_map = {ex["instance_id"]: ex for ex in ds}

    print(f"Managing eval for {len(patches)} patches", flush=True)
    active = {}
    queue = list(range(len(patches)))
    completed = set()

    while len(completed) < len(patches):
        while len(active) < 10 and queue:
            i = queue.pop(0)
            p = patches[i]
            ex = ds_map.get(p["instance_id"], {})
            tag = ex.get("dockerhub_tag", "")
            if not tag:
                completed.add(i)
                print(f"[eval-mgr] {i} no tag, skip", flush=True)
                continue

            job_name, pod_ip = create_eval_sidecar(i, tag)
            if pod_ip:
                active[i] = job_name
                line = json.dumps({"idx": i, "instance_id": p["instance_id"], "pod_ip": pod_ip}) + "\n"
                write_to_pvc(line)
                print(f"[eval-mgr] sidecar {i} at {pod_ip} image=...{tag[-30:]}", flush=True)
            else:
                completed.add(i)
                if job_name:
                    kubectl("delete", "job", job_name, "-n", "eidf230ns", "--wait=false", "--ignore-not-found=true")
                print(f"[eval-mgr] sidecar {i} failed", flush=True)

        time.sleep(30)
        for idx in list(active.keys()):
            r = kubectl("get", "pods", "-n", "eidf230ns", f"-l=job-name={active[idx]}",
                         "-o", "jsonpath={.items[0].status.phase}")
            if r.stdout.strip() in ("Failed", "Succeeded", ""):
                completed.add(idx)
                kubectl("delete", "job", active.pop(idx), "-n", "eidf230ns", "--wait=false", "--ignore-not-found=true")
                print(f"[eval-mgr] sidecar {idx} done, {len(completed)}/{len(patches)}", flush=True)

    print("All eval sidecars managed", flush=True)

if __name__ == "__main__":
    main()
