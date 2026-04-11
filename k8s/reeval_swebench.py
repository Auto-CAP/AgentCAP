#!/usr/bin/env python3
"""Re-evaluate SWE-bench results using official run_script.sh + parser.py.

For each task that has a patch.diff, this script:
1. Creates a sidecar pod with the correct Docker image
2. Applies the test_patch (from dataset)
3. Sets up git baseline
4. Applies the model's patch
5. Downloads and runs the official run_script.sh + parser.py
6. Saves updated test_result.json

Usage:
    python -u reeval_swebench.py --results-dir results/swebench_100 --concurrency 10
"""
import argparse
import glob
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


def kubectl(*args, check=True):
    r = subprocess.run(["kubectl"] + list(args), capture_output=True, text=True)
    return r


def create_sidecar(task_idx, dockerhub_tag):
    """Create sidecar Job, port-forward, return (job_name, local_port, pf_proc)."""
    EXEC_SCRIPT = r'''
import http.server, json, subprocess, socketserver
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))
        cmd, timeout = body.get("cmd","echo noop"), body.get("timeout",30)
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd="/app")
            r = {"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
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
    job_yaml = json.dumps({
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "generateName": f"swe-eval-{task_idx:03d}-",
            "namespace": "eidf230ns",
            "labels": {"app": "swebench-eval", "task-index": str(task_idx),
                        "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue"},
        },
        "spec": {"backoffLimit": 0, "template": {
            "metadata": {"labels": {"app": "swebench-eval", "task-index": str(task_idx),
                                     "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue"}},
            "spec": {"restartPolicy": "Never", "containers": [{
                "name": "swebench", "image": f"jefzda/sweap-images:{dockerhub_tag}",
                "command": ["python3", "-c", EXEC_SCRIPT],
                "ports": [{"containerPort": 9999}], "workingDir": "/app",
                "resources": {"requests": {"cpu": "1", "memory": "4Gi"},
                              "limits": {"cpu": "2", "memory": "8Gi"}},
            }]},
        }},
    })

    r = subprocess.run(
        ["kubectl", "create", "-f", "-", "-n", "eidf230ns", "-o", "jsonpath={.metadata.name}"],
        input=job_yaml, capture_output=True, text=True,
    )
    if r.returncode != 0:
        log(f"[eval {task_idx}] sidecar create failed: {r.stderr[:200]}")
        return None, None, None
    job_name = r.stdout.strip()

    for _ in range(90):
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

    local_port = 18900 + task_idx
    pf_proc = subprocess.Popen(
        ["kubectl", "port-forward", pod_name, f"{local_port}:9999", "-n", "eidf230ns"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    for _ in range(20):
        try:
            req = urllib.request.Request(
                f"http://localhost:{local_port}/exec",
                data=json.dumps({"cmd": "echo ok", "timeout": 5}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            return job_name, local_port, pf_proc
        except Exception:
            time.sleep(2)

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


def write_file_to_sidecar(port, path, content):
    """Write file to sidecar by splitting into chunks to avoid shell limits."""
    import base64
    b64 = base64.b64encode(content.encode()).decode()
    # Write in chunks of 32KB to avoid any shell/arg limits
    chunk_size = 32000
    for i in range(0, len(b64), chunk_size):
        chunk = b64[i:i+chunk_size]
        mode = ">" if i == 0 else ">>"
        r = exec_in_sidecar(port, f"printf '%s' '{chunk}' {mode} /tmp/_b64_tmp", 15)
        if r.get("returncode", 1) != 0:
            return r
    r = exec_in_sidecar(port, f"base64 -d /tmp/_b64_tmp > {path} && rm /tmp/_b64_tmp", 15)
    return r


def exec_in_sidecar(port, cmd, timeout=30):
    req = urllib.request.Request(
        f"http://localhost:{port}/exec",
        data=json.dumps({"cmd": f"cd /app && {cmd}", "timeout": timeout}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout + 10)
        return json.loads(resp.read())
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": str(e)}


def eval_one_task(task_idx, instance_id, dockerhub_tag, test_patch, fail_to_pass, patch_text):
    """Apply patch and run official tests. Returns test result dict."""
    log(f"[eval {task_idx}] START {instance_id[:60]}")

    job_name, port, pf_proc = create_sidecar(task_idx, dockerhub_tag)
    if not port:
        log(f"[eval {task_idx}] sidecar failed")
        delete_sidecar(job_name, pf_proc)
        return {"passed": False, "reason": "sidecar_failed"}

    try:
        # Apply test_patch
        if test_patch:
            write_file_to_sidecar(port, "/tmp/test.patch", test_patch)
            exec_in_sidecar(port, "git apply /tmp/test.patch", 30)

        # Git baseline
        exec_in_sidecar(port,
            "git init 2>/dev/null; git add -A 2>/dev/null; "
            "git -c user.email=bench@test -c user.name=bench commit -m baseline --allow-empty 2>/dev/null", 30)

        # Apply model's patch — write via HTTP to avoid shell quoting issues
        write_file_to_sidecar(port, "/tmp/model.patch", patch_text)
        r = exec_in_sidecar(port, "git apply /tmp/model.patch", 30)
        if r["returncode"] != 0:
            log(f"[eval {task_idx}] patch apply failed: {r['stderr'][:200]}")
            return {"passed": False, "reason": "patch_apply_failed", "stderr": r["stderr"][:500]}

        # Parse fail_to_pass
        try:
            tests = json.loads(fail_to_pass) if isinstance(fail_to_pass, str) else fail_to_pass
        except json.JSONDecodeError:
            tests = [fail_to_pass]

        # Download official run_script.sh + parser.py
        script_url = (
            f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/"
            f"run_scripts/{instance_id}/run_script.sh"
        )
        parser_url = (
            f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/"
            f"run_scripts/{instance_id}/parser.py"
        )
        exec_in_sidecar(port, f"curl -sL '{script_url}' -o /run_script.sh && chmod +x /run_script.sh", 30)
        exec_in_sidecar(port, f"curl -sL '{parser_url}' -o /parser.py", 30)

        # Run tests
        test_files_str = ",".join(t.split(" | ")[0].strip() for t in tests)
        r = exec_in_sidecar(port, f"bash /run_script.sh {test_files_str} 2>&1", 300)
        output = (r.get("stdout", "") + r.get("stderr", ""))[-2000:]

        # Parse results
        r2 = exec_in_sidecar(port,
            f"python3 /parser.py --log '{output[-8000:]}' --expected '{json.dumps(tests)}' 2>/dev/null", 30)
        parse_output = r2.get("stdout", "")
        ok = "PASS" in parse_output.upper() if parse_output else False

        log(f"[eval {task_idx}] {'PASS' if ok else 'FAIL'}")
        return {"passed": ok, "total": len(tests), "details": output[-500:]}

    except Exception as e:
        log(f"[eval {task_idx}] ERROR: {e}")
        return {"passed": False, "reason": str(e)}
    finally:
        delete_sidecar(job_name, pf_proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    log("Loading dataset for metadata...")
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    ds_map = {ex["instance_id"]: ex for ex in ds}

    # Find tasks with patches
    tasks_to_eval = []
    for d in sorted(results_dir.glob("task_*")):
        idx = int(d.name.replace("task_", ""))
        patches = list(d.rglob("patch.diff"))
        if not patches or os.path.getsize(patches[0]) == 0:
            continue
        patch_text = open(patches[0]).read()

        # Skip invalid patches (not starting with diff header)
        if not patch_text.strip().startswith(("diff --git", "---")):
            log(f"[task {idx}] skipping invalid patch")
            continue

        # Get instance_id from output_data
        instance_id = None
        for f in d.rglob("output_data_*.jsonl"):
            for line in open(f):
                if line.strip():
                    instance_id = json.loads(line).get("task_id", "")
                    break
            if instance_id:
                break
        if not instance_id or instance_id not in ds_map:
            log(f"[task {idx}] unknown instance_id: {instance_id}")
            continue

        ex = ds_map[instance_id]
        tasks_to_eval.append({
            "idx": idx, "instance_id": instance_id,
            "dockerhub_tag": ex["dockerhub_tag"],
            "test_patch": ex.get("test_patch", ""),
            "fail_to_pass": ex.get("fail_to_pass", ""),
            "patch_text": patch_text,
            "patch_path": patches[0].parent,
        })

    log(f"Found {len(tasks_to_eval)} tasks with patches to re-evaluate")

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for t in tasks_to_eval:
            f = pool.submit(
                eval_one_task, t["idx"], t["instance_id"], t["dockerhub_tag"],
                t["test_patch"], t["fail_to_pass"], t["patch_text"],
            )
            futures[f] = t

        for f in as_completed(futures):
            t = futures[f]
            result = f.result()
            results.append(result)
            # Save test_result.json
            out_path = t["patch_path"] / "test_result.json"
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2)
            done = len(results)
            passed = sum(1 for r in results if r.get("passed"))
            log(f"[progress] {done}/{len(tasks_to_eval)} — {t['instance_id'][:50]}: "
                f"{'PASS' if result.get('passed') else 'FAIL'} (running acc: {passed}/{done})")

    passed = sum(1 for r in results if r.get("passed"))
    log(f"\nDone! {passed}/{len(tasks_to_eval)} passed (of {len(tasks_to_eval)} with patches)")


if __name__ == "__main__":
    main()
