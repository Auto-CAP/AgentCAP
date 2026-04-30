#!/usr/bin/env python3
"""Evaluate SWE-bench patches using swerex sidecars.

For each patch: create sidecar → apply before_repo_set_cmd → apply patch → run official tests.
"""
import argparse, json, subprocess, sys, time, ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datasets import load_dataset

print_lock = Lock()
def log(msg):
    with print_lock:
        print(msg, flush=True)

def kubectl(*args):
    return subprocess.run(["kubectl"] + list(args), capture_output=True, text=True)

def create_eval_sidecar(task_idx, dockerhub_tag):
    """Create sidecar with python:3.12 + repo + git + swerex."""
    job_yaml = json.dumps({
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"generateName": f"swe-eval-{task_idx:03d}-", "namespace": "eidf230ns",
            "labels": {"app": "sweagent-eval", "task-index": str(task_idx),
                        "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue"}},
        "spec": {"backoffLimit": 0, "template": {
            "metadata": {"labels": {"app": "sweagent-eval", "task-index": str(task_idx),
                                     "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue"}},
            "spec": {
                "restartPolicy": "Never",
                "volumes": [{"name": "repo-data", "emptyDir": {}}],
                "initContainers": [{
                    "name": "copy-repo",
                    "image": f"jefzda/sweap-images:{dockerhub_tag}",
                    "command": ["bash", "-c", "cp -a /app/. /repo/"],
                    "volumeMounts": [{"name": "repo-data", "mountPath": "/repo"}],
                    "resources": {"requests": {"cpu": "500m", "memory": "2Gi"},
                                  "limits": {"cpu": "1", "memory": "4Gi"}},
                }],
                "containers": [{
                    "name": "eval",
                    "image": "python:3.12-slim",
                    "command": ["bash", "-c"],
                    "args": [
                        "apt-get update -qq && apt-get install -y -qq git curl > /dev/null 2>&1 && "
                        "git config --global --add safe.directory '*' && "
                        "pip install 'swe-rex>=1.4.0' 2>&1 | tail -1 && "
                        "python3 -m swerex --port 9999 --auth-token token123"
                    ],
                    "ports": [{"containerPort": 9999}],
                    "workingDir": "/app",
                    "volumeMounts": [{"name": "repo-data", "mountPath": "/app"}],
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
        phase, pod_ip = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
        if phase == "Running" and pod_ip:
            break
        if phase in ("Failed", "Succeeded"):
            return job_name, None
        time.sleep(3)
    else:
        return job_name, None

    # Wait for swerex
    import urllib.request
    for _ in range(120):
        try:
            req = urllib.request.Request(f"http://{pod_ip}:9999/is_alive", headers={"X-API-Key": "token123"})
            urllib.request.urlopen(req, timeout=5)
            return job_name, pod_ip
        except:
            time.sleep(3)
    return job_name, None

def delete_sidecar(job_name):
    if job_name:
        kubectl("delete", "job", job_name, "-n", "eidf230ns", "--wait=false", "--ignore-not-found=true")

def eval_one(task_idx, instance_id, dockerhub_tag, patch, before_cmd, selected_tests, fail_to_pass):
    """Eval one patch."""
    log(f"[eval {task_idx}] START {instance_id[:50]}")
    job_name, pod_ip = create_eval_sidecar(task_idx, dockerhub_tag)
    if not pod_ip:
        delete_sidecar(job_name)
        log(f"[eval {task_idx}] sidecar failed")
        return {"instance_id": instance_id, "passed": False, "reason": "sidecar_failed"}

    try:
        import asyncio
        from swerex.deployment.remote import RemoteDeployment
        from swerex.runtime.abstract import Command, WriteFileRequest

        async def run_eval():
            d = RemoteDeployment(host=f"http://{pod_ip}", port=9999, auth_token="token123")
            await d.start()
            r = d.runtime

            # 1. Apply before_repo_set_cmd
            for cmd in before_cmd.strip().split("\n"):
                cmd = cmd.strip()
                if cmd:
                    await r.execute(Command(command=f"cd /app && {cmd}", shell=True, timeout=30))

            # 2. Apply patch
            await r.write_file(WriteFileRequest(path="/tmp/model.patch", content=patch))
            result = await r.execute(Command(command="cd /app && git apply -v /tmp/model.patch", shell=True, timeout=30))
            if result.exit_code != 0:
                log(f"[eval {task_idx}] patch apply failed: {result.stderr[:100]}")
                return {"instance_id": instance_id, "passed": False, "reason": "patch_apply_failed"}

            # 3. Download + run official tests
            script_url = f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/run_scripts/{instance_id}/run_script.sh"
            parser_url = f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/run_scripts/{instance_id}/parser.py"
            await r.execute(Command(command=f"curl -sL '{script_url}' -o /tmp/run_script.sh && chmod +x /tmp/run_script.sh", shell=True, timeout=30))
            await r.execute(Command(command=f"curl -sL '{parser_url}' -o /tmp/parser.py", shell=True, timeout=30))

            try:
                test_files = json.loads(selected_tests) if selected_tests else []
            except:
                test_files = []
            tf_str = " ".join(test_files)

            await r.execute(Command(
                command=f"cd /app && bash /tmp/run_script.sh {tf_str} > /tmp/stdout.log 2> /tmp/stderr.log",
                shell=True, timeout=300))
            await r.execute(Command(
                command="cd /app && python3 /tmp/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json",
                shell=True, timeout=30))

            result = await r.execute(Command(command="cat /tmp/output.json", shell=True, timeout=10))
            output = result.stdout.strip()

            try:
                tests_json = json.loads(output) if output else {}
                test_results = tests_json.get("tests", [])
                try:
                    f2p = json.loads(fail_to_pass) if isinstance(fail_to_pass, str) else fail_to_pass
                except:
                    f2p = ast.literal_eval(fail_to_pass) if fail_to_pass else []

                passed_names = {t["name"] for t in test_results if t.get("status") == "PASSED"}
                ok = all(any(fp in pn for pn in passed_names) for fp in f2p) if f2p else False
                log(f"[eval {task_idx}] {'PASS' if ok else 'FAIL'} ({len(passed_names)} passed tests)")
                return {"instance_id": instance_id, "passed": ok}
            except:
                log(f"[eval {task_idx}] parse error")
                return {"instance_id": instance_id, "passed": False, "reason": "parse_error"}

        return asyncio.run(run_eval())
    except Exception as e:
        log(f"[eval {task_idx}] ERROR: {e}")
        return {"instance_id": instance_id, "passed": False, "reason": str(e)}
    finally:
        delete_sidecar(job_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    patches = json.load(open(args.patches))
    log(f"Evaluating {len(patches)} patches")

    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    ds_map = {ex["instance_id"]: ex for ex in ds}

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for i, p in enumerate(patches):
            ex = ds_map.get(p["instance_id"])
            if not ex:
                continue
            f = pool.submit(eval_one, i, p["instance_id"], ex["dockerhub_tag"], p["patch"],
                           ex.get("before_repo_set_cmd", ""), ex.get("selected_test_files_to_run", ""),
                           ex.get("fail_to_pass", ""))
            futures[f] = i

        for f in as_completed(futures):
            result = f.result()
            results.append(result)
            passed = sum(1 for r in results if r.get("passed"))
            log(f"[eval progress] {len(results)}/{len(patches)} done, {passed} passed")

    passed = sum(1 for r in results if r.get("passed"))
    log(f"\nDone! {passed}/{len(patches)} passed = {passed/len(patches)*100:.1f}% (of patches)")
    log(f"Overall: {passed}/100 = {passed}%")

if __name__ == "__main__":
    main()
