#!/usr/bin/env python3
"""Evaluate SWE-bench Pro patches on K8s - mirrors swe_bench_pro_eval.py exactly.

For each patch:
  1. Create K8s Job with correct swebench Docker image
  2. kubectl cp patch + run_script.sh + parser.py into pod
  3. kubectl exec the official entry script
  4. kubectl cp output.json back
  5. Parse result

Usage:
    python -u eval_k8s.py --patches patches.json --concurrency 10
"""
import argparse, json, os, subprocess, sys, time, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from datasets import load_dataset

print_lock = Lock()
def log(msg):
    with print_lock:
        print(msg, flush=True)

def kubectl(*args):
    return subprocess.run(["kubectl"] + list(args), capture_output=True, text=True, timeout=600)


def eval_one(idx, instance_id, dockerhub_tag, patch, before_repo_set_cmd,
             selected_test_files, base_commit, scripts_dir):
    """Evaluate one patch - mirrors swe_bench_pro_eval.py exactly."""
    log(f"[eval {idx}] START {instance_id[:50]}")

    # 1. Create pod
    job_yaml = json.dumps({
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"generateName": f"swe-eval-{idx:03d}-", "namespace": "eidf230ns",
            "labels": {"app": "swebench-eval-k8s", "task-index": str(idx),
                        "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue"}},
        "spec": {"backoffLimit": 0, "template": {
            "metadata": {"labels": {"app": "swebench-eval-k8s", "task-index": str(idx),
                                     "kueue.x-k8s.io/queue-name": "eidf230ns-user-queue"}},
            "spec": {
                "restartPolicy": "Never",
                "containers": [{
                    "name": "eval",
                    "image": f"jefzda/sweap-images:{dockerhub_tag}",
                    "command": ["bash", "-c", "sleep 3600"],
                    "workingDir": "/app",
                    "resources": {"requests": {"cpu": "1", "memory": "4Gi"},
                                  "limits": {"cpu": "2", "memory": "8Gi"}},
                }],
            },
        }},
    })
    r = subprocess.run(["kubectl", "create", "-f", "-", "-n", "eidf230ns", "-o", "jsonpath={.metadata.name}"],
                       input=job_yaml, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"[eval {idx}] job create failed: {r.stderr[:100]}")
        return {"instance_id": instance_id, "passed": False, "reason": "job_create_failed"}
    job_name = r.stdout.strip()

    try:
        # 2. Wait for pod running
        pod_name = ""
        for _ in range(120):
            r = kubectl("get", "pods", "-n", "eidf230ns", f"-l=job-name={job_name}",
                         "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}")
            parts = r.stdout.strip().split("|")
            phase = parts[0] if parts else ""
            pod_name = parts[1] if len(parts) > 1 else ""
            if phase == "Running" and pod_name:
                break
            if phase in ("Failed", "Succeeded"):
                return {"instance_id": instance_id, "passed": False, "reason": "pod_failed"}
            time.sleep(3)
        else:
            return {"instance_id": instance_id, "passed": False, "reason": "pod_timeout"}

        # 3. Copy files into pod
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write patch
            patch_path = os.path.join(tmpdir, "patch.diff")
            with open(patch_path, "w") as f:
                f.write(patch)

            # Copy run_script.sh and parser.py from official repo
            run_script = os.path.join(scripts_dir, instance_id, "run_script.sh")
            parser_py = os.path.join(scripts_dir, instance_id, "parser.py")

            if not os.path.exists(run_script) or not os.path.exists(parser_py):
                log(f"[eval {idx}] scripts not found for {instance_id}")
                return {"instance_id": instance_id, "passed": False, "reason": "scripts_missing"}

            kubectl("exec", pod_name, "-n", "eidf230ns", "--", "mkdir", "-p", "/workspace")
            kubectl("cp", patch_path, f"eidf230ns/{pod_name}:/workspace/patch.diff")
            kubectl("cp", run_script, f"eidf230ns/{pod_name}:/workspace/run_script.sh")
            kubectl("cp", parser_py, f"eidf230ns/{pod_name}:/workspace/parser.py")
            kubectl("exec", pod_name, "-n", "eidf230ns", "--", "chmod", "+x", "/workspace/run_script.sh")

        # 4. Run official entry script (exact copy from swe_bench_pro_eval.py)
        entry_script = f"""
cd /app
git config --global --add safe.directory '*'
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
bash /workspace/run_script.sh {selected_test_files} > /workspace/stdout.log 2> /workspace/stderr.log
python3 /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
        r = kubectl("exec", pod_name, "-n", "eidf230ns", "--", "bash", "-c", entry_script)
        log(f"[eval {idx}] entry script rc={r.returncode}")

        # 5. Get output.json
        r = kubectl("exec", pod_name, "-n", "eidf230ns", "--", "cat", "/workspace/output.json")
        output = r.stdout.strip()

        if not output:
            log(f"[eval {idx}] no output.json")
            return {"instance_id": instance_id, "passed": False, "reason": "no_output"}

        # 6. Parse result (exact copy from swe_bench_pro_eval.py line 555-558)
        try:
            output_json = json.loads(output)
            passed_tests = {x["name"] for x in output_json.get("tests", []) if x["status"] == "PASSED"}

            import ast
            raw_f2p = ds_map[instance_id].get("fail_to_pass", "[]") if instance_id in ds_map else "[]"
            raw_p2p = ds_map[instance_id].get("pass_to_pass", "[]") if instance_id in ds_map else "[]"
            try:
                f2p = set(json.loads(raw_f2p)) if isinstance(raw_f2p, str) else set(raw_f2p)
            except json.JSONDecodeError:
                f2p = set(ast.literal_eval(raw_f2p))
            try:
                p2p = set(json.loads(raw_p2p)) if isinstance(raw_p2p, str) else set(raw_p2p)
            except json.JSONDecodeError:
                p2p = set(ast.literal_eval(raw_p2p))

            resolved = (f2p | p2p) <= passed_tests
            log(f"[eval {idx}] {'PASS' if resolved else 'FAIL'} "
                f"(f2p={len(f2p)}, p2p={len(p2p)}, passed={len(passed_tests)})")
            return {"instance_id": instance_id, "passed": resolved,
                    "f2p_count": len(f2p), "p2p_count": len(p2p), "passed_count": len(passed_tests)}
        except Exception as e:
            log(f"[eval {idx}] parse error: {e}")
            return {"instance_id": instance_id, "passed": False, "reason": f"parse: {e}"}

    except Exception as e:
        log(f"[eval {idx}] ERROR: {e}")
        return {"instance_id": instance_id, "passed": False, "reason": str(e)}
    finally:
        kubectl("delete", "job", job_name, "-n", "eidf230ns", "--wait=false", "--ignore-not-found=true")


# Global dataset map
ds_map = {}

def main():
    global ds_map
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", required=True)
    parser.add_argument("--scripts-dir", default="/tmp/swebench_pro_os/run_scripts")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    patches = json.load(open(args.patches))
    log(f"Evaluating {len(patches)} patches")

    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    ds_map.update({ex["instance_id"]: ex for ex in ds})

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for i, p in enumerate(patches):
            ex = ds_map.get(p["instance_id"], {})
            if not ex:
                continue

            try:
                selected = json.loads(ex.get("selected_test_files_to_run", "[]"))
                selected_str = " ".join(selected)
            except:
                selected_str = ""

            f = pool.submit(eval_one, i, p["instance_id"], ex["dockerhub_tag"], p["patch"],
                           ex.get("before_repo_set_cmd", ""), selected_str,
                           ex.get("base_commit", "HEAD"), args.scripts_dir)
            futures[f] = i

        for f in as_completed(futures):
            result = f.result()
            results.append(result)
            passed = sum(1 for r in results if r.get("passed"))
            log(f"[progress] {len(results)}/{len(patches)} done, {passed} passed")

    passed = sum(1 for r in results if r.get("passed"))
    log(f"\nDone! {passed}/{len(patches)} passed ({passed/max(len(patches),1)*100:.1f}%)")
    log(f"Overall acc: {passed}/100 = {passed}%")

    with open(args.output, "w") as f:
        json.dump({"passed": passed, "total": len(patches), "total_tasks": 100,
                   "acc": passed / 100, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
