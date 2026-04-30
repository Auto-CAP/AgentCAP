#!/usr/bin/env python3
"""Evaluate patches from K8s pod. Reads eval_queue.jsonl for sidecar pod IPs."""
import argparse, json, ast, sys, time, asyncio, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from datasets import load_dataset
from swerex.deployment.remote import RemoteDeployment
from swerex.runtime.abstract import Command, WriteFileRequest

print_lock = Lock()
def log(msg):
    with print_lock:
        print(msg, flush=True)


def wait_for_swerex(pod_ip, timeout=300):
    for _ in range(timeout // 3):
        try:
            req = urllib.request.Request(f"http://{pod_ip}:9999/is_alive", headers={"X-API-Key": "token123"})
            urllib.request.urlopen(req, timeout=5)
            return True
        except:
            time.sleep(3)
    return False


def eval_one(idx, instance_id, pod_ip, patch, before_cmd, selected_tests, fail_to_pass):
    log(f"[eval {idx}] START {instance_id[:50]}")

    if not wait_for_swerex(pod_ip):
        log(f"[eval {idx}] swerex not ready at {pod_ip}")
        return {"instance_id": instance_id, "passed": False, "reason": "swerex_failed"}

    async def run():
        d = RemoteDeployment(host=f"http://{pod_ip}", port=9999, auth_token="token123")
        await d.start()
        r = d.runtime

        # 1. before_repo_set_cmd
        for cmd in before_cmd.strip().split("\n"):
            cmd = cmd.strip()
            if cmd:
                await r.execute(Command(command=f"cd /app && {cmd}", shell=True, timeout=60))

        # 2. Apply patch
        await r.write_file(WriteFileRequest(path="/tmp/model.patch", content=patch))
        result = await r.execute(Command(command="cd /app && git apply -v /tmp/model.patch", shell=True, timeout=30))
        if result.exit_code != 0:
            log(f"[eval {idx}] patch apply failed: {result.stderr[:100]}")
            return {"instance_id": instance_id, "passed": False, "reason": "patch_apply_failed"}

        # 3. Download + run official tests
        base = f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/run_scripts/{instance_id}"
        await r.execute(Command(command=f"curl -sL '{base}/run_script.sh' -o /tmp/run_script.sh && chmod +x /tmp/run_script.sh", shell=True, timeout=30))
        await r.execute(Command(command=f"curl -sL '{base}/parser.py' -o /tmp/parser.py", shell=True, timeout=30))

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
        try:
            tests_json = json.loads(result.stdout.strip()) if result.stdout.strip() else {}
            test_results = tests_json.get("tests", [])
            try:
                f2p = json.loads(fail_to_pass) if isinstance(fail_to_pass, str) else fail_to_pass
            except:
                f2p = ast.literal_eval(fail_to_pass) if fail_to_pass else []
            passed_names = {t["name"] for t in test_results if t.get("status") == "PASSED"}
            ok = all(any(fp in pn for pn in passed_names) for fp in f2p) if f2p else False
            log(f"[eval {idx}] {'PASS' if ok else 'FAIL'} ({len(passed_names)} passed)")
            return {"instance_id": instance_id, "passed": ok}
        except:
            log(f"[eval {idx}] parse error")
            return {"instance_id": instance_id, "passed": False, "reason": "parse_error"}

    try:
        return asyncio.run(run())
    except Exception as e:
        log(f"[eval {idx}] ERROR: {e}")
        return {"instance_id": instance_id, "passed": False, "reason": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", required=True)
    parser.add_argument("--eval-queue", required=True, help="JSONL with {idx, instance_id, pod_ip}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    patches = json.load(open(args.patches))
    log(f"Loaded {len(patches)} patches")

    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    ds_map = {ex["instance_id"]: ex for ex in ds}

    # Wait for eval_queue
    log("Waiting for eval queue...")
    total_expected = len(patches)
    results = []
    seen = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        idle = 0
        while len(results) < total_expected:
            # Read new entries
            new = []
            try:
                with open(args.eval_queue) as f:
                    for i, line in enumerate(f):
                        if i < seen:
                            continue
                        if line.strip():
                            new.append(json.loads(line))
                            seen = i + 1
            except:
                pass

            for entry in new:
                idx = entry["idx"]
                iid = entry["instance_id"]
                pod_ip = entry["pod_ip"]
                ex = ds_map.get(iid, {})
                # Find matching patch
                patch = ""
                for p in patches:
                    if p["instance_id"] == iid:
                        patch = p["patch"]
                        break
                if not patch:
                    continue

                f = pool.submit(eval_one, idx, iid, pod_ip, patch,
                               ex.get("before_repo_set_cmd", ""),
                               ex.get("selected_test_files_to_run", ""),
                               ex.get("fail_to_pass", ""))
                futures[f] = idx
                idle = 0

            done_futs = [f for f in futures if f.done()]
            for f in done_futs:
                result = f.result()
                results.append(result)
                del futures[f]
                passed = sum(1 for r in results if r.get("passed"))
                log(f"[progress] {len(results)}/{total_expected} done, {passed} passed")

            if not new and not done_futs:
                idle += 1
                if idle > 360:
                    break
                time.sleep(5)

    passed = sum(1 for r in results if r.get("passed"))
    log(f"\nDone! {passed}/{len(results)} passed = {passed}%")

    with open(args.output, "w") as f:
        json.dump({"passed": passed, "total": len(results), "acc": passed / max(len(results), 1), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
