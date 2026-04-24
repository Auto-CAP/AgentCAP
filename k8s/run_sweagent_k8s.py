#!/usr/bin/env python3
"""SWE-bench Pro orchestrator that runs INSIDE K8s.

This script runs as a K8s Job (CPU only). It:
  1. Reads task queue from a JSONL file on PVC
  2. For each task, waits for its sidecar pod to be ready (created by login node)
  3. Connects to sidecar via Pod IP (no port-forward needed)
  4. Connects to vLLM via ClusterIP Service
  5. Runs SWE-agent
  6. Writes results to PVC
  7. Writes "done" marker so login node knows to create next sidecar

Login node runs a simple loop:
  - Monitors done markers on PVC
  - Creates new sidecar jobs for next tasks
  - Maintains 10 concurrent sidecars

Usage (inside K8s pod):
    python -u run_sweagent_k8s.py \
        --vllm-url http://vllm-gptoss:30002/v1 \
        --task-queue /workspace/task_queue.jsonl \
        --output-dir /workspace/results/sweagent_h200
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

print_lock = Lock()

def log(msg):
    with print_lock:
        print(msg, flush=True)


def wait_for_sidecar(task_idx, pod_ip, timeout=300):
    """Wait for swerex server at pod_ip:9999 to be ready."""
    for _ in range(timeout // 3):
        try:
            req = urllib.request.Request(
                f"http://{pod_ip}:9999/is_alive",
                headers={"X-API-Key": "token123"},
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def run_one_task(task_idx, instance_id, pod_ip, problem_statement,
                 vllm_url, output_dir, sweagent_dir):
    """Run SWE-agent on one task via swerex at pod_ip."""
    log(f"[task {task_idx}] START {instance_id[:60]}")

    if not wait_for_sidecar(task_idx, pod_ip):
        log(f"[task {task_idx}] sidecar not ready at {pod_ip}")
        return {"index": task_idx, "instance_id": instance_id, "status": "sidecar_failed"}

    log(f"[task {task_idx}] swerex ready at {pod_ip}:9999")

    try:
        task_output = output_dir / f"task_{task_idx:03d}"
        task_output.mkdir(parents=True, exist_ok=True)

        ps_file = task_output / "problem.txt"
        ps_file.write_text(problem_statement)

        cmd = [
            sys.executable, "-m", "sweagent", "run",
            "--config", str(sweagent_dir / "config" / "bash_only.yaml"),
            "--agent.model.name", f"hosted_vllm/openai/gpt-oss-120b",
            "--agent.model.api_base", vllm_url,
            "--agent.model.per_instance_cost_limit", "0",
            "--agent.model.total_cost_limit", "0",
            "--agent.model.per_instance_call_limit", "50",
            "--agent.model.completion_kwargs", '{"stop_token_ids": [200002, 200012]}',
            "--env.deployment.type", "remote",
            "--env.deployment.host", f"http://{pod_ip}",
            "--env.deployment.port", "9999",
            "--env.deployment.auth_token", "token123",
            "--env.repo.type", "preexisting",
            "--env.repo.repo_name", "app",
            "--problem_statement.path", str(ps_file),
        ]

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "dummy"

        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=1800, cwd=str(task_output))
        # Save SWE-agent stdout/stderr for debugging
        (task_output / "sweagent_stdout.log").write_text(r.stdout[-5000:] if r.stdout else "")
        (task_output / "sweagent_stderr.log").write_text(r.stderr[-5000:] if r.stderr else "")

        # Find patch - SWE-agent writes to trajectories/ relative to cwd
        patch = ""
        traj_files = list(task_output.rglob("*.traj"))
        for tf in traj_files:
            try:
                traj = json.loads(tf.read_text())
                p = traj.get("info", {}).get("submission", "") or traj.get("info", {}).get("model_patch", "")
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
            log(f"[task {task_idx}] DONE no patch")

        # Write done marker
        (task_output / "DONE").write_text(instance_id)

        return {"index": task_idx, "instance_id": instance_id,
                "status": "ok", "has_patch": bool(patch)}

    except subprocess.TimeoutExpired:
        log(f"[task {task_idx}] TIMEOUT")
        return {"index": task_idx, "instance_id": instance_id, "status": "timeout"}
    except Exception as exc:
        log(f"[task {task_idx}] ERROR: {exc}")
        return {"index": task_idx, "instance_id": instance_id, "status": f"error: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://vllm-gptoss:30002/v1")
    parser.add_argument("--task-queue", required=True, help="JSONL file with tasks (idx, instance_id, pod_ip, problem)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--total-tasks", type=int, default=100)
    parser.add_argument("--sweagent-dir", default="/workspace/swe_agent")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweagent_dir = Path(args.sweagent_dir)

    # Wait for vLLM
    log(f"Waiting for vLLM at {args.vllm_url} ...")
    for i in range(180):
        try:
            urllib.request.urlopen(f"{args.vllm_url}/models", timeout=5)
            log("vLLM ready!")
            break
        except Exception:
            if i % 12 == 0:
                log(f"  still waiting ({i*5}s)...")
            time.sleep(5)

    # Stream task queue - continuously read new lines
    total_expected = args.total_tasks
    results = []
    seen_lines = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        idle_count = 0

        while len(results) < total_expected:
            # Read new tasks from queue file
            new_tasks = []
            try:
                with open(args.task_queue) as f:
                    for i, line in enumerate(f):
                        if i < seen_lines:
                            continue
                        if line.strip():
                            new_tasks.append(json.loads(line))
                            seen_lines = i + 1
            except Exception:
                pass

            for t in new_tasks:
                log(f"Picked up task {t['idx']}")
                fut = pool.submit(
                    run_one_task, t["idx"], t["instance_id"], t["pod_ip"],
                    t["problem_statement"], args.vllm_url, output_dir, sweagent_dir,
                )
                futures[fut] = t["idx"]
                idle_count = 0

            # Collect completed futures
            done_futures = [f for f in futures if f.done()]
            for f in done_futures:
                result = f.result()
                results.append(result)
                del futures[f]
                patches = sum(1 for r in results if r.get("has_patch"))
                log(f"[progress] {len(results)}/{total_expected} done, {patches} patches — "
                    f"{result['instance_id'][:50]}: {result['status']}")

            if not new_tasks and not done_futures:
                idle_count += 1
                if idle_count > 360:  # 30 min idle = give up
                    log("Idle timeout, exiting")
                    break
                time.sleep(5)

    results.sort(key=lambda x: x["index"])
    with open(output_dir / "batch_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    ok = sum(1 for r in results if r["status"] == "ok")
    patches = sum(1 for r in results if r.get("has_patch"))
    log(f"\nDone! {ok}/{len(results)} ok, {patches} patches.")


if __name__ == "__main__":
    main()
