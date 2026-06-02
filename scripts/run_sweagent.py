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
        if not self.image_repo:
            image = dockerhub_tag
        else:
            image = f"{self.image_repo}:{dockerhub_tag}"
        return {"image": image}

    def sweagent_args(self, ctx):
        return [
            "--env.deployment.type", "docker",
            "--env.deployment.image", ctx["image"],
            "--env.repo.type", "preexisting",
            "--env.repo.repo_name", "testbed",
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
        if dockerhub_tag.startswith("docker.io/") or "://" in dockerhub_tag:
            return {"image": dockerhub_tag}
        if not self.image_repo:
            return {"image": dockerhub_tag}
        return {"image": f"docker.io/{self.image_repo}:{dockerhub_tag}"}

    def sweagent_args(self, ctx):
        return [
            "--env.deployment.type", "modal",
            "--env.deployment.image", ctx["image"],
            "--env.deployment.deployment_timeout", "14400",
            "--env.deployment.runtime_timeout", "900",
            "--env.repo.type", "preexisting",
            "--env.repo.repo_name", "testbed",
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
                 vllm_url, output_dir, sweagent_dir, deploy, model_name,
                 pf_mgr=None):
    log(f"[task {task_idx}] START {instance_id[:60]}")
    if pf_mgr:
        pf_mgr.ensure_alive()

    ctx = deploy.prepare(task_idx, instance_id, dockerhub_tag)
    if ctx is None:
        return {"index": task_idx, "instance_id": instance_id, "status": "deploy_failed"}

    import time
    t_start_task = time.perf_counter()
    try:
        task_output = output_dir / f"task_{task_idx:03d}"
        task_output.mkdir(parents=True, exist_ok=True)
        ps_file = task_output / "problem.txt"
        rules = (
            "MANDATORY RULES (read carefully before starting):\n"
            "- Do NOT modify any test files. Only edit source code.\n"
            "- Do NOT submit until you have run the tests and they all pass.\n"
            "- If tests fail, keep trying — analyze the error, fix your code, re-run. "
            "Do NOT give up after one failure.\n\n"
        )
        ps_file.write_text(rules + problem_statement)

        traj_dir = task_output / "sweagent_traj"
        traj_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "sweagent", "run",
            "--config", str(sweagent_dir / "config" / "default.yaml"),
            "--agent.model.name", model_name,
            "--agent.model.api_base", vllm_url,
            "--agent.model.per_instance_cost_limit", "0",
            "--agent.model.total_cost_limit", "0",
            "--agent.model.per_instance_call_limit", "200",
            "--agent.model.completion_kwargs", '{"extra_body": {"stop_token_ids": [200012, 200002]}}',
            "--agent.templates.put_demos_in_history", "false",
            "--problem_statement.path", str(ps_file),
            "--output_dir", str(traj_dir),
        ] + deploy.sweagent_args(ctx)

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "dummy"
        env["SWEAGENT_STREAM_STATS_PATH"] = str(task_output / "stream_stats.jsonl")

        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=1800, cwd=str(sweagent_dir))
        (task_output / "sweagent_stdout.log").write_text(r.stdout or "")
        (task_output / "sweagent_stderr.log").write_text(r.stderr or "")

        traj_files = list(traj_dir.rglob("*.traj"))
        patch = ""
        tool_calls_count = 0
        for tf in sorted(traj_files, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                traj = json.loads(tf.read_text())
                p = traj.get("info", {}).get("submission") or traj.get("info", {}).get("model_patch") or ""
                hist = traj.get("history") or traj.get("trajectory") or []
                tool_calls_count = sum(len(m.get("tool_calls") or []) for m in hist)
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

        stats_path = task_output / "stream_stats.jsonl"
        prompt_total = completion_total = reasoning_total = cached_total = 0
        ttft_list: list[float] = []
        tpot_list: list[float] = []
        requests = 0
        if stats_path.exists():
            for line in stats_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_total += int(s.get("prompt_tokens") or 0)
                completion_total += int(s.get("completion_tokens") or 0)
                reasoning_total += int(s.get("reasoning_tokens") or 0)
                cached_total += int(s.get("cached_tokens") or 0)
                if s.get("ttft_ms"):
                    ttft_list.append(float(s["ttft_ms"]))
                if s.get("tpot_ms"):
                    tpot_list.append(float(s["tpot_ms"]))
                requests += 1
        def _mean(xs):
            return (sum(xs) / len(xs)) if xs else 0.0
        def _p50(xs):
            if not xs:
                return 0.0
            ys = sorted(xs)
            return ys[len(ys) // 2]
        e2e = time.perf_counter() - t_start_task
        usage = {
            "input_tokens": prompt_total,
            "output_tokens": completion_total + reasoning_total,
            "completion_tokens": completion_total,
            "reasoning_tokens": reasoning_total,
            "cached_tokens": cached_total,
            "requests": requests,
        }
        result_row = {
            "task_id": instance_id,
            "strategy": "single",
            "output_text": patch,
            "e2e_latency_s": e2e,
            "per_role_usage": {"agent": usage},
            "total_usage": usage,
            "errors": [],
            "num_turns": requests,
            "tool_calls": tool_calls_count,
            "extras": {
                "ttft_ms_mean": _mean(ttft_list),
                "ttft_ms_p50": _p50(ttft_list),
                "tpot_ms_mean": _mean(tpot_list),
                "tpot_ms_p50": _p50(tpot_list),
                "has_patch": bool(patch),
                "sweagent_rc": r.returncode,
            },
            "eval_passed": False,
            "eval_score": 0.0,
            "eval_details": {"evaluator": "swebench"},
        }
        (task_output / "result.json").write_text(json.dumps(result_row, indent=2))
        return {"index": task_idx, "instance_id": instance_id,
                "status": "ok", "has_patch": bool(patch), "result": result_row}

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
    ap.add_argument("--task-indices", default=None,
                    help="Path to JSON file with 'indices' or 'new_indices' key, "
                         "or a comma-separated list of dataset row indices. "
                         "When provided, --num-tasks/--task-offset are ignored.")
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
    ap.add_argument("--model", default="openai/unsloth/gpt-oss-120b",
                    help="LiteLLM model id. Use openai/<hf-id> for vllm/sglang OpenAI-compat endpoints "
                         "(hosted_vllm/ prefix drops prompt_tokens_details.cached_tokens).")
    args = ap.parse_args()

    if not (args.vllm_url or args.vllm_job):
        ap.error("Either --vllm-url or --vllm-job must be provided")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweagent_dir = Path(args.sweagent_dir)

    log(f"Loading dataset {args.dataset}...")
    if args.dataset == "swe-bench-lite":
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        def _tag(ex):
            iid = ex["instance_id"].lower().replace("/", "__")
            if args.deployment == "modal":
                escaped = iid.replace("__", "_1776_")
                return f"docker.io/swebench/sweb.eval.x86_64.{escaped}:latest"
            if args.image_repo:
                return f"swebench/sweb.eval.x86_64.{iid}"
            return f"sweb.eval.x86_64.{iid}:latest"
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

    if args.task_indices:
        if args.task_indices.endswith(".json"):
            spec = json.loads(Path(args.task_indices).read_text())
            task_range = spec.get("indices") or spec.get("new_indices") or []
            log(f"Loaded {len(task_range)} indices from {args.task_indices}")
        else:
            task_range = [int(x) for x in args.task_indices.split(",")]
    else:
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
                    vllm_url, output_dir, sweagent_dir, deploy, args.model,
                    pf_mgr,
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
        json.dump([{k: v for k, v in r.items() if k != "result"} for r in results], fh, indent=2)
    ok = sum(1 for r in results if r["status"] == "ok")
    patches = sum(1 for r in results if r.get("has_patch"))
    log(f"\nDone! {ok}/{len(results)} ok, {patches} patches.")

    preds = []
    for r in results:
        if r.get("has_patch"):
            preds.append({
                "instance_id": r["instance_id"],
                "model_patch": (output_dir / f"task_{r['index']:03d}" / "patch.diff").read_text(),
                "model_name_or_path": args.model.replace("/", "-").replace(":", "-"),
            })
    preds_path = output_dir / "predictions.json"
    preds_path.write_text(json.dumps(preds, indent=2))
    log(f"Wrote {len(preds)} predictions to {preds_path}")

    if preds:
        run_id = f"sweagent_{output_dir.name[-40:]}"
        eval_cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "princeton-nlp/SWE-bench_Lite" if args.dataset == "swe-bench-lite" else "ScaleAI/SWE-bench_Pro",
            "--predictions_path", str(preds_path),
            "--max_workers", str(min(args.concurrency, 8)),
            "--run_id", run_id,
            "--cache_level", "instance",
        ]
        log(f"Running swebench evaluator ({len(preds)} predictions)...")
        with open(output_dir / "eval.log", "w") as ef:
            subprocess.run(eval_cmd, stdout=ef, stderr=subprocess.STDOUT, timeout=3600 * 6)

    eval_reports_dir = Path("logs/run_evaluation") / (
        f"sweagent_{output_dir.name[-40:]}" if preds else ""
    )
    eval_results: dict[str, dict] = {}
    model_name_dirs = list(eval_reports_dir.glob("*")) if eval_reports_dir.exists() else []
    for model_dir in model_name_dirs:
        for rep in model_dir.glob("*/report.json"):
            iid = rep.parent.name
            try:
                info = json.loads(rep.read_text()).get(iid, {})
                eval_results[iid] = {"resolved": bool(info.get("resolved")), "details": info}
            except Exception:
                pass

    rows_path = output_dir / "results.jsonl"
    with rows_path.open("w") as fh:
        for r in results:
            row = r.get("result") or {
                "task_id": r["instance_id"],
                "strategy": "single",
                "output_text": "",
                "e2e_latency_s": 0.0,
                "per_role_usage": {},
                "total_usage": {},
                "errors": [r.get("status", "")] if r.get("status") != "ok" else [],
                "num_turns": 0,
                "extras": {},
                "eval_passed": False,
                "eval_score": 0.0,
                "eval_details": {"evaluator": "swebench"},
            }
            ev = eval_results.get(r["instance_id"])
            if ev:
                row["eval_passed"] = ev["resolved"]
                row["eval_score"] = 1.0 if ev["resolved"] else 0.0
                row["eval_details"] = {"evaluator": "swebench", **ev["details"]}
            fh.write(json.dumps(row) + "\n")
    log(f"Wrote per-task rows to {rows_path}")

    n = len(results)
    pass_ = sum(1 for r in results if eval_results.get(r["instance_id"], {}).get("resolved"))
    acc = pass_ / n if n else 0.0
    log(f"FINAL: n={n}  pass={pass_}/{n}  acc={acc:.3f}  evaluator=swebench")


if __name__ == "__main__":
    main()
