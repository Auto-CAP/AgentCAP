#!/usr/bin/env python3
"""Generate per-task K8s Job YAMLs for SWE-bench Pro.

Each job has:
  - sidecar: the correct Docker image for that task (HTTP exec server)
  - runner: lightweight CPU container that connects to shared vLLM service

Usage:
    python gen_swebench_jobs.py --num-tasks 100 --vllm-url http://vllm-gptoss:30002/v1 --output-dir /tmp/swebench_jobs
    # Then: kubectl create -f /tmp/swebench_jobs/
"""
import argparse
import os
import textwrap

from datasets import load_dataset


EXEC_SERVER_CODE = r'''
import http.server, json, subprocess, socketserver

class ExecHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        cmd = body.get("cmd", "echo no command")
        timeout = body.get("timeout", 30)
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd="/app"
            )
            result = {"returncode": proc.returncode, "stdout": proc.stdout[-8000:], "stderr": proc.stderr[-4000:]}
        except subprocess.TimeoutExpired:
            result = {"returncode": 124, "stdout": "", "stderr": "timeout"}
        except Exception as e:
            result = {"returncode": 1, "stdout": "", "stderr": str(e)}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    def log_message(self, format, *args):
        pass

with socketserver.TCPServer(("", 9999), ExecHandler) as s:
    print("Exec server on :9999", flush=True)
    s.serve_forever()
'''


JOB_TEMPLATE = textwrap.dedent('''\
apiVersion: batch/v1
kind: Job
metadata:
  generateName: swe-task-{safe_id}-
  namespace: eidf230ns
  labels:
    app: swebench-batch
    task-index: "{index}"
    kueue.x-k8s.io/queue-name: eidf230ns-user-queue
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: swebench-batch
        task-index: "{index}"
        kueue.x-k8s.io/queue-name: eidf230ns-user-queue
    spec:
      restartPolicy: Never
      securityContext:
        fsGroup: 2000
      volumes:
        - name: llm-cache
          persistentVolumeClaim:
            claimName: llm-cache-pvc
      containers:
        # ---- Sidecar: SWE-bench task environment ----
        - name: swebench
          image: jefzda/sweap-images:{dockerhub_tag}
          command: ["python3", "-c"]
          args:
            - |
{exec_server_indented}
          ports:
            - containerPort: 9999
          workingDir: /app
          resources:
            requests:
              cpu: "1"
              memory: 4Gi
            limits:
              cpu: "2"
              memory: 8Gi

        # ---- Runner: connects to shared vLLM service ----
        - name: runner
          image: ghcr.io/pytorch/pytorch:2.10.0-cuda12.9-cudnn9-devel
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -uo pipefail
              cd /workspace
              . /workspace/venv_smoke/bin/activate
              pip install -e /workspace/AgentCAP_smoke 2>&1 | tail -1

              # Skip pip install if already installed (avoid PVC contention)
              python -c "import agent_cap" 2>/dev/null || pip install -e /workspace/AgentCAP_smoke 2>&1 | tail -1

              # Wait for sidecar
              for i in $(seq 1 30); do
                  if python -c "import urllib.request; urllib.request.urlopen('http://localhost:9999/exec', data=b'{{\"cmd\":\"echo ok\",\"timeout\":5}}', timeout=3)" 2>/dev/null; then
                      echo "Sidecar ready"; break
                  fi
                  sleep 2
              done

              # Wait for vLLM service
              echo "Waiting for vLLM at {vllm_url} ..."
              for i in $(seq 1 120); do
                  if python -c "import urllib.request; urllib.request.urlopen('{vllm_url}/models', timeout=5)" 2>/dev/null; then
                      echo "vLLM ready"; break
                  fi
                  sleep 5
              done

              echo "=== Running task {index}: {instance_id} ==="
              python -m agent_cap.runner.unified_runner \\
                  --model-name openai/gpt-oss-120b \\
                  --dataset swe-bench-pro \\
                  --backend swebench-k8s \\
                  --serving-engine vllm \\
                  --base-url {vllm_url} \\
                  --max-turns 50 \\
                  --num-tasks 1 \\
                  --task-offset {index} \\
                  --output-dir /workspace/results/swebench_batch_task{index_pad}

              echo "=== Done task {index} ==="
          env:
            - name: HF_HOME
              value: /workspace/hf_cache
            - name: HUGGINGFACE_HUB_CACHE
              value: /workspace/hf_cache/hub
            - name: PIP_CACHE_DIR
              value: /workspace/pip_cache
            - name: PIP_BREAK_SYSTEM_PACKAGES
              value: "1"
            - name: GITHUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: yufan-github-pat
                  key: GITHUB_TOKEN
          resources:
            requests:
              cpu: "2"
              memory: 8Gi
            limits:
              cpu: "2"
              memory: 16Gi
          volumeMounts:
            - name: llm-cache
              mountPath: /workspace
''')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--vllm-url", type=str, default="http://vllm-gptoss:30002/v1")
    parser.add_argument("--output-dir", type=str, default="/tmp/swebench_jobs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading SWE-bench Pro dataset...")
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")

    exec_indented = "\n".join(
        f"              {line}" for line in EXEC_SERVER_CODE.strip().split("\n")
    )

    for idx in range(args.task_offset, min(args.task_offset + args.num_tasks, len(ds))):
        ex = ds[idx]
        instance_id = ex["instance_id"]
        dockerhub_tag = ex["dockerhub_tag"]
        # Safe ID for K8s name (max 63 chars, lowercase, alphanumeric + dash)
        safe_id = instance_id[:40].lower().replace("_", "-").replace("/", "-").rstrip("-")

        job_yaml = JOB_TEMPLATE.format(
            index=idx,
            index_pad=f"{idx:03d}",
            safe_id=safe_id,
            instance_id=instance_id,
            dockerhub_tag=dockerhub_tag,
            vllm_url=args.vllm_url,
            exec_server_indented=exec_indented,
        )

        outpath = os.path.join(args.output_dir, f"task_{idx:03d}.yaml")
        with open(outpath, "w") as f:
            f.write(job_yaml)

    print(f"Generated {min(args.num_tasks, len(ds) - args.task_offset)} job YAMLs in {args.output_dir}")


if __name__ == "__main__":
    main()
