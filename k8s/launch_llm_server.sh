#!/usr/bin/env bash
# Launch an sglang/vllm OFFICIAL-image server for unsloth/gpt-oss-120b on EIDF k8s.
#
# Usage:
#   bash k8s/launch_llm_server.sh <sglang|vllm> <a100|h100|h200>          # create
#   bash k8s/launch_llm_server.sh <sglang|vllm> <a100|h100|h200> --stop   # delete
#
# Images: lmsysorg/sglang:v0.5.9  /  vllm/vllm-openai:v0.21.0
# Flags follow scripts/run_swebench_docker_100.sh header:
#   SGLang: --reasoning-parser gpt-oss --tool-call-parser gpt-oss
#           --context-length 131072 --enable-cache-report
#   vLLM:   --reasoning-parser openai_gptoss --tool-call-parser openai
#           --max-model-len 131072 --enable-prompt-tokens-details
#
# After launch:  kubectl port-forward svc/<engine>-gptoss-<gpu> 8000:8000
set -euo pipefail

ENGINE="${1:?usage: launch_llm_server.sh <sglang|vllm> <a100|h100|h200> [--stop]}"
GPU="${2:?usage: launch_llm_server.sh <sglang|vllm> <a100|h100|h200> [--stop]}"
NAMESPACE="${NAMESPACE:-eidf230ns}"
MODEL_PATH="/workspace/models/unsloth/gpt-oss-120b"
SERVED_NAME="unsloth/gpt-oss-120b"
NAME="${ENGINE}-gptoss-${GPU}"

case "$GPU" in
  a100) PRODUCT="NVIDIA-A100-SXM4-80GB"; TP=2 ;;
  h100) PRODUCT="NVIDIA-H100-80GB-HBM3"; TP=2 ;;
  h200) PRODUCT="NVIDIA-H200";           TP=1 ;;
  *) echo "unknown gpu: $GPU"; exit 1 ;;
esac

if [[ "${3:-}" == "--stop" ]]; then
    kubectl -n "$NAMESPACE" delete job -l "app=$NAME" --ignore-not-found=true
    kubectl -n "$NAMESPACE" delete svc "$NAME" --ignore-not-found=true
    exit 0
fi

case "$ENGINE" in
  sglang)
    IMAGE="lmsysorg/sglang:v0.5.9"
    CMD=$(cat <<EOF
exec python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --served-model-name $SERVED_NAME \
    --host 0.0.0.0 --port 8000 \
    --tp $TP \
    --context-length 131072 \
    --reasoning-parser gpt-oss \
    --tool-call-parser gpt-oss \
    --enable-cache-report \
    --mem-fraction-static 0.85
EOF
)
    ;;
  vllm)
    IMAGE="vllm/vllm-openai:v0.21.0"
    CMD=$(cat <<EOF
exec vllm serve $MODEL_PATH \
    --served-model-name $SERVED_NAME \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size $TP \
    --max-model-len 131072 \
    --reasoning-parser openai_gptoss \
    --tool-call-parser openai \
    --enable-auto-tool-choice \
    --enable-prompt-tokens-details \
    --gpu-memory-utilization 0.92
EOF
)
    ;;
  *) echo "unknown engine: $ENGINE"; exit 1 ;;
esac

kubectl -n "$NAMESPACE" create -f - <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  generateName: ${NAME}-
  namespace: ${NAMESPACE}
  labels:
    app: ${NAME}
    kueue.x-k8s.io/queue-name: ${NAMESPACE}-user-queue
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: ${NAME}
    spec:
      restartPolicy: Never
      nodeSelector:
        nvidia.com/gpu.present: "true"
        nvidia.com/gpu.product: ${PRODUCT}
      securityContext:
        fsGroup: 2000
      volumes:
        - name: llm-cache
          persistentVolumeClaim:
            claimName: llm-cache-pvc
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
      containers:
        - name: ${ENGINE}
          image: ${IMAGE}
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              nvidia-smi -L
              ${CMD//$'\n'/ }
          ports:
            - containerPort: 8000
          env:
            - name: HF_HOME
              value: /tmp/hf_cache   # llm-cache-pvc is at quota; keep caches off it
          resources:
            requests:
              cpu: "8"
              memory: 128Gi
              nvidia.com/gpu: "${TP}"
            limits:
              cpu: "8"
              memory: 128Gi
              nvidia.com/gpu: "${TP}"
          volumeMounts:
            - name: llm-cache
              mountPath: /workspace
            - name: dshm
              mountPath: /dev/shm
YAML

kubectl -n "$NAMESPACE" get svc "$NAME" >/dev/null 2>&1 || kubectl -n "$NAMESPACE" create -f - <<YAML
apiVersion: v1
kind: Service
metadata:
  name: ${NAME}
  namespace: ${NAMESPACE}
spec:
  selector:
    app: ${NAME}
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
YAML

echo "launched $NAME (image=$IMAGE, tp=$TP, gpu=$PRODUCT)"
echo "watch:  kubectl -n $NAMESPACE get pods -l app=$NAME -w"
echo "logs:   kubectl -n $NAMESPACE logs -l app=$NAME -f"
echo "tunnel: bash k8s/port_forward_llm.sh $NAME"
