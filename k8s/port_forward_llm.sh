#!/usr/bin/env bash
# Keep a kubectl port-forward to the LLM service alive (auto-restart on drop).
# Usage: bash k8s/port_forward_llm.sh <service-name> [local_port]
set -u
SVC="${1:?usage: port_forward_llm.sh <service-name> [local_port]}"
PORT="${2:-8000}"
NAMESPACE="${NAMESPACE:-eidf230ns}"
echo "port-forwarding svc/$SVC -> 127.0.0.1:$PORT (ctrl-c to stop)"
while true; do
    SVC_PORT="$(kubectl -n "$NAMESPACE" get svc "$SVC" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null)"
    if [ -n "$SVC_PORT" ]; then
        kubectl -n "$NAMESPACE" port-forward "svc/$SVC" "$PORT:$SVC_PORT" 2>&1 | sed "s/^/[pf $SVC] /"
    fi
    echo "[pf $SVC] dropped, restarting in 3s..."
    sleep 3
done
