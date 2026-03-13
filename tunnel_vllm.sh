#!/bin/sh
# SSH tunnel: forward remote vLLM API port to local 0.0.0.0.
# Use this when the server only listens on 127.0.0.1 (nc 10.10.1.153 8000 refused, nc 127.0.0.1 8000 ok).
# Usage: ./tunnel_vllm.sh [local_port] [remote_port]
#   Default: local 8000 -> remote 8000.
#   If port in use: ./tunnel_vllm.sh 18080   or   ./tunnel_vllm.sh auto
#   Override host: REMOTE_HOST=host ./tunnel_vllm.sh

set -e

REMOTE_USER="${REMOTE_USER:-cloud-admin}"
REMOTE_HOST="${REMOTE_HOST:-10.10.1.153}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/cloud-admin.sshkey}"
REMOTE_PORT="${2:-8000}"

# First arg: port number or "auto" to pick first free port
if [ "$1" = "auto" ]; then
    LOCAL_PORT=""
    for p in 8000 8001 8002 8003 8004 8005 8010 8080 18080 28080; do
        if ! lsof -i ":$p" >/dev/null 2>&1; then
            LOCAL_PORT="$p"
            break
        fi
    done
    if [ -z "$LOCAL_PORT" ]; then
        echo "No free port found in 8000-8010, 8080, 18080, 28080. Try: ./tunnel_vllm.sh <port>"
        exit 1
    fi
    echo "[INFO] Using local port $LOCAL_PORT (run test with --port $LOCAL_PORT)"
else
    LOCAL_PORT="${1:-8000}"
fi

echo "Tunnel: local 0.0.0.0:${LOCAL_PORT} -> ${REMOTE_USER}@${REMOTE_HOST} localhost:${REMOTE_PORT}"
echo "vLLM API: http://localhost:${LOCAL_PORT}"
echo "Press Ctrl+C to stop."
echo ""

exec ssh -i "$SSH_KEY" \
    -o StrictHostKeyChecking=accept-new \
    -L "0.0.0.0:${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}" \
    -N
