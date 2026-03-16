#!/bin/sh
# Self-destructing vLLM Docker installer for "curl | sh" use.
# - Will auto-clean after optional timer expires (except for HuggingFace model cache)
# - Only persists HuggingFace model files for cache/reuse
# - Prompts for HuggingFace token on deploy, cancels timer if desired

# POSIX shell compatible version

set -e

# --- Self-remove: schedule deletion of this (downloaded) script and temp files ---

cleanup() {
    # Do not kill SELF_DESTRUCT_TIMER_PID here: timer runs in nohup and must survive
    # script exit so the container is stopped after RUN_TIME_MINUTES. Only confirm_deployment kills it on "yes".
    # Remove this script after run when executed as a file (./install.sh, sh 1.sh). Do not remove when run as "curl | sh" ($0 is "sh").
    _self="$0"
    case "$_self" in
        */*|*.sh)
            if [ -f "$_self" ] && [ -w "$_self" ]; then
                rm -f -- "$_self"
            fi
            ;;
        *) ;;
    esac
    unset HUGGING_FACE_HUB_TOKEN
}
trap cleanup EXIT

# --- Read from terminal so piped stdin (e.g. yes | install.sh) doesn't flood prompts ---
read_tty() {
    if [ -c /dev/tty ]; then
        read "$@" </dev/tty
    else
        read "$@"
    fi
}

# --- Ask user for timer duration for self-destruct ---
# Set VLLM_NO_SELF_DESTRUCT=1 to disable the timer entirely (container runs until you docker stop it).
ask_for_timer() {
    if [ -n "$VLLM_NO_SELF_DESTRUCT" ]; then
        RUN_TIME_MINUTES=0
        echo "[INFO] Self-destruct timer disabled (VLLM_NO_SELF_DESTRUCT is set). Container will run until you stop it."
        return
    fi
    printf "Enter number of minutes to keep vLLM running before self-destruct [default: 240]: "
    # If user just presses Enter, that's fine (use default)
    read_tty -r input_minutes
    if [ -z "$input_minutes" ]; then
        RUN_TIME_MINUTES=240
    else
        # Check if input is a valid positive integer
        if echo "$input_minutes" | grep -Eq '^[0-9]+$'; then
            RUN_TIME_MINUTES="$input_minutes"
        else
            echo "[WARN] Invalid input, using default: 240 minutes"
            RUN_TIME_MINUTES=240
        fi
    fi
}

ask_for_timer

SELF_DESTRUCTED=0

start_timer() {
    # No timer when disabled via VLLM_NO_SELF_DESTRUCT
    [ "$RUN_TIME_MINUTES" -eq 0 ] 2>/dev/null && return 0
    # $1 = container name to stop when timer expires; $2 = path to cancel file (touch to cancel timer)
    _cid="${1:-vllm-openai}"
    _cancel="${2:-/tmp/vllm-timer-cancel-$$}"
    TIMER_CANCEL_FILE="$_cancel"
    # Kill any leftover timer from a previous run (nohup survives script/terminal; old timer would stop current container)
    _pidfile="/tmp/vllm-timer-pid-$_cid"
    if [ -f "$_pidfile" ]; then
        _oldpid=$(cat "$_pidfile" 2>/dev/null)
        kill "$_oldpid" 2>/dev/null || true
        rm -f "$_pidfile"
    fi
    # Timer runs in nohup so it survives script exit when user keeps the timer
    nohup sh -c "
        _i=0
        while [ \$_i -lt $RUN_TIME_MINUTES ]; do
            sleep 60
            [ -f '$_cancel' ] && exit 0
            _i=\$((_i + 1))
        done
        printf '\n[INFO] Timer expired - %s min. Stopping container %s...\n' $RUN_TIME_MINUTES '$_cid'
        docker stop '$_cid' 2>/dev/null || true
        rm -f '$_cancel'
    " >/dev/null 2>&1 &
    SELF_DESTRUCT_TIMER_PID=$!
    echo "$SELF_DESTRUCT_TIMER_PID" > "$_pidfile"
}

# --- Prompt for HuggingFace token (optional for public models; never stored) ---
prompt_for_token() {
    printf "Enter your HuggingFace Hub Token - optional for public models; press Enter to skip:\n"
    stty_saved=""
    if [ -c /dev/tty ]; then
        stty_saved=$(stty -g </dev/tty 2>/dev/null)
        stty -echo </dev/tty 2>/dev/null
    fi
    printf "> "
    read_tty HUGGING_FACE_HUB_TOKEN
    if [ -n "$stty_saved" ]; then
        stty "$stty_saved" </dev/tty 2>/dev/null
        printf "\n"
    fi
    export HUGGING_FACE_HUB_TOKEN
}

# --- Option to approve deployment and cancel timer ---
confirm_deployment() {
    if [ "$RUN_TIME_MINUTES" -eq 0 ] 2>/dev/null; then
        echo "[INFO] No self-destruct timer. Container will keep running until you run: docker stop $VLLM_CONTAINER_NAME"
        return 0
    fi
    printf "Deployment started. Type 'yes' and press Enter within %s minutes to keep this running without auto-destruction.\n" "$RUN_TIME_MINUTES"
    printf "Type yes to cancel self-destruct; or press Enter to keep the timer: "
    read_tty approve
    if [ "$approve" = "yes" ]; then
        touch "$TIMER_CANCEL_FILE" 2>/dev/null || true
        kill "$SELF_DESTRUCT_TIMER_PID" 2>/dev/null || true
        rm -f "/tmp/vllm-timer-pid-$VLLM_CONTAINER_NAME" 2>/dev/null || true
        echo "[INFO] Deployment approved. The installer will NOT self-destruct."
    else
        echo "[INFO] Self-destruct timer continues. Script and container will be removed after $RUN_TIME_MINUTES minutes."
    fi
}

# --- Only HuggingFace model cache is persisted ---
if [ -z "$VLLM_MODEL_CACHE" ]; then
    HUGGINGFACE_MODEL_CACHE="$HOME/.cache/huggingface"
else
    HUGGINGFACE_MODEL_CACHE="$VLLM_MODEL_CACHE"
fi
mkdir -p "$HUGGINGFACE_MODEL_CACHE"

# --- Install Docker if not present ---
ensure_docker() {
    if command -v docker >/dev/null 2>&1; then
        return 0
    fi
    if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
        echo "[ERROR] Docker not found and cannot install without root/sudo."
        return 1
    fi
    _run() { [ "$(id -u)" -eq 0 ] && "$@" || sudo "$@"; }
    if command -v apt-get >/dev/null 2>&1; then
        echo "[INFO] Docker not found. Installing Docker..."
        echo "[INFO] Step 1/6: apt update and install ca-certificates, curl..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
        _run env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates curl 2>/dev/null || true
        echo "[INFO] Step 2/6: adding Docker GPG key..."
        _run install -m 0755 -d /etc/apt/keyrings
        _run curl -fsSL --connect-timeout 30 --max-time 60 https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        _run chmod a+r /etc/apt/keyrings/docker.asc
        echo "[INFO] Step 3/6: adding Docker repository..."
        _suite="noble"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            _suite="${UBUNTU_CODENAME:-$VERSION_CODENAME}"
            [ -z "$_suite" ] && _suite="noble"
        fi
        {
            echo "Types: deb"
            echo "URIs: https://download.docker.com/linux/ubuntu"
            echo "Suites: $_suite"
            echo "Components: stable"
            echo "Signed-By: /etc/apt/keyrings/docker.asc"
        } | _run tee /etc/apt/sources.list.d/docker.sources >/dev/null
        echo "[INFO] Step 4/6: apt update..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
        echo "[INFO] Step 5/6: installing Docker packages..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 2>/dev/null || return 1
        echo "[INFO] Step 6/6: starting Docker..."
        _run systemctl start docker 2>/dev/null || _run service docker start 2>/dev/null || true
        echo "[INFO] Docker installed and started."
    elif command -v yum >/dev/null 2>&1 || command -v dnf >/dev/null 2>&1; then
        echo "[INFO] Docker not found. On RHEL/CentOS please install Docker first (e.g. dnf install docker-ce) then re-run this script."
        return 1
    else
        echo "[ERROR] Docker not found and no supported package manager to install it."
        return 1
    fi
}

# --- Install nvidia-container-toolkit if Docker has no NVIDIA runtime ---
ensure_nvidia_container_toolkit() {
    if docker info 2>/dev/null | grep -q 'nvidia'; then
        return 0
    fi
    # Need root to install
    if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
        return 1
    fi
    _run() { [ "$(id -u)" -eq 0 ] && "$@" || sudo "$@"; }
    if command -v apt-get >/dev/null 2>&1; then
        echo "[INFO] Installing nvidia-container-toolkit - required for GPU..."
        echo "[INFO] Step 1/7: apt-get update..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
        echo "[INFO] Step 2/7: installing ca-certificates and curl..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates curl 2>/dev/null || true
        echo "[INFO] Step 3/7: adding NVIDIA repo GPG key..."
        curl -fsSL --connect-timeout 30 --max-time 60 https://nvidia.github.io/libnvidia-container/gpgkey | _run gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
        echo "[INFO] Step 4/7: adding NVIDIA repo list..."
        curl -s -L --connect-timeout 30 --max-time 60 https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            _run tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        echo "[INFO] Step 5/7: apt-get update (NVIDIA repo)..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
        echo "[INFO] Step 6/7: installing nvidia-container-toolkit..."
        _run env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-container-toolkit 2>/dev/null || return 1
        echo "[INFO] Step 7/7: configuring runtime and restarting Docker (may take 1–2 min)..."
        _run nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
        _run systemctl restart docker 2>/dev/null || _run service docker restart 2>/dev/null || true
        echo "[INFO] nvidia-container-toolkit installed. Docker restarted."
    elif command -v yum >/dev/null 2>&1 || command -v dnf >/dev/null 2>&1; then
        echo "[INFO] Installing nvidia-container-toolkit - required for GPU..."
        _pkginstall() { command -v dnf >/dev/null 2>&1 && _run dnf install -y "$@" || _run yum install -y "$@"; }
        echo "[INFO] Step 1/5: installing curl..."
        _pkginstall curl
        echo "[INFO] Step 2/5: adding NVIDIA repo..."
        curl -s -L --connect-timeout 30 --max-time 60 https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | _run tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
        echo "[INFO] Step 3/5: installing nvidia-container-toolkit..."
        _pkginstall nvidia-container-toolkit 2>/dev/null || return 1
        echo "[INFO] Step 4/5: configuring runtime and restarting Docker (may take 1–2 min)..."
        _run nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
        _run systemctl restart docker 2>/dev/null || _run service docker restart 2>/dev/null || true
        echo "[INFO] nvidia-container-toolkit installed. Docker restarted."
    else
        return 1
    fi
    # Give Docker a moment to see the new runtime
    sleep 2
}

if ! command -v docker >/dev/null 2>&1; then
    echo "[INFO] Docker not detected. Attempting to install Docker..."
    if ! ensure_docker; then
        echo "[ERROR] Could not install Docker. Please install Docker and re-run this script."
        exit 1
    fi
fi

DOCKER_HAS_NVIDIA=0
if docker info 2>/dev/null | grep -q 'nvidia'; then
    DOCKER_HAS_NVIDIA=1
fi

if [ "$DOCKER_HAS_NVIDIA" -eq 0 ]; then
    echo "[INFO] NVIDIA runtime not detected. Attempting to install nvidia-container-toolkit..."
    if ensure_nvidia_container_toolkit; then
        if docker info 2>/dev/null | grep -q 'nvidia'; then
            DOCKER_HAS_NVIDIA=1
            echo "[INFO] NVIDIA runtime is now available."
        fi
    fi
fi

# --- Docker NVIDIA runtime compatibility check ---
if [ "$DOCKER_HAS_NVIDIA" -eq 1 ]; then
    DOCKER_RUNTIME_ARGS="--gpus all --runtime=nvidia"
else
    # Check if 'docker run --gpus' is supported, else warn and set no-GPU
    if docker run --help 2>&1 | grep -q -- '--gpus'; then
        DOCKER_RUNTIME_ARGS="--gpus all"
        echo "[WARN] Detected no Nvidia runtime, but '--gpus all' supported. Proceeding without '--runtime=nvidia' option."
    else
        DOCKER_RUNTIME_ARGS=""
        echo "[WARN] No GPU detected or supported by Docker. vLLM may not start or may run very slowly on CPU-only."
    fi
fi
# --- Main vLLM config (minimax-M2.5 230B MoE for H200; use vllm-openai:nightly if :latest lacks MiniMax support) ---
GPU_ID="0,1,2,3,4,5,6,7"
PORT=8000
# Bind port to host IP so Docker creates listen+NAT for it. Optional: set VLLM_HOST_IP to override.
if [ -z "$VLLM_HOST_IP" ]; then
    VLLM_HOST_IP=$(ip -4 addr show bond0 2>/dev/null | sed -n 's/.*inet \([0-9.]*\)\/.*/\1/p')
    [ -z "$VLLM_HOST_IP" ] && VLLM_HOST_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p')
    [ -z "$VLLM_HOST_IP" ] && VLLM_HOST_IP=$(ip -4 route show default 2>/dev/null | sed -n '1s/.* src \([0-9.]*\).*/\1/p')
    [ -z "$VLLM_HOST_IP" ] && VLLM_HOST_IP=$(hostname -I 2>/dev/null | sed 's/ .*//')
fi
MAX_MODEL_LEN=196608
MAX_NUM_SEQS=128
GPU_MEMORY_UTILIZATION=0.92
DTYPE="bfloat16"
MODEL_PATH="MiniMaxAI/minimax-M2.5"
SERVED_MODEL_NAME="minimax-M2.5"
# minimax-M2.5 is MoE; pure TP8 not supported — use TP8+EP with expert parallel
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

# Count GPUs in GPU_ID comma-separated for tensor parallel size
old_IFS="$IFS"
IFS=,
set -- $GPU_ID
TENSOR_PARALLEL_SIZE=$#
IFS="$old_IFS"

prompt_for_token

# Run Docker container in background detached; kept after exit so you can docker logs on failure
VLLM_CONTAINER_NAME="vllm-openai"
TIMER_CANCEL_FILE="/tmp/vllm-timer-cancel-$$"
docker rm -f "$VLLM_CONTAINER_NAME" 2>/dev/null || true
# Escape token so a double-quote in it cannot break the docker run line
_HF_TOKEN_SAFE=$(printf '%s' "${HUGGING_FACE_HUB_TOKEN}" | sed 's/"/\\"/g')
PORT_BIND="-p 127.0.0.1:${PORT}:8000"
[ -n "$VLLM_HOST_IP" ] && PORT_BIND="$PORT_BIND -p ${VLLM_HOST_IP}:${PORT}:8000"
# Put --name after $DOCKER_RUNTIME_ARGS so a trailing dash in env cannot produce "---name"
# shellcheck disable=SC2086
eval docker run --rm -d $DOCKER_RUNTIME_ARGS --name "$VLLM_CONTAINER_NAME" \
    -v "$HUGGINGFACE_MODEL_CACHE":/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=${_HF_TOKEN_SAFE}" \
    --env "VLLM_API_KEY=not-needed" \
    --env "VLLM_FLOAT32_MATMUL_PRECISION=high" \
    $PORT_BIND \
    --ipc=host \
    "$VLLM_IMAGE" \
    "$MODEL_PATH" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --enable-expert-parallel \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --swap-space 0 \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS"

start_timer "$VLLM_CONTAINER_NAME" "$TIMER_CANCEL_FILE"

echo ""
if [ -n "$VLLM_HOST_IP" ]; then
    echo "[INFO] API port ${PORT} bound on 127.0.0.1 and ${VLLM_HOST_IP} (nc -zv ${VLLM_HOST_IP} ${PORT} to test)."
    echo "[INFO] If the host IP is refused, use SSH tunnel from your laptop: ./tunnel_vllm.sh then http://localhost:${PORT}"
else
    echo "[INFO] API port ${PORT} bound on 127.0.0.1 only. Set VLLM_HOST_IP to your host IP to bind it too."
fi
echo "vLLM is running in the background. Container: $VLLM_CONTAINER_NAME."
echo "First startup can take 10-15 min (model load + compile). If curl gets 'Connection reset' or empty reply, wait and retry."
echo "You can close this session; the server will keep running."
echo ""
echo "OpenAI-compatible API: http://<your_server>:$PORT/v1"
echo "To stop the server: docker stop $VLLM_CONTAINER_NAME"
echo "To view logs:  docker logs $VLLM_CONTAINER_NAME   or  docker logs -f $VLLM_CONTAINER_NAME"
echo "If the container exited, it is kept so you can still run the above to see the error."
echo ""

# Optional: cancel self-destruct timer (type 'yes') or press Enter to keep timer
confirm_deployment

echo "Serving $SERVED_MODEL_NAME on port $PORT. HuggingFace cache is kept for reuse."
