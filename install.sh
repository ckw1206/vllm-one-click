#!/bin/sh
# Self-destructing one-click image-generation server installer for "curl | sh" use.
# - Serves a Hugging Face *diffusers* text-to-image model (default: Qwen/Qwen-Image)
#   behind an OpenAI-compatible /v1/images/generations API on port 8000.
# - NOTE: standard vLLM cannot serve diffusion image models, so this uses diffusers.
#   The whole server (FastAPI + Dockerfile) is embedded below so this stays ONE file.
# - Will auto-clean after optional timer expires (except for HuggingFace model cache).
# - Only persists HuggingFace model files for cache/reuse.
# - Prompts for HuggingFace token on deploy, cancels timer if desired.
#
# POSIX shell compatible version

set -e

# --- Self-remove: schedule deletion of this (downloaded) script and temp files ---
BUILD_DIR=""
cleanup() {
    _self="$0"
    case "$_self" in
        */*|*.sh)
            if [ -f "$_self" ] && [ -w "$_self" ]; then
                rm -f -- "$_self"
            fi
            ;;
        *) ;;
    esac
    [ -n "$BUILD_DIR" ] && [ -d "$BUILD_DIR" ] && rm -rf -- "$BUILD_DIR"
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
# Set NO_SELF_DESTRUCT=1 to disable the timer entirely (container runs until you docker stop it).
ask_for_timer() {
    if [ -n "$NO_SELF_DESTRUCT" ] || [ -n "$VLLM_NO_SELF_DESTRUCT" ]; then
        RUN_TIME_MINUTES=0
        echo "[INFO] Self-destruct timer disabled. Container will run until you stop it."
        return
    fi
    printf "Enter number of minutes to keep the server running before self-destruct [default: 240]: "
    read_tty -r input_minutes
    if [ -z "$input_minutes" ]; then
        RUN_TIME_MINUTES=240
    else
        if echo "$input_minutes" | grep -Eq '^[0-9]+$'; then
            RUN_TIME_MINUTES="$input_minutes"
        else
            echo "[WARN] Invalid input, using default: 240 minutes"
            RUN_TIME_MINUTES=240
        fi
    fi
}

ask_for_timer

start_timer() {
    [ "$RUN_TIME_MINUTES" -eq 0 ] 2>/dev/null && return 0
    _cid="${1:-image-api}"
    _cancel="${2:-/tmp/imgsrv-timer-cancel-$$}"
    TIMER_CANCEL_FILE="$_cancel"
    # Kill any leftover timer from a previous run (glob by container name, any PID suffix)
    _oldpidfile=$(ls /tmp/imgsrv-timer-pid-${_cid}-* 2>/dev/null | head -1)
    if [ -n "$_oldpidfile" ]; then
        _oldpid=$(cat "$_oldpidfile" 2>/dev/null)
        kill "$_oldpid" 2>/dev/null || true
        rm -f "$_oldpidfile"
    fi
    # PID file includes $$ so concurrent runs don't stomp each other
    _pidfile="/tmp/imgsrv-timer-pid-${_cid}-$$"
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
        echo "[INFO] No self-destruct timer. Container will keep running until you run: docker stop $CONTAINER_NAME"
        return 0
    fi
    printf "Deployment started. Type 'yes' and press Enter within %s minutes to keep this running without auto-destruction.\n" "$RUN_TIME_MINUTES"
    printf "Type yes to cancel self-destruct; or press Enter to keep the timer: "
    read_tty approve
    if [ "$approve" = "yes" ]; then
        touch "$TIMER_CANCEL_FILE" 2>/dev/null || true
        kill "$SELF_DESTRUCT_TIMER_PID" 2>/dev/null || true
        rm -f "/tmp/imgsrv-timer-pid-$CONTAINER_NAME-$$" 2>/dev/null || true
        echo "[INFO] Deployment approved. The installer will NOT self-destruct."
    else
        echo "[INFO] Self-destruct timer continues. Script and container will be removed after $RUN_TIME_MINUTES minutes."
    fi
}

# --- Only HuggingFace model cache is persisted ---
if [ -z "$MODEL_CACHE" ] && [ -z "$VLLM_MODEL_CACHE" ]; then
    HUGGINGFACE_MODEL_CACHE="$HOME/.cache/huggingface"
else
    HUGGINGFACE_MODEL_CACHE="${MODEL_CACHE:-$VLLM_MODEL_CACHE}"
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
        echo "[INFO] Step 7/7: configuring runtime and restarting Docker (may take 1-2 min)..."
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
        echo "[INFO] Step 4/5: configuring runtime and restarting Docker (may take 1-2 min)..."
        _run nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
        _run systemctl restart docker 2>/dev/null || _run service docker restart 2>/dev/null || true
        echo "[INFO] nvidia-container-toolkit installed. Docker restarted."
    else
        return 1
    fi
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
    if docker run --help 2>&1 | grep -q -- '--gpus'; then
        DOCKER_RUNTIME_ARGS="--gpus all"
        echo "[WARN] Detected no Nvidia runtime, but '--gpus all' supported. Proceeding without '--runtime=nvidia' option."
    else
        DOCKER_RUNTIME_ARGS=""
        echo "[WARN] No GPU detected or supported by Docker. Image generation will be unusably slow on CPU."
    fi
fi

# --- Main config (override any of these via environment) ---
# Single replica -> a single GPU. Set GPU_ID to pick which one (default 0).
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
# diffusers model repo. Default Qwen/Qwen-Image; swap to black-forest-labs/FLUX.2-dev etc.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen-Image}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen-Image}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
IMAGE_TAG="${IMAGE_TAG:-diffusers-image-api:latest}"
# Base image providing torch + CUDA; diffusers/transformers are pip-installed on top.
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
CONTAINER_NAME="image-api"

# Default: bind on all interfaces (0.0.0.0). Set HOST_IP to bind to a specific IP instead.
HOST_IP="${HOST_IP:-$VLLM_HOST_IP}"
if [ -n "$HOST_IP" ] && [ "$HOST_IP" != "0.0.0.0" ]; then
    PORT_BIND="-p ${HOST_IP}:${PORT}:8000"
else
    PORT_BIND="-p 0.0.0.0:${PORT}:8000"
fi

# --- Build the diffusers image server image (embedded; keeps this a single file) ---
BUILD_DIR=$(mktemp -d)

cat > "$BUILD_DIR/Dockerfile" <<DOCKEREOF
FROM ${BASE_IMAGE}
ENV PYTHONUNBUFFERED=1 \\
    HF_HUB_ENABLE_HF_TRANSFER=1 \\
    PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir -U \\
        diffusers transformers accelerate safetensors sentencepiece \\
        protobuf "fastapi" "uvicorn[standard]" pillow hf_transfer
COPY server.py /app/server.py
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKEREOF

cat > "$BUILD_DIR/server.py" <<'PYEOF'
#!/usr/bin/env python3
"""OpenAI-compatible text-to-image server backed by Hugging Face diffusers.

Default model: Qwen/Qwen-Image (set MODEL_PATH to swap). Pipeline kwargs are
filtered against the loaded pipeline signature, so the same server works across
diffusers pipelines (Qwen-Image: true_cfg_scale, FLUX: guidance_scale, ...).

Also serves a minimal browser UI at GET / (open http://<host>:8000/ to generate
images interactively). The OpenAI API stays at POST /v1/images/generations.
"""
import base64
import inspect
import io
import os
import threading
import time
import uuid

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen-Image")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "Qwen-Image")
API_KEY = os.environ.get("API_KEY") or os.environ.get("VLLM_API_KEY") or ""
DEFAULT_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "50"))
DEFAULT_TRUE_CFG = float(os.environ.get("TRUE_CFG_SCALE", "4.0"))
DEFAULT_GUIDANCE = float(os.environ.get("GUIDANCE_SCALE", "4.0"))
DEFAULT_NEGATIVE = os.environ.get("NEGATIVE_PROMPT", " ")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") not in ("0", "false", "False", "")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/tmp/generated_images")
MAX_DIM = int(os.environ.get("MAX_DIM", "2048"))

os.makedirs(IMAGE_DIR, exist_ok=True)

app = FastAPI(title="diffusers OpenAI-compatible image server")

_pipe = None
_ready = threading.Event()
_load_error = None
_gpu_lock = threading.Lock()  # one generation at a time on a single replica


def _load_model():
    global _pipe, _load_error
    try:
        from diffusers import DiffusionPipeline

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=dtype, trust_remote_code=TRUST_REMOTE_CODE
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        _pipe = pipe
        _ready.set()
        print(f"[server] Model ready: {MODEL_PATH} (served as {SERVED_MODEL_NAME})", flush=True)
    except Exception as exc:
        _load_error = exc
        print(f"[server] Model load FAILED: {exc!r}", flush=True)


@app.on_event("startup")
def _startup():
    print(f"[server] Loading {MODEL_PATH} in background (first load downloads weights)...", flush=True)
    threading.Thread(target=_load_model, daemon=True).start()


def _check_auth(authorization):
    if not API_KEY or API_KEY == "not-needed":
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def _parse_size(size):
    try:
        w, h = size.lower().split("x")
        w, h = int(w), int(h)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid size '{size}', expected e.g. '1024x1024'")
    if w <= 0 or h <= 0 or w > MAX_DIM or h > MAX_DIM:
        raise HTTPException(status_code=400, detail=f"size out of range (1..{MAX_DIM} per side)")
    return w, h


def _supported_kwargs(pipe, candidate):
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (ValueError, TypeError):
        return candidate
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return candidate
    return {k: v for k, v in candidate.items() if k in params}


class ImageRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "url"  # "url" or "b64_json"
    num_inference_steps: int | None = None
    true_cfg_scale: float | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    seed: int | None = None


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Generator</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 system-ui, sans-serif; background: #0f1115; color: #e6e6e6; }
  .wrap { max-width: 900px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #8a93a3; margin: 0 0 20px; font-size: 13px; }
  .status { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; background: #2a2f3a; }
  .status.ready { background: #173a25; color: #5fd38b; }
  .status.loading { background: #3a3417; color: #e0c24a; }
  .status.error { background: #3a1a1a; color: #f08a8a; }
  textarea, input, select { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #2a2f3a;
    background: #161922; color: #e6e6e6; font: inherit; }
  textarea { resize: vertical; min-height: 78px; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
  .row > div { flex: 1 1 130px; }
  label { display: block; font-size: 12px; color: #8a93a3; margin: 0 0 4px; }
  button { margin-top: 16px; padding: 11px 18px; border: 0; border-radius: 8px; cursor: pointer;
    background: #3b6ef0; color: #fff; font-weight: 600; font-size: 15px; }
  button:disabled { opacity: .55; cursor: not-allowed; }
  .out { margin-top: 24px; }
  .out img { max-width: 100%; border-radius: 10px; border: 1px solid #2a2f3a; }
  .err { color: #f08a8a; white-space: pre-wrap; }
  .muted { color: #8a93a3; font-size: 13px; }
  a { color: #7aa2ff; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Image Generator <span id="status" class="status">checking…</span></h1>
  <p class="sub">Model: <code id="model">…</code> · OpenAI API at <code>/v1/images/generations</code></p>

  <label for="prompt">Prompt</label>
  <textarea id="prompt" placeholder="a red fox in deep snow, golden hour, photorealistic"></textarea>

  <div class="row">
    <div>
      <label for="size">Size</label>
      <select id="size">
        <option>1024x1024</option>
        <option>1280x768</option>
        <option>768x1280</option>
        <option>1536x1024</option>
        <option>1024x1536</option>
      </select>
    </div>
    <div>
      <label for="steps">Steps</label>
      <input id="steps" type="number" min="1" max="100" value="">
    </div>
    <div>
      <label for="seed">Seed (optional)</label>
      <input id="seed" type="number" placeholder="random">
    </div>
  </div>

  <button id="go">Generate</button>
  <div class="out" id="out"></div>
</div>

<script>
const $ = (id) => document.getElementById(id);

async function refreshStatus() {
  try {
    const r = await fetch("health");
    const j = await r.json();
    $("model").textContent = j.model || "?";
    const el = $("status");
    el.className = "status " + (j.status || "");
    el.textContent = j.status === "ready" ? "ready" : (j.status || "unknown");
    return j.status === "ready";
  } catch (e) {
    $("status").className = "status error";
    $("status").textContent = "offline";
    return false;
  }
}

async function generate() {
  const prompt = $("prompt").value.trim();
  if (!prompt) { $("prompt").focus(); return; }
  const body = { prompt, size: $("size").value, response_format: "b64_json", n: 1 };
  const steps = parseInt($("steps").value, 10);
  if (!isNaN(steps)) body.num_inference_steps = steps;
  const seed = parseInt($("seed").value, 10);
  if (!isNaN(seed)) body.seed = seed;

  $("go").disabled = true;
  const started = Date.now();
  $("out").innerHTML = '<p class="muted">Generating… (first run may take a few minutes while the model loads)</p>';
  try {
    const r = await fetch("v1/images/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const t = await r.text();
      $("out").innerHTML = '<p class="err">Error ' + r.status + ': ' + t + '</p>';
      return;
    }
    const j = await r.json();
    const secs = ((Date.now() - started) / 1000).toFixed(1);
    $("out").innerHTML = '<p class="muted">Done in ' + secs + 's</p>'
      + '<img src="data:image/png;base64,' + j.data[0].b64_json + '">';
  } catch (e) {
    $("out").innerHTML = '<p class="err">Request failed: ' + e + '</p>';
  } finally {
    $("go").disabled = false;
  }
}

$("go").addEventListener("click", generate);
$("prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") generate();
});
refreshStatus();
setInterval(refreshStatus, 5000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    if _load_error is not None:
        return {"status": "error", "model": SERVED_MODEL_NAME, "error": repr(_load_error)}
    return {"status": "ready" if _ready.is_set() else "loading", "model": SERVED_MODEL_NAME}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": SERVED_MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "local"}
        ],
    }


@app.get("/images/{name}")
def get_image(name: str):
    path = os.path.join(IMAGE_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="image/png")


@app.post("/v1/images/generations")
def generate(req: ImageRequest, request: Request, authorization: str | None = Header(default=None)):
    _check_auth(authorization)

    if _load_error is not None:
        raise HTTPException(status_code=500, detail=f"model failed to load: {_load_error!r}")
    if not _ready.wait(timeout=1800):
        raise HTTPException(status_code=503, detail="model still loading, retry shortly")

    width, height = _parse_size(req.size)
    n = max(1, min(int(req.n), 8))

    generator = None
    if req.seed is not None and torch.cuda.is_available():
        generator = torch.Generator(device="cuda").manual_seed(int(req.seed))

    candidate = {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt if req.negative_prompt is not None else DEFAULT_NEGATIVE,
        "width": width,
        "height": height,
        "num_inference_steps": req.num_inference_steps or DEFAULT_STEPS,
        "true_cfg_scale": req.true_cfg_scale if req.true_cfg_scale is not None else DEFAULT_TRUE_CFG,
        "guidance_scale": req.guidance_scale if req.guidance_scale is not None else DEFAULT_GUIDANCE,
        "num_images_per_prompt": n,
        "generator": generator,
    }
    kwargs = _supported_kwargs(_pipe, candidate)

    with _gpu_lock:
        try:
            result = _pipe(**kwargs)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"generation failed: {exc!r}")

    data = []
    created = int(time.time())
    base = str(request.base_url).rstrip("/")
    for img in result.images:
        if req.response_format == "b64_json":
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data.append({"b64_json": base64.b64encode(buf.getvalue()).decode("utf-8")})
        else:
            name = f"{uuid.uuid4().hex}.png"
            img.save(os.path.join(IMAGE_DIR, name), format="PNG")
            data.append({"url": f"{base}/images/{name}"})

    return {"created": created, "data": data}
PYEOF

echo "[INFO] Building image server container (first build pip-installs torch stack, can take several minutes)..."
docker build -t "$IMAGE_TAG" "$BUILD_DIR"

prompt_for_token

TIMER_CANCEL_FILE="/tmp/imgsrv-timer-cancel-$$"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
_HF_TOKEN_SAFE=$(printf '%s' "${HUGGING_FACE_HUB_TOKEN}" | sed 's/"/\\"/g')

# shellcheck disable=SC2086
eval docker run --rm -d $DOCKER_RUNTIME_ARGS --name "$CONTAINER_NAME" \
    -v "$HUGGINGFACE_MODEL_CACHE":/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=${_HF_TOKEN_SAFE}" \
    --env "CUDA_VISIBLE_DEVICES=${GPU_ID}" \
    --env "MODEL_PATH=${MODEL_PATH}" \
    --env "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}" \
    --env "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}" \
    --env "TRUE_CFG_SCALE=${TRUE_CFG_SCALE}" \
    --env "GUIDANCE_SCALE=${GUIDANCE_SCALE}" \
    --env "HF_HUB_ENABLE_HF_TRANSFER=1" \
    $PORT_BIND \
    --ipc=host \
    "$IMAGE_TAG"

start_timer "$CONTAINER_NAME" "$TIMER_CANCEL_FILE"

echo ""
if [ -n "$HOST_IP" ] && [ "$HOST_IP" != "0.0.0.0" ]; then
    echo "[INFO] API port ${PORT} bound on ${HOST_IP} (nc -zv ${HOST_IP} ${PORT} to test)."
    echo "[INFO] If the host IP is refused, use SSH tunnel from your laptop: ./tunnel_vllm.sh then http://localhost:${PORT}"
else
    echo "[INFO] API port ${PORT} bound on 0.0.0.0 (all interfaces)."
fi
echo "Image server is running in the background. Container: $CONTAINER_NAME."
echo "Serving diffusers model: $MODEL_PATH (as '$SERVED_MODEL_NAME')."
echo "First request can take 10-15 min (weight download + model load). /v1/models responds immediately;"
echo "image requests block until the model is ready. Watch 'docker logs -f $CONTAINER_NAME' for 'Model ready'."
echo "You can close this session; the server will keep running."
echo ""
echo "OpenAI-compatible API: http://<your_server>:$PORT/v1   (POST /v1/images/generations)"
echo "Health/readiness:      curl http://127.0.0.1:$PORT/health"
echo "To stop the server:    docker stop $CONTAINER_NAME"
echo "To view logs:          docker logs $CONTAINER_NAME   or  docker logs -f $CONTAINER_NAME"
echo ""

confirm_deployment

echo "Serving $SERVED_MODEL_NAME on port $PORT. HuggingFace cache is kept for reuse."
