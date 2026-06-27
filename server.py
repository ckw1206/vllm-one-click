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
