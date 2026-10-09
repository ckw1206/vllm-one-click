#!/usr/bin/env bash
set -euo pipefail

# MiniMax-M2.7 on SGLang (8x H200 141GB)
#
# Recipe from the MiniMax SGLang deploy guide:
#   https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/docs/sglang_deploy_guide.md
#
# Notes:
#   - MoE model, ~220GB weights + ~240GB KV per 1M tokens. 8 GPUs run
#     TP8 + EP8 (guide's 8-GPU cell); 4 GPUs run plain TP4 (still fits
#     on H200: 4x141GB). Max context per sequence is 196K.
#   - Needs SGLang >= v0.5.4.post1 ("MiniMax-M2 model is not currently
#     supported" means the image is too old).
#   - --reasoning-parser minimax-append-think keeps <think>...</think>
#     inline in `content` (guide default; MiniMax wants the thinking sent
#     back in history for interleaved thinking). REASONING_PARSER=minimax
#     splits it into reasoning_content instead.
#   - Tool calling: --tool-call-parser minimax-m2; SGLang needs no
#     --enable-auto-tool-choice, just send `tools` in the request.
#   - DFLASH=1 turns on speculative decoding with NVIDIA's block-diffusion
#     draft (https://huggingface.co/nvidia/MiniMax-M2.7-DFlash, 5-layer
#     BF16 draft, block size 8 -> 8 verify tokens). Card reports accept
#     length ~3.0 on vLLM TP4/H100 only; SGLang on H200 is untested here.
#     Card marks it demo-only, under NVIDIA eval + MiniMax non-commercial
#     licenses, so it is off by default.

# Load optional .env overrides. Shell env wins; .env fills gaps; defaults last.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

MODEL_ID="${MODEL_ID:-MiniMaxAI/MiniMax-M2.7}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniMax-M2.7}"
IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.21}"
GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
REASONING_PARSER="${REASONING_PARSER:-minimax-append-think}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
# Free-form extra SGLang flags, appended LAST (argparse last-wins):
#   EXTRA_ARGS="--context-length 131072" ./start.sh
read -ra EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

TP_SIZE="$(tr ',' '\n' <<< "${GPU_ID}" | grep -c '[0-9]' || true)"
case "${TP_SIZE}" in
  8) PARALLEL_ARGS=(--tp-size 8 --ep-size 8) ;;
  4) PARALLEL_ARGS=(--tp-size 4) ;;
  *) echo "GPU_ID='${GPU_ID}' gives ${TP_SIZE} GPUs; MiniMax-M2.7 recipe supports 4 or 8"; exit 1 ;;
esac

DFLASH="${DFLASH:-0}"
DRAFT_MODEL="${DRAFT_MODEL:-nvidia/MiniMax-M2.7-DFlash}"
# SGLang (v0.5.21 and main as of 2026-10-09) never wired MiniMaxM2ForCausalLM
# for DFLASH; two gaps, grafted onto the class at container start:
#   1. no set_dflash_layers_to_capture -> dies at init. The llama/qwen3_moe
#      DFlash hook is the EAGLE3 hook with explicit layer ids; reuse it.
#   2. get_input_embeddings(input_ids) returns a tensor, but DFlash calls it
#      with no args for the embedding module (like llama) -> dies on the first
#      request. No args now returns embed_tokens; with input_ids, unchanged.
# ponytail: patches the image's source in place; drop once upstream adds both.
DFLASH_PATCH='
def _set_dflash_layers_to_capture(self, layer_ids):
    if layer_ids is None:
        raise ValueError("DFLASH requires explicit layer_ids for aux hidden capture.")
    self.set_eagle3_layers_to_capture(layer_ids)
MiniMaxM2ForCausalLM.set_dflash_layers_to_capture = _set_dflash_layers_to_capture
_orig_get_input_embeddings = MiniMaxM2ForCausalLM.get_input_embeddings
def _get_input_embeddings(self, input_ids=None):
    if input_ids is None:
        return self.model.embed_tokens
    return _orig_get_input_embeddings(self, input_ids)
MiniMaxM2ForCausalLM.get_input_embeddings = _get_input_embeddings'
LAUNCH=(python3 -m sglang.launch_server)
case "${DFLASH}" in
  0) SPEC_ARGS=() ;;
  # Target config says 204800 but the draft tops out at 196608 (YaRN x48
  # over 4096); SGLang refuses a draft shorter than the target, so cap
  # the target to the draft. 196608 is also the guide's per-sequence max.
  1) SPEC_ARGS=(--speculative-algorithm DFLASH
       --speculative-draft-model-path "${DRAFT_MODEL}"
       --speculative-num-draft-tokens 8
       --context-length 196608)
     LAUNCH=(bash -c 'f=$(python3 -c "import importlib.util as u; print(u.find_spec(\"sglang.srt.models.minimax_m2\").origin)") && printf "%s\n" "${DFLASH_PATCH}" >> "${f}" && exec "$@"' bash "${LAUNCH[@]}") ;;
  *) echo "DFLASH must be 0 or 1, got '${DFLASH}'"; exit 1 ;;
esac

CONTAINER_NAME="minimax-m2.7-sglang"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }

mkdir -p "${HF_CACHE}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting SGLang container for ${MODEL_ID}"
echo "GPUs: ${GPU_ID} (${PARALLEL_ARGS[*]})"
echo "Reasoning parser: ${REASONING_PARSER}"
echo "Spec decode: $( (( DFLASH )) && echo "DFlash (${DRAFT_MODEL})" || echo off)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

echo "[$(date -Is)] launching SGLang container" > "${LOG_FILE}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --cap-add SYS_NICE \
  --gpus "\"device=${GPU_ID}\"" \
  --shm-size 32g \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e DFLASH_PATCH="${DFLASH_PATCH}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${IMAGE}" \
  "${LAUNCH[@]}" \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  "${PARALLEL_ARGS[@]}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --tool-call-parser minimax-m2 \
  --reasoning-parser "${REASONING_PARSER}" \
  "${SPEC_ARGS[@]}" \
  --enable-metrics \
  --host "${HOST}" \
  --port "${PORT}" \
  "${EXTRA_ARGS_ARR[@]}" \
  >/dev/null

docker inspect -f '{{.Id}}' "${CONTAINER_NAME}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} ($(cat "${PID_FILE}"))"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" &
log_follow_pid=$!

# First boot downloads ~220GB of weights; that alone can take a while.
echo "Waiting for HTTP readiness at ${READY_URL}"
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "SGLang is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Stop with ./stop.sh; logs: docker logs -f ${CONTAINER_NAME}"
