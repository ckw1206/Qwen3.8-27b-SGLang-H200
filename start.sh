#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B on SGLang (H200 141GB, 2 GPUs by default, 1 GPU supported)
#
# Recipe from the SGLang cookbook (H200 cell):
#   https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
#
# Notes:
#   - Dense 27B hybrid GDN (Gated DeltaNet + gated attention) VLM; BF16
#     weights ~55GB, so one H200 already fits. 2 GPUs run TP2 for more KV
#     cache and lower per-token latency; 1 GPU is the cookbook's own cell.
#   - Cookbook H200 flags: fp8_e4m3 KV, flashinfer attention, 32K prefill
#     chunks, extra_buffer radix strategy (low-latency tier), fp32 GDN state.
#   - --reasoning-parser qwen3 splits thinking into reasoning_content;
#     --tool-call-parser qwen3_coder matches the chat template's
#     <tool_call><function=...> format (hermes would never parse it).
#   - MTP=1 turns on speculative decoding with the in-checkpoint MTP head
#     (EAGLE 3/1/4, no extra download). With spec on, SGLang caps
#     --max-running-requests at 48 unless EXTRA_ARGS sets it.
#   - Context is 262,144 natively; YaRN to 1M is an EXTRA_ARGS override
#     (see README).

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

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.21}"
GPU_ID="${GPU_ID:-0,1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
# Free-form extra SGLang flags, appended LAST (argparse last-wins):
#   EXTRA_ARGS="--context-length 131072" ./start.sh
read -ra EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

TP_SIZE="$(tr ',' '\n' <<< "${GPU_ID}" | grep -c '[0-9]' || true)"
case "${TP_SIZE}" in
  2|1) ;;
  *) echo "GPU_ID='${GPU_ID}' gives ${TP_SIZE} GPUs; Qwen3.8-27B setup supports 1 or 2"; exit 1 ;;
esac

MTP="${MTP:-0}"
case "${MTP}" in
  0) SPEC_ARGS=() ;;
  1) SPEC_ARGS=(--speculative-algorithm EAGLE
       --speculative-num-steps 3
       --speculative-eagle-topk 1
       --speculative-num-draft-tokens 4) ;;
  *) echo "MTP must be 0 or 1, got '${MTP}'"; exit 1 ;;
esac

CONTAINER_NAME="qwen3.8-27b-sglang"
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
echo "GPUs: ${GPU_ID} (TP${TP_SIZE})"
echo "Spec decode: $( (( MTP )) && echo "MTP (EAGLE 3/1/4)" || echo off)"
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
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype float32 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
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

# First boot downloads ~55GB of weights.
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
