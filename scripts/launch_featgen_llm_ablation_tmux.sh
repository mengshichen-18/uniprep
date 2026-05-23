#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_LAUNCHER="${REPO_DIR}/scripts/launch_featgen_main_pool14_seed5_tmux.sh"

MODEL_NAMES_CSV="${MODEL_NAMES_CSV:-qwen3.6-plus,deepseek-v3.2}"
SEEDS_CSV="${SEEDS_CSV:-0}"
TASKS_CSV="${TASKS_CSV:-entity_matching,joinable_table_search,schema_matching,union_table_search}"
DATASETS_CSV="${DATASETS_CSV:-magellan,santos_benchmark,wikidbs}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1}"
SESSION_PREFIX_BASE="${SESSION_PREFIX_BASE:-featgen_llm_ablation_pool14}"

BUILD_CONTEXTS="${BUILD_CONTEXTS:-0}"
CONTEXTS_DIR="${CONTEXTS_DIR:-${REPO_DIR}/outputs/featgen_contexts}"
TOP_ATOMS="${TOP_ATOMS:-8}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${REPO_DIR}/outputs/tmux_logs/${SESSION_PREFIX_BASE}}"
ARTIFACT_ROOT_BASE="${ARTIFACT_ROOT_BASE:-${REPO_DIR}/outputs/${SESSION_PREFIX_BASE}}"

TARGET_TOTAL_POOL_SIZE="${TARGET_TOTAL_POOL_SIZE:-14}"
NUM_GENERATED_FEATURES="${NUM_GENERATED_FEATURES:-auto}"
NUM_SYMBOLIC_CHANNELS="${NUM_SYMBOLIC_CHANNELS:-auto}"
SYMBOLIC_MIN_CHANNELS="${SYMBOLIC_MIN_CHANNELS:-auto}"
ATOM_TIMEOUT_SEC="${ATOM_TIMEOUT_SEC:-120}"
SYMBOLIC_TIMEOUT_SEC="${SYMBOLIC_TIMEOUT_SEC:-120}"
ATOM_MAX_REPAIR_ATTEMPTS="${ATOM_MAX_REPAIR_ATTEMPTS:-5}"
SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO="${SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO:--1}"
SYMBOLIC_MAX_REPAIR_ATTEMPTS="${SYMBOLIC_MAX_REPAIR_ATTEMPTS:-1}"

QWEN36PLUS_BASE_URL="${QWEN36PLUS_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
DEEPSEEKV32_BASE_URL="${DEEPSEEKV32_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-192}"
GNN_TYPE="${GNN_TYPE:-our}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEBUG_MAX_TRAIN_EDGES="${DEBUG_MAX_TRAIN_EDGES:-0}"
DEBUG_MAX_VAL_EDGES="${DEBUG_MAX_VAL_EDGES:-0}"
DEBUG_MAX_TEST_EDGES="${DEBUG_MAX_TEST_EDGES:-0}"
EM_PAIR_CACHE_MODE="${EM_PAIR_CACHE_MODE:-readwrite}"
EM_PAIR_CACHE_ROOT_BASE="${EM_PAIR_CACHE_ROOT_BASE:-${REPO_DIR}/outputs/em_pair_cache}"
EM_ROW_STATS_MODE="${EM_ROW_STATS_MODE:-full}"
FORCE_ATOM_REGEN="${FORCE_ATOM_REGEN:-1}"
FORCE_SYMBOLIC_REGEN="${FORCE_SYMBOLIC_REGEN:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

VARIANT_MODE="${VARIANT_MODE:-sequential}"  # sequential | parallel

parse_csv() {
  local raw="$1"
  IFS=',' read -r -a items <<< "${raw}"
  for item in "${items[@]}"; do
    item="$(echo "${item}" | xargs)"
    if [[ -n "${item}" ]]; then
      echo "${item}"
    fi
  done
}

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9'
}

wait_for_variant_sessions() {
  local prefix="$1"
  local gpu
  mapfile -t _gpus < <(parse_csv "${GPU_IDS_CSV}")
  while true; do
    local alive=0
    for gpu in "${_gpus[@]}"; do
      if tmux has-session -t "${prefix}_gpu${gpu}" 2>/dev/null; then
        alive=1
        break
      fi
    done
    if [[ "${alive}" -eq 0 ]]; then
      break
    fi
    sleep 30
  done
}

model_base_url() {
  case "$1" in
    qwen3.6-plus) echo "${QWEN36PLUS_BASE_URL}" ;;
    deepseek-v3.2) echo "${DEEPSEEKV32_BASE_URL}" ;;
    *) echo "" ;;
  esac
}

if [[ ! -x "${MAIN_LAUNCHER}" ]]; then
  echo "[ERROR] main launcher missing: ${MAIN_LAUNCHER}" >&2
  exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux not found in PATH" >&2
  exit 2
fi

mkdir -p "${TRAIN_LOG_ROOT}" "${ARTIFACT_ROOT_BASE}"

mapfile -t MODELS < <(parse_csv "${MODEL_NAMES_CSV}")
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "[ERROR] no models provided" >&2
  exit 2
fi

build_contexts_flag="${BUILD_CONTEXTS}"
for model_name in "${MODELS[@]}"; do
  model_slug="$(slugify "${model_name}")"
  base_url="$(model_base_url "${model_name}")"
  session_prefix="${SESSION_PREFIX_BASE}_${model_slug}"
  train_log_dir="${TRAIN_LOG_ROOT}/${model_slug}"
  artifact_root="${ARTIFACT_ROOT_BASE}/${model_slug}"

  if [[ -z "${base_url}" ]]; then
    echo "[ERROR] unsupported model for llm ablation launcher: ${model_name}" >&2
    exit 2
  fi

  echo "[INFO] launching llm_model=${model_name} prefix=${session_prefix}"
  SESSION_PREFIX="${session_prefix}" \
  SEEDS_CSV="${SEEDS_CSV}" \
  TASKS_CSV="${TASKS_CSV}" \
  DATASETS_CSV="${DATASETS_CSV}" \
  GPU_IDS_CSV="${GPU_IDS_CSV}" \
  BUILD_CONTEXTS="${build_contexts_flag}" \
  CONTEXTS_DIR="${CONTEXTS_DIR}" \
  TOP_ATOMS="${TOP_ATOMS}" \
  TRAIN_LOG_DIR="${train_log_dir}" \
  ARTIFACT_ROOT="${artifact_root}" \
  ATOM_TIMEOUT_SEC="${ATOM_TIMEOUT_SEC}" \
  SYMBOLIC_TIMEOUT_SEC="${SYMBOLIC_TIMEOUT_SEC}" \
  TARGET_TOTAL_POOL_SIZE="${TARGET_TOTAL_POOL_SIZE}" \
  NUM_GENERATED_FEATURES="${NUM_GENERATED_FEATURES}" \
  NUM_SYMBOLIC_CHANNELS="${NUM_SYMBOLIC_CHANNELS}" \
  SYMBOLIC_MIN_CHANNELS="${SYMBOLIC_MIN_CHANNELS}" \
  ATOM_MODEL="${model_name}" \
  ATOM_BASE_URL="${base_url}" \
  ATOM_MAX_REPAIR_ATTEMPTS="${ATOM_MAX_REPAIR_ATTEMPTS}" \
  SYMBOLIC_MODEL="${model_name}" \
  SYMBOLIC_BASE_URL="${base_url}" \
  SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO="${SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO}" \
  SYMBOLIC_MAX_REPAIR_ATTEMPTS="${SYMBOLIC_MAX_REPAIR_ATTEMPTS}" \
  EPOCHS="${EPOCHS}" \
  PATIENCE="${PATIENCE}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GNN_TYPE="${GNN_TYPE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  DEBUG_MAX_TRAIN_EDGES="${DEBUG_MAX_TRAIN_EDGES}" \
  DEBUG_MAX_VAL_EDGES="${DEBUG_MAX_VAL_EDGES}" \
  DEBUG_MAX_TEST_EDGES="${DEBUG_MAX_TEST_EDGES}" \
  EM_PAIR_CACHE_MODE="${EM_PAIR_CACHE_MODE}" \
  EM_PAIR_CACHE_ROOT_BASE="${EM_PAIR_CACHE_ROOT_BASE}" \
  EM_ROW_STATS_MODE="${EM_ROW_STATS_MODE}" \
  FORCE_ATOM_REGEN="${FORCE_ATOM_REGEN}" \
  FORCE_SYMBOLIC_REGEN="${FORCE_SYMBOLIC_REGEN}" \
  SKIP_TRAIN="${SKIP_TRAIN}" \
  bash "${MAIN_LAUNCHER}"

  build_contexts_flag="0"
  if [[ "${VARIANT_MODE}" == "sequential" ]]; then
    wait_for_variant_sessions "${session_prefix}"
  fi
done

echo "[DONE] launched llm ablation with prefix_base=${SESSION_PREFIX_BASE}"
