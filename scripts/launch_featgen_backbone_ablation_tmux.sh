#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_LAUNCHER="${REPO_DIR}/scripts/launch_featgen_main_pool14_seed5_tmux.sh"

BACKBONES_CSV="${BACKBONES_CSV:-gcn,gat}"
SEEDS_CSV="${SEEDS_CSV:-0}"
TASKS_CSV="${TASKS_CSV:-entity_matching,joinable_table_search,schema_matching,union_table_search}"
DATASETS_CSV="${DATASETS_CSV:-magellan,santos_benchmark,wikidbs}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1}"
SESSION_PREFIX_BASE="${SESSION_PREFIX_BASE:-featgen_backbone_ablation_pool14}"

BUILD_CONTEXTS="${BUILD_CONTEXTS:-0}"
CONTEXTS_DIR="${CONTEXTS_DIR:-${REPO_DIR}/outputs/featgen_contexts}"
TOP_ATOMS="${TOP_ATOMS:-8}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${REPO_DIR}/outputs/tmux_logs/${SESSION_PREFIX_BASE}}"
ARTIFACT_ROOT_BASE="${ARTIFACT_ROOT_BASE:-${REPO_DIR}/outputs/${SESSION_PREFIX_BASE}}"

TARGET_TOTAL_POOL_SIZE="${TARGET_TOTAL_POOL_SIZE:-14}"
NUM_GENERATED_FEATURES="${NUM_GENERATED_FEATURES:-auto}"
NUM_SYMBOLIC_CHANNELS="${NUM_SYMBOLIC_CHANNELS:-auto}"
SYMBOLIC_MIN_CHANNELS="${SYMBOLIC_MIN_CHANNELS:-auto}"
ATOM_MODEL="${ATOM_MODEL:-gpt-5}"
ATOM_BASE_URL="${ATOM_BASE_URL:-}"
ATOM_MAX_REPAIR_ATTEMPTS="${ATOM_MAX_REPAIR_ATTEMPTS:-5}"
SYMBOLIC_MODEL="${SYMBOLIC_MODEL:-gpt-5}"
SYMBOLIC_BASE_URL="${SYMBOLIC_BASE_URL:-}"
SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO="${SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO:--1}"
SYMBOLIC_MAX_REPAIR_ATTEMPTS="${SYMBOLIC_MAX_REPAIR_ATTEMPTS:-1}"

EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-192}"
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

if [[ ! -x "${MAIN_LAUNCHER}" ]]; then
  echo "[ERROR] main launcher missing: ${MAIN_LAUNCHER}" >&2
  exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux not found in PATH" >&2
  exit 2
fi

mkdir -p "${TRAIN_LOG_ROOT}" "${ARTIFACT_ROOT_BASE}"

mapfile -t BACKBONES < <(parse_csv "${BACKBONES_CSV}")
if [[ "${#BACKBONES[@]}" -eq 0 ]]; then
  echo "[ERROR] no backbones provided" >&2
  exit 2
fi

build_contexts_flag="${BUILD_CONTEXTS}"
for backbone in "${BACKBONES[@]}"; do
  backbone_slug="$(slugify "${backbone}")"
  session_prefix="${SESSION_PREFIX_BASE}_${backbone_slug}"
  train_log_dir="${TRAIN_LOG_ROOT}/${backbone_slug}"
  artifact_root="${ARTIFACT_ROOT_BASE}/${backbone_slug}"

  echo "[INFO] launching backbone=${backbone} prefix=${session_prefix}"
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
  TARGET_TOTAL_POOL_SIZE="${TARGET_TOTAL_POOL_SIZE}" \
  NUM_GENERATED_FEATURES="${NUM_GENERATED_FEATURES}" \
  NUM_SYMBOLIC_CHANNELS="${NUM_SYMBOLIC_CHANNELS}" \
  SYMBOLIC_MIN_CHANNELS="${SYMBOLIC_MIN_CHANNELS}" \
  ATOM_MODEL="${ATOM_MODEL}" \
  ATOM_BASE_URL="${ATOM_BASE_URL}" \
  ATOM_MAX_REPAIR_ATTEMPTS="${ATOM_MAX_REPAIR_ATTEMPTS}" \
  SYMBOLIC_MODEL="${SYMBOLIC_MODEL}" \
  SYMBOLIC_BASE_URL="${SYMBOLIC_BASE_URL}" \
  SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO="${SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO}" \
  SYMBOLIC_MAX_REPAIR_ATTEMPTS="${SYMBOLIC_MAX_REPAIR_ATTEMPTS}" \
  EPOCHS="${EPOCHS}" \
  PATIENCE="${PATIENCE}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GNN_TYPE="${backbone}" \
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

echo "[DONE] launched backbone ablation with prefix_base=${SESSION_PREFIX_BASE}"
