#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TASKS_CSV="${TASKS_CSV:-entity_matching,joinable_table_search,schema_matching,union_table_search}"
DATASETS_CSV="${DATASETS_CSV:-magellan,santos_benchmark,wikidbs}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0}"
SESSION_PREFIX="${SESSION_PREFIX:-featgen0428}"
BUILD_CONTEXTS="${BUILD_CONTEXTS:-1}"
CONTEXTS_DIR="${CONTEXTS_DIR:-${REPO_DIR}/outputs/featgen_contexts}"
TOP_ATOMS="${TOP_ATOMS:-8}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${REPO_DIR}/outputs/tmux_logs/featgen_grid}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_DIR}/outputs/featgen_pipeline}"

ATOM_GEN_MODE="${ATOM_GEN_MODE:-real}"
SYMBOLIC_GEN_MODE="${SYMBOLIC_GEN_MODE:-real}"
NUM_GENERATED_FEATURES="${NUM_GENERATED_FEATURES:-auto}"
TARGET_TOTAL_POOL_SIZE="${TARGET_TOTAL_POOL_SIZE:-12}"
NUM_SYMBOLIC_CHANNELS="${NUM_SYMBOLIC_CHANNELS:-auto}"
SYMBOLIC_MIN_CHANNELS="${SYMBOLIC_MIN_CHANNELS:-auto}"
ATOM_MODEL="${ATOM_MODEL:-gpt-5}"
ATOM_MAX_REPAIR_ATTEMPTS="${ATOM_MAX_REPAIR_ATTEMPTS:-2}"
SYMBOLIC_MODEL="${SYMBOLIC_MODEL:-gpt-5}"
EPOCHS="${EPOCHS:-1}"
PATIENCE="${PATIENCE:-1}"
BATCH_SIZE="${BATCH_SIZE:-192}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEBUG_MAX_TRAIN_EDGES="${DEBUG_MAX_TRAIN_EDGES:-128}"
DEBUG_MAX_VAL_EDGES="${DEBUG_MAX_VAL_EDGES:-64}"
DEBUG_MAX_TEST_EDGES="${DEBUG_MAX_TEST_EDGES:-64}"
EM_PAIR_CACHE_MODE="${EM_PAIR_CACHE_MODE:-readwrite}"
EM_PAIR_CACHE_ROOT_BASE="${EM_PAIR_CACHE_ROOT_BASE:-${REPO_DIR}/outputs/em_pair_cache}"
EM_ROW_STATS_MODE="${EM_ROW_STATS_MODE:-full}"
FORCE_ATOM_REGEN="${FORCE_ATOM_REGEN:-1}"
FORCE_SYMBOLIC_REGEN="${FORCE_SYMBOLIC_REGEN:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

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

short_name() {
  case "$1" in
    entity_matching) echo "em" ;;
    joinable_table_search) echo "jts" ;;
    schema_matching) echo "sm" ;;
    union_table_search) echo "uts" ;;
    *) echo "unk" ;;
  esac
}

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux not found in PATH" >&2
  exit 2
fi

mkdir -p "${TRAIN_LOG_DIR}"

if [[ "${BUILD_CONTEXTS}" == "1" ]]; then
  python "${REPO_DIR}/scripts/build_featgen_dataset_contexts.py" \
    --output-dir "${CONTEXTS_DIR}" \
    --datasets "${DATASETS_CSV}" \
    --tasks "${TASKS_CSV}" \
    --top-atoms "${TOP_ATOMS}"
fi

mapfile -t TASKS < <(parse_csv "${TASKS_CSV}")
mapfile -t DATASETS < <(parse_csv "${DATASETS_CSV}")
mapfile -t GPUS < <(parse_csv "${GPU_IDS_CSV}")

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  GPUS=("0")
fi

idx=0
for task in "${TASKS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    short="$(short_name "${task}")"
    session="${SESSION_PREFIX}_${short}_${dataset}"
    log_path="${TRAIN_LOG_DIR}/${session}.log"
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "[SKIP] tmux session already exists: ${session}"
      idx=$((idx + 1))
      continue
    fi

    cmd="cd '${REPO_DIR}/..' && \
TASK='${task}' DATASET='${dataset}' GPU_ID='${gpu}' \
CONTEXTS_DIR='${CONTEXTS_DIR}' \
ARTIFACT_DIR='${ARTIFACT_ROOT}/${task}/${dataset}' \
RUN_TAG='${session}' \
ATOM_GEN_MODE='${ATOM_GEN_MODE}' \
SYMBOLIC_GEN_MODE='${SYMBOLIC_GEN_MODE}' \
NUM_GENERATED_FEATURES='${NUM_GENERATED_FEATURES}' \
TARGET_TOTAL_POOL_SIZE='${TARGET_TOTAL_POOL_SIZE}' \
NUM_SYMBOLIC_CHANNELS='${NUM_SYMBOLIC_CHANNELS}' \
SYMBOLIC_MIN_CHANNELS='${SYMBOLIC_MIN_CHANNELS}' \
ATOM_MODEL='${ATOM_MODEL}' \
ATOM_MAX_REPAIR_ATTEMPTS='${ATOM_MAX_REPAIR_ATTEMPTS}' \
SYMBOLIC_MODEL='${SYMBOLIC_MODEL}' \
EPOCHS='${EPOCHS}' \
PATIENCE='${PATIENCE}' \
BATCH_SIZE='${BATCH_SIZE}' \
NUM_WORKERS='${NUM_WORKERS}' \
DEBUG_MAX_TRAIN_EDGES='${DEBUG_MAX_TRAIN_EDGES}' \
DEBUG_MAX_VAL_EDGES='${DEBUG_MAX_VAL_EDGES}' \
DEBUG_MAX_TEST_EDGES='${DEBUG_MAX_TEST_EDGES}' \
EM_PAIR_CACHE_MODE='${EM_PAIR_CACHE_MODE}' \
EM_PAIR_CACHE_ROOT='${EM_PAIR_CACHE_ROOT_BASE}/${dataset}' \
EM_ROW_STATS_MODE='${EM_ROW_STATS_MODE}' \
FORCE_ATOM_REGEN='${FORCE_ATOM_REGEN}' \
FORCE_SYMBOLIC_REGEN='${FORCE_SYMBOLIC_REGEN}' \
SKIP_TRAIN='${SKIP_TRAIN}' \
bash '${REPO_DIR}/scripts/run_task_featgen_pipeline.sh' 2>&1 | tee '${log_path}'"

    tmux new-session -d -s "${session}" "${cmd}"
    echo "[START] session=${session} gpu=${gpu} task=${task} dataset=${dataset} log=${log_path}"
    idx=$((idx + 1))
  done
done

echo "[DONE] launched tmux grid with prefix=${SESSION_PREFIX}"
