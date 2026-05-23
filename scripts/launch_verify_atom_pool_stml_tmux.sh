#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEEDS_CSV="${SEEDS_CSV:-0}"
POOL_SIZES_CSV="${POOL_SIZES_CSV:-8,10,12,14,16}"
TASKS_CSV="${TASKS_CSV:-entity_matching,joinable_table_search,schema_matching,union_table_search}"
DATASETS_CSV="${DATASETS_CSV:-santos_benchmark,magellan}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1}"
SESSION_PREFIX="${SESSION_PREFIX:-verify_atom_pool_stml_20260430}"

BUILD_CONTEXTS="${BUILD_CONTEXTS:-0}"
CONTEXTS_DIR="${CONTEXTS_DIR:-${REPO_DIR}/outputs/featgen_contexts}"
TOP_ATOMS="${TOP_ATOMS:-8}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${REPO_DIR}/outputs/tmux_logs/${SESSION_PREFIX}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_DIR}/outputs/${SESSION_PREFIX}}"
QUEUE_DIR="${QUEUE_DIR:-${TRAIN_LOG_DIR}/queues}"

ATOM_GEN_MODE="${ATOM_GEN_MODE:-real}"
SYMBOLIC_GEN_MODE="${SYMBOLIC_GEN_MODE:-real}"
ATOM_TIMEOUT_SEC="${ATOM_TIMEOUT_SEC:-120}"
SYMBOLIC_TIMEOUT_SEC="${SYMBOLIC_TIMEOUT_SEC:-120}"
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

job_weight() {
  local task="$1"
  local dataset="$2"
  if [[ "${task}" == "entity_matching" && "${dataset}" == "magellan" ]]; then
    echo 9
  elif [[ "${task}" == "entity_matching" ]]; then
    echo 5
  elif [[ "${task}" == "schema_matching" ]]; then
    echo 3
  else
    echo 2
  fi
}

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux not found in PATH" >&2
  exit 2
fi

mkdir -p "${TRAIN_LOG_DIR}" "${QUEUE_DIR}" "${ARTIFACT_ROOT}"

if [[ "${BUILD_CONTEXTS}" == "1" ]]; then
  "${PYTHON_BIN:-python3}" "${REPO_DIR}/scripts/build_featgen_dataset_contexts.py" \
    --output-dir "${CONTEXTS_DIR}" \
    --datasets "${DATASETS_CSV}" \
    --tasks "${TASKS_CSV}" \
    --top-atoms "${TOP_ATOMS}"
fi

mapfile -t SEEDS < <(parse_csv "${SEEDS_CSV}")
mapfile -t POOLS < <(parse_csv "${POOL_SIZES_CSV}")
mapfile -t TASKS < <(parse_csv "${TASKS_CSV}")
mapfile -t DATASETS < <(parse_csv "${DATASETS_CSV}")
mapfile -t GPUS < <(parse_csv "${GPU_IDS_CSV}")

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[ERROR] no gpu ids provided" >&2
  exit 2
fi

declare -A QUEUE_SCRIPTS
declare -A TOTAL_GPU_LOADS

for gpu in "${GPUS[@]}"; do
  TOTAL_GPU_LOADS["${gpu}"]=0
  queue_script="${QUEUE_DIR}/${SESSION_PREFIX}_gpu${gpu}.sh"
  QUEUE_SCRIPTS["${gpu}"]="${queue_script}"
  cat > "${queue_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_DIR}/..'
EOF
  chmod +x "${queue_script}"
done

jobs_tmp="$(mktemp)"
trap 'rm -f "${jobs_tmp}"' EXIT

for pool_size in "${POOLS[@]}"; do
  declare -A POOL_GPU_LOADS=()
  for gpu in "${GPUS[@]}"; do
    POOL_GPU_LOADS["${gpu}"]=0
  done

  : > "${jobs_tmp}"
  for seed in "${SEEDS[@]}"; do
    for task in "${TASKS[@]}"; do
      for dataset in "${DATASETS[@]}"; do
        weight="$(job_weight "${task}" "${dataset}")"
        printf "%s\t%s\t%s\t%s\t%s\n" "${weight}" "${seed}" "${task}" "${dataset}" "${pool_size}" >> "${jobs_tmp}"
      done
    done
  done

  for gpu in "${GPUS[@]}"; do
    cat >> "${QUEUE_SCRIPTS[${gpu}]}" <<EOF
echo "[POOL-START] pool=${pool_size} gpu=${gpu}"
mkdir -p '${TRAIN_LOG_DIR}/pool${pool_size}'
EOF
  done

  while IFS=$'\t' read -r weight seed task dataset pool; do
    best_gpu=""
    best_load=""
    for gpu in "${GPUS[@]}"; do
      load="${POOL_GPU_LOADS[${gpu}]}"
      if [[ -z "${best_gpu}" || "${load}" -lt "${best_load}" ]]; then
        best_gpu="${gpu}"
        best_load="${load}"
      fi
    done

    short="$(short_name "${task}")"
    run_tag="${SESSION_PREFIX}_pool${pool}_s${seed}_${short}_${dataset}"
    job_log="${TRAIN_LOG_DIR}/pool${pool}/${run_tag}.log"
    artifact_dir="${ARTIFACT_ROOT}/pool${pool}/seed${seed}/${task}/${dataset}"
    queue_script="${QUEUE_SCRIPTS[${best_gpu}]}"

    cat >> "${queue_script}" <<EOF
echo "[START] pool=${pool} seed=${seed} task=${task} dataset=${dataset} gpu=${best_gpu} run_tag=${run_tag}"
TASK='${task}' DATASET='${dataset}' GPU_ID='${best_gpu}' SEED='${seed}' \
CONTEXTS_DIR='${CONTEXTS_DIR}' \
ARTIFACT_DIR='${artifact_dir}' \
RUN_TAG='${run_tag}' \
API_KEY_FILE='${API_KEY_FILE:-}' \
API_KEY_LABEL='${API_KEY_LABEL:-}' \
ATOM_GEN_MODE='${ATOM_GEN_MODE}' \
SYMBOLIC_GEN_MODE='${SYMBOLIC_GEN_MODE}' \
ATOM_TIMEOUT_SEC='${ATOM_TIMEOUT_SEC}' \
SYMBOLIC_TIMEOUT_SEC='${SYMBOLIC_TIMEOUT_SEC}' \
NUM_GENERATED_FEATURES='${NUM_GENERATED_FEATURES}' \
TARGET_TOTAL_POOL_SIZE='${pool}' \
NUM_SYMBOLIC_CHANNELS='${NUM_SYMBOLIC_CHANNELS}' \
SYMBOLIC_MIN_CHANNELS='${SYMBOLIC_MIN_CHANNELS}' \
ATOM_MODEL='${ATOM_MODEL}' \
ATOM_BASE_URL='${ATOM_BASE_URL}' \
ATOM_MAX_REPAIR_ATTEMPTS='${ATOM_MAX_REPAIR_ATTEMPTS}' \
SYMBOLIC_MODEL='${SYMBOLIC_MODEL}' \
SYMBOLIC_BASE_URL='${SYMBOLIC_BASE_URL}' \
SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO='${SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO}' \
SYMBOLIC_MAX_REPAIR_ATTEMPTS='${SYMBOLIC_MAX_REPAIR_ATTEMPTS}' \
EPOCHS='${EPOCHS}' \
PATIENCE='${PATIENCE}' \
HIDDEN_DIM='${HIDDEN_DIM:-}' \
NUM_NEIGHBORS='${NUM_NEIGHBORS:-}' \
GNN_LAYERS='${GNN_LAYERS:-}' \
BATCH_SIZE='${BATCH_SIZE}' \
GNN_TYPE='${GNN_TYPE}' \
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
bash '${REPO_DIR}/scripts/run_task_featgen_pipeline.sh' 2>&1 | tee '${job_log}'
echo "[DONE] pool=${pool} seed=${seed} task=${task} dataset=${dataset} gpu=${best_gpu} run_tag=${run_tag}"
EOF

    POOL_GPU_LOADS["${best_gpu}"]=$(( POOL_GPU_LOADS["${best_gpu}"] + weight ))
    TOTAL_GPU_LOADS["${best_gpu}"]=$(( TOTAL_GPU_LOADS["${best_gpu}"] + weight ))
  done < <(sort -r -n "${jobs_tmp}")

  for gpu in "${GPUS[@]}"; do
    cat >> "${QUEUE_SCRIPTS[${gpu}]}" <<EOF
echo "[POOL-DONE] pool=${pool_size} gpu=${gpu}"
EOF
  done
done

for gpu in "${GPUS[@]}"; do
  queue_script="${QUEUE_SCRIPTS[${gpu}]}"
  session="${SESSION_PREFIX}_gpu${gpu}"
  launcher_log="${TRAIN_LOG_DIR}/${session}.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[SKIP] tmux session already exists: ${session}"
    continue
  fi
  tmux new-session -d -s "${session}" "bash '${queue_script}' 2>&1 | tee '${launcher_log}'"
  echo "[START] session=${session} gpu=${gpu} queue=${queue_script} log=${launcher_log} assigned_weight=${TOTAL_GPU_LOADS[${gpu}]}"
done

echo "[DONE] launched atom-pool sweep with prefix=${SESSION_PREFIX}"
