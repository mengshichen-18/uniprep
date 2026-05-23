#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SESSION_PREFIX="${SESSION_PREFIX:-featgen_nosymbolic_ablation_resume1}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${REPO_DIR}/outputs/tmux_logs/${SESSION_PREFIX}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_DIR}/outputs/${SESSION_PREFIX}}"
QUEUE_DIR="${QUEUE_DIR:-${TRAIN_LOG_DIR}/queues}"
CONTEXTS_DIR="${CONTEXTS_DIR:-${REPO_DIR}/outputs/featgen_contexts}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

ATOM_GEN_MODE="${ATOM_GEN_MODE:-real}"
SYMBOLIC_GEN_MODE="${SYMBOLIC_GEN_MODE:-real}"
NUM_GENERATED_FEATURES="${NUM_GENERATED_FEATURES:-auto}"
TARGET_TOTAL_POOL_SIZE="${TARGET_TOTAL_POOL_SIZE:-12}"
NUM_SYMBOLIC_CHANNELS="${NUM_SYMBOLIC_CHANNELS:-auto}"
SYMBOLIC_MIN_CHANNELS="${SYMBOLIC_MIN_CHANNELS:-auto}"
ATOM_MODEL="${ATOM_MODEL:-gpt-5}"
ATOM_BASE_URL="${ATOM_BASE_URL:-}"
ATOM_MAX_REPAIR_ATTEMPTS="${ATOM_MAX_REPAIR_ATTEMPTS:-2}"
ATOM_ALLOW_EXTRA_FEATURES_TRUNCATE="${ATOM_ALLOW_EXTRA_FEATURES_TRUNCATE:-1}"
SYMBOLIC_MODEL="${SYMBOLIC_MODEL:-gpt-5}"
SYMBOLIC_BASE_URL="${SYMBOLIC_BASE_URL:-}"
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
FORCE_ATOM_REGEN="${FORCE_ATOM_REGEN:-0}"
FORCE_SYMBOLIC_REGEN="${FORCE_SYMBOLIC_REGEN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux not found in PATH" >&2
  exit 2
fi

mkdir -p "${TRAIN_LOG_DIR}" "${QUEUE_DIR}"

run_job_block() {
  local task="$1"
  local dataset="$2"
  local gpu="$3"
  local run_tag="$4"
  local job_log="$5"
  local artifact_dir="$6"
  cat <<EOF
echo "[START] task=${task} dataset=${dataset} gpu=${gpu} run_tag=${run_tag}"
TASK='${task}' DATASET='${dataset}' GPU_ID='${gpu}' SEED='0' \
CONTEXTS_DIR='${CONTEXTS_DIR}' \
ARTIFACT_DIR='${artifact_dir}' \
RUN_TAG='${run_tag}' \
ATOM_GEN_MODE='${ATOM_GEN_MODE}' \
SYMBOLIC_GEN_MODE='${SYMBOLIC_GEN_MODE}' \
NUM_GENERATED_FEATURES='${NUM_GENERATED_FEATURES}' \
TARGET_TOTAL_POOL_SIZE='${TARGET_TOTAL_POOL_SIZE}' \
NUM_SYMBOLIC_CHANNELS='${NUM_SYMBOLIC_CHANNELS}' \
SYMBOLIC_MIN_CHANNELS='${SYMBOLIC_MIN_CHANNELS}' \
ATOM_MODEL='${ATOM_MODEL}' \
ATOM_BASE_URL='${ATOM_BASE_URL}' \
ATOM_MAX_REPAIR_ATTEMPTS='${ATOM_MAX_REPAIR_ATTEMPTS}' \
ATOM_ALLOW_EXTRA_FEATURES_TRUNCATE='${ATOM_ALLOW_EXTRA_FEATURES_TRUNCATE}' \
SYMBOLIC_MODEL='${SYMBOLIC_MODEL}' \
SYMBOLIC_BASE_URL='${SYMBOLIC_BASE_URL}' \
EPOCHS='${EPOCHS}' \
PATIENCE='${PATIENCE}' \
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
ENABLE_SYMBOLIC='0' \
SKIP_TRAIN='${SKIP_TRAIN}' \
bash '${REPO_DIR}/scripts/run_task_featgen_pipeline.sh' 2>&1 | tee '${job_log}'
echo "[DONE] task=${task} dataset=${dataset} gpu=${gpu} run_tag=${run_tag}"
EOF
}

queue0="${QUEUE_DIR}/${SESSION_PREFIX}_gpu${GPU0}.sh"
queue1="${QUEUE_DIR}/${SESSION_PREFIX}_gpu${GPU1}.sh"

cat > "${queue0}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_DIR}/..'
$(run_job_block \
  "union_table_search" \
  "magellan" \
  "${GPU0}" \
  "${SESSION_PREFIX}_s0_uts_magellan" \
  "${TRAIN_LOG_DIR}/${SESSION_PREFIX}_s0_uts_magellan.log" \
  "${ARTIFACT_ROOT}/seed0/union_table_search/magellan")
EOF

cat > "${queue1}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '${REPO_DIR}/..'
$(run_job_block \
  "union_table_search" \
  "wikidbs" \
  "${GPU1}" \
  "${SESSION_PREFIX}_s0_uts_wikidbs" \
  "${TRAIN_LOG_DIR}/${SESSION_PREFIX}_s0_uts_wikidbs.log" \
  "${ARTIFACT_ROOT}/seed0/union_table_search/wikidbs")
$(run_job_block \
  "joinable_table_search" \
  "santos_benchmark" \
  "${GPU1}" \
  "${SESSION_PREFIX}_s0_jts_santos_benchmark" \
  "${TRAIN_LOG_DIR}/${SESSION_PREFIX}_s0_jts_santos_benchmark.log" \
  "${ARTIFACT_ROOT}/seed0/joinable_table_search/santos_benchmark")
EOF

chmod +x "${queue0}" "${queue1}"

for gpu in "${GPU0}" "${GPU1}"; do
  session="${SESSION_PREFIX}_gpu${gpu}"
  queue_script="${QUEUE_DIR}/${session}.sh"
  launcher_log="${TRAIN_LOG_DIR}/${session}.log"
  if [[ "${gpu}" == "${GPU0}" ]]; then
    queue_script="${queue0}"
  else
    queue_script="${queue1}"
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[SKIP] tmux session already exists: ${session}"
    continue
  fi
  tmux new-session -d -s "${session}" "bash '${queue_script}' 2>&1 | tee '${launcher_log}'"
  echo "[START] session=${session} gpu=${gpu} queue=${queue_script} log=${launcher_log}"
done

echo "[DONE] launched no-symbolic remaining-job resume with prefix=${SESSION_PREFIX}"
