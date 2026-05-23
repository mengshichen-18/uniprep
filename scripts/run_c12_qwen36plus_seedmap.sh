#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GEN_SCRIPT="${REPO_DIR}/scripts/generate_symbolic_spec_gpt5_v3.py"
SEEDMAP_SCRIPT="${REPO_DIR}/scripts/run_seedmap_c4c8c12_3d4t.sh"

API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
MODEL="${MODEL:-qwen3.6-plus}"
PARALLEL_GEN="${PARALLEL_GEN:-2}"

FEATURE_POOL_DIR="${FEATURE_POOL_DIR:-${REPO_DIR}/symbolic_specs/batches/v3_tasklevel_nocontext_trainhint_gpt5_20260417_180446/feature_pools}"
BATCH_PARENT="${BATCH_PARENT:-${REPO_DIR}/symbolic_specs/batches}"

DATASETS_CSV="${DATASETS_CSV:-magellan,santos_benchmark,wikidbs}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
TASKS_CSV="${TASKS_CSV:-em,jts,sm,uts}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1}"

EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-192}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_NEIGHBORS="${NUM_NEIGHBORS:-10,5}"
GNN_LAYERS="${GNN_LAYERS:-2}"
GNN_TYPE="${GNN_TYPE:-our}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DROP_CELL_EDGES="${DROP_CELL_EDGES:-1}"
SYMBOLIC_NORMALIZE="${SYMBOLIC_NORMALIZE:-zscore}"
SYMBOLIC_TILE_REPEAT="${SYMBOLIC_TILE_REPEAT:-1}"
STATIC_PRESET="${STATIC_PRESET:-full}"

EM_DECODER_GROUPS="${EM_DECODER_GROUPS:-serial_value_alignment}"
SM_DECODER_GROUPS="${SM_DECODER_GROUPS:-value_stats}"
JTS_DECODER_GROUPS="${JTS_DECODER_GROUPS:-}"
UTS_DECODER_GROUPS="${UTS_DECODER_GROUPS:-}"

usage() {
  cat <<EOF
Usage:
  OPENAI_API_KEY=... bash scripts/run_c12_qwen36plus_seedmap.sh [options]

Options:
  --api-key <key>
  --base-url <url>
  --model <name>                 default: qwen3.6-plus
  --parallel-gen <n>             default: 2
  --datasets <csv>               default: ${DATASETS_CSV}
  --seeds <csv>                  default: ${SEEDS_CSV}
  --tasks <csv>                  default: ${TASKS_CSV}
  --gpu-ids <csv>                default: ${GPU_IDS_CSV}
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-key) API_KEY="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --parallel-gen) PARALLEL_GEN="$2"; shift 2 ;;
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --tasks) TASKS_CSV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${API_KEY}" ]]; then
  echo "[ERROR] missing api key"
  exit 1
fi
if [[ ! -d "${FEATURE_POOL_DIR}" ]]; then
  echo "[ERROR] feature pool dir not found: ${FEATURE_POOL_DIR}"
  exit 1
fi
if [[ ! -f "${GEN_SCRIPT}" ]]; then
  echo "[ERROR] gen script missing: ${GEN_SCRIPT}"
  exit 1
fi

task_name_of() {
  case "$1" in
    em) echo "entity_matching" ;;
    jts) echo "joinable_table_search" ;;
    sm) echo "schema_matching" ;;
    uts) echo "union_table_search" ;;
    *) return 1 ;;
  esac
}

TS="$(date +%Y%m%d_%H%M%S)"
BATCH_ROOT="${BATCH_PARENT}/v3_tasklevel_nocontext_trainhint_qwen36plus_c12_${TS}"
PROMPT_DIR="${BATCH_ROOT}/prompts"
LOG_DIR="${BATCH_ROOT}/logs"
mkdir -p "${PROMPT_DIR}" "${LOG_DIR}"
for tk in em jts sm uts; do
  tn="$(task_name_of "${tk}")"
  mkdir -p "${BATCH_ROOT}/c12/${tn}"
done

JOBS_FILE="${BATCH_ROOT}/generation_jobs.tsv"
: > "${JOBS_FILE}"
for tk in em jts sm uts; do
  tn="$(task_name_of "${tk}")"
  fp="${FEATURE_POOL_DIR}/${tn}.json"
  [[ -f "${fp}" ]] || { echo "[ERROR] missing feature pool: ${fp}"; exit 1; }
  for cand in 01 02 03 04 05; do
    echo -e "${tn}\t${fp}\t${cand}" >> "${JOBS_FILE}"
  done
done

echo "[INFO] batch_root=${BATCH_ROOT}"
echo "[INFO] model=${MODEL} base_url=${BASE_URL}"

while IFS=$'\t' read -r task_name fp cand; do
  (
    set -euo pipefail
    out="${BATCH_ROOT}/c12/${task_name}/cand_${cand}.json"
    prompt="${PROMPT_DIR}/c12_${task_name}_cand_${cand}_prompt.json"
    log="${LOG_DIR}/c12_${task_name}_cand_${cand}.log"
    "${PYTHON_BIN}" "${GEN_SCRIPT}" \
      --task "${task_name}" \
      --feature-pool-file "${fp}" \
      --spec-version v2 \
      --num-channels 12 \
      --output "${out}" \
      --dump-prompt "${prompt}" \
      --model "${MODEL}" \
      --base-url "${BASE_URL}" \
      --api-key "${API_KEY}" \
      --temperature 1.0 \
      --reasoning-effort low \
      --timeout-sec 300 \
      --max-completion-tokens 0 \
      --allow-dataset-context 0 \
      --disallow-group-tokens 1 \
      --enable-single-atom-hint 1 \
      --passthrough-ratio 0 \
      --max-repair-attempts 3 > "${log}" 2>&1
    echo "[OK] ${task_name} cand_${cand}"
  ) &
  while [[ "$(jobs -rp | wc -l)" -ge "${PARALLEL_GEN}" ]]; do
    sleep 0.5
  done
done < "${JOBS_FILE}"
wait

count="$(find "${BATCH_ROOT}/c12" -type f -name 'cand_*.json' | wc -l | xargs)"
if [[ "${count}" != "20" ]]; then
  echo "[ERROR] generation incomplete: ${count}/20 specs"
  exit 1
fi
echo "[INFO] generation done: 20/20"

SPEC_ROOT="${BATCH_ROOT}" \
DIMS_CSV="c12" \
DATASETS_CSV="${DATASETS_CSV}" \
SEEDS_CSV="${SEEDS_CSV}" \
TASKS_CSV="${TASKS_CSV}" \
GPU_IDS_CSV="${GPU_IDS_CSV}" \
EPOCHS="${EPOCHS}" \
PATIENCE="${PATIENCE}" \
BATCH_SIZE="${BATCH_SIZE}" \
LR="${LR}" \
HIDDEN_DIM="${HIDDEN_DIM}" \
NUM_NEIGHBORS="${NUM_NEIGHBORS}" \
GNN_LAYERS="${GNN_LAYERS}" \
GNN_TYPE="${GNN_TYPE}" \
NUM_WORKERS="${NUM_WORKERS}" \
DROP_CELL_EDGES="${DROP_CELL_EDGES}" \
SYMBOLIC_NORMALIZE="${SYMBOLIC_NORMALIZE}" \
SYMBOLIC_TILE_REPEAT="${SYMBOLIC_TILE_REPEAT}" \
STATIC_PRESET="${STATIC_PRESET}" \
EM_DECODER_GROUPS="${EM_DECODER_GROUPS}" \
JTS_DECODER_GROUPS="${JTS_DECODER_GROUPS}" \
SM_DECODER_GROUPS="${SM_DECODER_GROUPS}" \
UTS_DECODER_GROUPS="${UTS_DECODER_GROUPS}" \
RUN_TAG_PREFIX="seedmap_c12_qwen36plus" \
bash "${SEEDMAP_SCRIPT}"

echo "[DONE] c12 qwen3.6-plus pipeline done."
