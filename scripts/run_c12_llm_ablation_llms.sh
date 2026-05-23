#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GEN_SCRIPT="${REPO_DIR}/scripts/generate_symbolic_spec_gpt5_v3.py"
SEEDMAP_SCRIPT="${REPO_DIR}/scripts/run_seedmap_c4c8c12_3d4t.sh"

BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
PARALLEL_GEN="${PARALLEL_GEN:-4}"
RUN_EVAL="${RUN_EVAL:-1}"

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
ALLOW_EMPTY_DECODER_GROUPS="${ALLOW_EMPTY_DECODER_GROUPS:-1}"

EM_DECODER_GROUPS="${EM_DECODER_GROUPS:-serial_value_alignment}"
SM_DECODER_GROUPS="${SM_DECODER_GROUPS:-value_stats}"
JTS_DECODER_GROUPS="${JTS_DECODER_GROUPS:-}"
UTS_DECODER_GROUPS="${UTS_DECODER_GROUPS:-}"

FEATURE_POOL_DIR="${FEATURE_POOL_DIR:-${REPO_DIR}/symbolic_specs/batches/v3_tasklevel_nocontext_trainhint_gpt5_20260417_180446/feature_pools}"
BATCH_PARENT="${BATCH_PARENT:-${REPO_DIR}/symbolic_specs/batches}"

usage() {
  cat <<EOF
Usage:
  OPENAI_API_KEY=... bash scripts/run_c12_llm_ablation_llms.sh [options]

Options:
  --api-key <key>                 API key (or use OPENAI_API_KEY env)
  --base-url <url>                default: ${BASE_URL}
  --parallel-gen <n>              default: ${PARALLEL_GEN}
  --run-eval <0|1>                default: ${RUN_EVAL}
  --datasets <csv>                default: ${DATASETS_CSV}
  --seeds <csv>                   default: ${SEEDS_CSV}
  --tasks <csv>                   default: ${TASKS_CSV}
  --gpu-ids <csv>                 default: ${GPU_IDS_CSV}
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-key) API_KEY="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --parallel-gen) PARALLEL_GEN="$2"; shift 2 ;;
    --run-eval) RUN_EVAL="$2"; shift 2 ;;
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --tasks) TASKS_CSV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -x "${GEN_SCRIPT}" && ! -f "${GEN_SCRIPT}" ]]; then
  echo "[ERROR] generator script missing: ${GEN_SCRIPT}"
  exit 1
fi
if [[ ! -x "${SEEDMAP_SCRIPT}" ]]; then
  echo "[ERROR] seedmap script missing: ${SEEDMAP_SCRIPT}"
  exit 1
fi
if [[ ! -d "${FEATURE_POOL_DIR}" ]]; then
  echo "[ERROR] feature pool dir missing: ${FEATURE_POOL_DIR}"
  exit 1
fi
if [[ -z "${API_KEY}" ]]; then
  echo "[ERROR] missing API key. Set --api-key or OPENAI_API_KEY."
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${REPO_DIR}/outputs/symbolic_eval/c12_llm_ablation_${TS}"
mkdir -p "${RUN_ROOT}"

log() { echo "[$(date +'%F %T')] $*"; }

task_name_of() {
  case "$1" in
    em) echo "entity_matching" ;;
    jts) echo "joinable_table_search" ;;
    sm) echo "schema_matching" ;;
    uts) echo "union_table_search" ;;
    *) return 1 ;;
  esac
}

run_one_model() {
  local model_label="$1"
  local model_name="$2"
  local model_dir="${BATCH_PARENT}/v3_tasklevel_nocontext_trainhint_${model_label}_c12_${TS}"
  local gen_log_dir="${model_dir}/logs"
  local prompt_dir="${model_dir}/prompts"
  mkdir -p "${gen_log_dir}" "${prompt_dir}"

  for task_key in em jts sm uts; do
    local task_name
    task_name="$(task_name_of "${task_key}")"
    mkdir -p "${model_dir}/c12/${task_name}"
  done

  local jobs_file="${RUN_ROOT}/${model_label}_gen_jobs.tsv"
  : > "${jobs_file}"
  for task_key in em jts sm uts; do
    local task_name
    task_name="$(task_name_of "${task_key}")"
    local fp="${FEATURE_POOL_DIR}/${task_name}.json"
    if [[ ! -f "${fp}" ]]; then
      echo "[ERROR] missing feature pool: ${fp}"
      exit 1
    fi
    for cand in 01 02 03 04 05; do
      echo -e "${task_name}\t${fp}\t${cand}\t${model_name}" >> "${jobs_file}"
    done
  done

  log "[GEN] model=${model_name} label=${model_label} -> ${model_dir}"
  while IFS=$'\t' read -r task_name fp cand model; do
    (
      set -euo pipefail
      local_out="${model_dir}/c12/${task_name}/cand_${cand}.json"
      local_prompt="${prompt_dir}/c12_${task_name}_cand_${cand}_prompt.json"
      local_log="${gen_log_dir}/c12_${task_name}_cand_${cand}.log"
      "${PYTHON_BIN}" "${GEN_SCRIPT}" \
        --task "${task_name}" \
        --feature-pool-file "${fp}" \
        --spec-version v2 \
        --num-channels 12 \
        --output "${local_out}" \
        --dump-prompt "${local_prompt}" \
        --model "${model}" \
        --base-url "${BASE_URL}" \
        --api-key "${API_KEY}" \
        --temperature 1.0 \
        --reasoning-effort low \
        --max-completion-tokens 0 \
        --allow-dataset-context 0 \
        --disallow-group-tokens 1 \
        --enable-single-atom-hint 1 \
        --passthrough-ratio 0 \
        --max-repair-attempts 1 \
        > "${local_log}" 2>&1
    ) &
    while [[ "$(jobs -rp | wc -l)" -ge "${PARALLEL_GEN}" ]]; do
      sleep 0.5
    done
  done < "${jobs_file}"
  wait

  local count
  count="$(find "${model_dir}/c12" -type f -name 'cand_*.json' | wc -l | xargs)"
  if [[ "${count}" != "20" ]]; then
    echo "[ERROR] ${model_label} generation incomplete: got ${count}/20 specs"
    exit 1
  fi
  log "[GEN-DONE] ${model_label} specs=20"

  if [[ "${RUN_EVAL}" == "1" ]]; then
    log "[EVAL] ${model_label} seed-map eval start"
    SPEC_ROOT="${model_dir}" \
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
    RUN_TAG_PREFIX="seedmap_c12_${model_label}" \
    bash "${SEEDMAP_SCRIPT}" | tee "${RUN_ROOT}/eval_${model_label}.log"
  fi
}

detect_qwen_model() {
  local probe_task="joinable_table_search"
  local probe_fp="${FEATURE_POOL_DIR}/${probe_task}.json"
  local probe_out="${RUN_ROOT}/qwen_probe.json"
  local probe_log="${RUN_ROOT}/qwen_probe.log"
  for cand_model in qwen3.6-plus qwen-plus; do
    if timeout 120s "${PYTHON_BIN}" "${GEN_SCRIPT}" \
      --task "${probe_task}" \
      --feature-pool-file "${probe_fp}" \
      --spec-version v2 \
      --num-channels 12 \
      --output "${probe_out}" \
      --model "${cand_model}" \
      --base-url "${BASE_URL}" \
      --api-key "${API_KEY}" \
      --temperature 1.0 \
      --reasoning-effort low \
      --max-completion-tokens 0 \
      --allow-dataset-context 0 \
      --disallow-group-tokens 1 \
      --enable-single-atom-hint 1 \
      --passthrough-ratio 0 \
      --max-repair-attempts 1 > "${probe_log}" 2>&1; then
      echo "${cand_model}"
      return 0
    fi
  done
  echo ""
  return 1
}

QWEN_MODEL="$(detect_qwen_model || true)"
if [[ -z "${QWEN_MODEL}" ]]; then
  echo "[ERROR] Qwen model probe failed on base_url=${BASE_URL}. See ${RUN_ROOT}/qwen_probe.log"
  exit 1
fi
log "[PROBE] selected Qwen model: ${QWEN_MODEL}"

run_one_model "deepseekv32" "deepseek-v3.2"
run_one_model "qwen" "${QWEN_MODEL}"

cat > "${RUN_ROOT}/meta.md" <<EOF
# c12 LLM Ablation Run

- time: ${TS}
- base_url: ${BASE_URL}
- qwen_model: ${QWEN_MODEL}
- deepseek_model: deepseek-v3.2
- run_eval: ${RUN_EVAL}
- datasets: ${DATASETS_CSV}
- tasks: ${TASKS_CSV}
- seeds: ${SEEDS_CSV}
- gpu_ids: ${GPU_IDS_CSV}
- em_decoder_groups: ${EM_DECODER_GROUPS}
- jts_decoder_groups: ${JTS_DECODER_GROUPS}
- sm_decoder_groups: ${SM_DECODER_GROUPS}
- uts_decoder_groups: ${UTS_DECODER_GROUPS}
EOF

log "[DONE] run_root=${RUN_ROOT}"
