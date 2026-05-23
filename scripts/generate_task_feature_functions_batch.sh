#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TASKS_CSV="${TASKS_CSV:-entity_matching,joinable_table_search,schema_matching,union_table_search}"
DATASETS_CSV="${DATASETS_CSV:-magellan,santos_benchmark,wikidbs}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/generated_feature_examples/batch_generated}"
DATASET_CONTEXT_DIR="${DATASET_CONTEXT_DIR:-}"
DRY_RUN="${DRY_RUN:-1}"
NUM_FEATURES="${NUM_FEATURES:-auto}"
TARGET_TOTAL_POOL_SIZE="${TARGET_TOTAL_POOL_SIZE:-12}"
MODEL_NAME="${MODEL_NAME:-gpt-5}"
API_KEY_FILE="${API_KEY_FILE:-${REPO_DIR}/../0325_policy_pro/LIGHTNING_API_KEY.md}"
API_KEY_LABEL="${API_KEY_LABEL:-}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-2200}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"

mkdir -p "${OUTPUT_DIR}"

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

resolve_context_path() {
  local task="$1"
  local dataset="$2"
  if [[ -z "${DATASET_CONTEXT_DIR}" ]]; then
    return 0
  fi
  local candidate_a="${DATASET_CONTEXT_DIR}/${task}__${dataset}.json"
  local candidate_b="${DATASET_CONTEXT_DIR}/${dataset}__${task}.json"
  if [[ -f "${candidate_a}" ]]; then
    echo "${candidate_a}"
    return 0
  fi
  if [[ -f "${candidate_b}" ]]; then
    echo "${candidate_b}"
    return 0
  fi
  return 0
}

resolve_num_features() {
  local task="$1"
  if [[ -n "${NUM_FEATURES}" && "${NUM_FEATURES}" != "auto" ]]; then
    echo "${NUM_FEATURES}"
    return 0
  fi
  PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" python - "${task}" "${TARGET_TOTAL_POOL_SIZE}" <<'PY'
import sys
from task_featgen_config import TASK_FEATGEN_CONFIGS

task = sys.argv[1]
target_total = int(sys.argv[2])
cfg = TASK_FEATGEN_CONFIGS[task]
exemplar_count = len(list(cfg.get("human_exemplar_atoms", [])))
print(max(0, target_total - exemplar_count))
PY
}

while IFS= read -r task; do
  [[ -n "${task}" ]] || continue
  while IFS= read -r dataset; do
    [[ -n "${dataset}" ]] || continue
    out_json="${OUTPUT_DIR}/${task}__${dataset}.json"
    prompt_json="${OUTPUT_DIR}/${task}__${dataset}.prompt.json"
    response_txt="${OUTPUT_DIR}/${task}__${dataset}.response.txt"
    validation_json="${OUTPUT_DIR}/${task}__${dataset}.validation.json"
    context_path="$(resolve_context_path "${task}" "${dataset}")"
    num_features="$(resolve_num_features "${task}")"

    cmd=(
      python "${REPO_DIR}/scripts/generate_task_feature_functions_gpt5.py"
      --task "${task}"
      --output "${out_json}"
      --num-features "${num_features}"
      --target-total-pool-size "${TARGET_TOTAL_POOL_SIZE}"
      --model "${MODEL_NAME}"
      --api-key-file "${API_KEY_FILE}"
      --api-key-label "${API_KEY_LABEL}"
      --max-completion-tokens "${MAX_COMPLETION_TOKENS}"
      --reasoning-effort "${REASONING_EFFORT}"
      --timeout-sec "${TIMEOUT_SEC}"
      --dump-prompt "${prompt_json}"
      --dump-response "${response_txt}"
      --dump-validation "${validation_json}"
      --dry-run "${DRY_RUN}"
    )
    if [[ -n "${context_path}" ]]; then
      cmd+=(--dataset-context-json "${context_path}")
    fi

    echo "[featgen-batch] task=${task} dataset=${dataset} dry_run=${DRY_RUN} num_features=${num_features} output=${out_json}"
    "${cmd[@]}"
  done < <(parse_csv "${DATASETS_CSV}")
done < <(parse_csv "${TASKS_CSV}")
