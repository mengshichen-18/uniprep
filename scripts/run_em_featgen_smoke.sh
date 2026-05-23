#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/ajoint_0318_v3c.py}"
GEN_SCRIPT="${GEN_SCRIPT:-${REPO_DIR}/scripts/generate_task_feature_functions_gpt5.py}"
SYMBOLIC_GEN_SCRIPT="${SYMBOLIC_GEN_SCRIPT:-${REPO_DIR}/scripts/generate_symbolic_spec_gpt5_v3.py}"
API_KEY_FILE="${API_KEY_FILE:-${REPO_DIR}/../0325_policy_pro/LIGHTNING_API_KEY.md}"
API_KEY_LABEL="${API_KEY_LABEL:-0428_KEY}"

DATASET="${DATASET:-magellan}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-featgen_smoke_em_${DATASET}}"
GEN_MODE="${GEN_MODE:-dryrun}" # dryrun | real
GEN_SPECS_PATH="${GEN_SPECS_PATH:-${REPO_DIR}/generated_feature_examples/entity_matching_generated_teacher3plus7_dryrun.json}"
FORCE_REGEN="${FORCE_REGEN:-0}"

TEACHER_FEATURE_POOL="${TEACHER_FEATURE_POOL:-row_emb_cosine,row_emb_l1_sim,row_value_jaccard,row_value_containment_max,row_token_jaccard,row_nonempty_ratio,row_numeric_ratio_sim,row_avg_len_ratio,row_serial_token_jaccard,row_serial_edit_similarity,row_numeric_value_overlap,row_serial_char3_jaccard,row_serial_char4_jaccard,row_token_idf_jaccard,row_numeric_rel_diff_sim}"
GENERATOR_EXEMPLAR_FEATURES="${GENERATOR_EXEMPLAR_FEATURES:-${PRESERVED_FEATURES:-row_value_jaccard,row_serial_token_jaccard,row_numeric_rel_diff_sim}}"
EM_BASE_STATIC_FEATURES="${EM_BASE_STATIC_FEATURES:-row_serial_token_jaccard,row_serial_edit_similarity,row_numeric_value_overlap}"
EM_SYMBOLIC_SOURCE_FEATURES="${EM_SYMBOLIC_SOURCE_FEATURES:-${EM_PAIR_FEATURES:-${EM_BASE_STATIC_FEATURES}}}"
EM_DECODER_STATIC_FEATURES="${EM_DECODER_STATIC_FEATURES:-${EM_BASE_STATIC_FEATURES}}"
NUM_GENERATED_FEATURES="${NUM_GENERATED_FEATURES:-7}"
TARGET_TOTAL_POOL_SIZE="${TARGET_TOTAL_POOL_SIZE:-10}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Entity matching on heterogeneous table rows that may describe the same real-world entity despite formatting drift, partial missingness, and reordered cells.}"
DATA_DESCRIPTION="${DATA_DESCRIPTION:-Rows expose normalized value sets, token sets, serialized text, simple numeric summaries, header-aware structures, and row embeddings. Generated atoms should be smooth, interpretable, and robust across datasets.}"
SELECTION_NOTES="${SELECTION_NOTES:-The preserved exemplar atoms are manually selected by us as trusted reference features. Do not regenerate them; generate complementary atoms that help complete a compact EM pool.}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_DIR}/outputs/featgen_artifacts}"
ENABLE_SYMBOLIC="${ENABLE_SYMBOLIC:-0}"
SYMBOLIC_GEN_MODE="${SYMBOLIC_GEN_MODE:-dryrun}" # dryrun | real
SYMBOLIC_SPEC_PATH="${SYMBOLIC_SPEC_PATH:-${ARTIFACT_DIR}/em_symbolic_teacher3plus7_c4.json}"
FORCE_SYMBOLIC_REGEN="${FORCE_SYMBOLIC_REGEN:-0}"
NUM_SYMBOLIC_CHANNELS="${NUM_SYMBOLIC_CHANNELS:-auto}"
SYMBOLIC_MIN_CHANNELS="${SYMBOLIC_MIN_CHANNELS:-auto}"
SYMBOLIC_REPR="${SYMBOLIC_REPR:-concat}"
SYMBOLIC_NORMALIZE="${SYMBOLIC_NORMALIZE:-none}"
SYMBOLIC_TILE_REPEAT="${SYMBOLIC_TILE_REPEAT:-1}"
SYMBOLIC_TASK_HINT_STRONG_ATOMS_SPLIT="${SYMBOLIC_TASK_HINT_STRONG_ATOMS_SPLIT:-train}"
SYMBOLIC_TASK_HINT_STRONG_ATOMS_TOPK="${SYMBOLIC_TASK_HINT_STRONG_ATOMS_TOPK:-8}"
SYMBOLIC_MODEL="${SYMBOLIC_MODEL:-gpt-5-mini}"
SYMBOLIC_REASONING_EFFORT="${SYMBOLIC_REASONING_EFFORT:-low}"
SYMBOLIC_MAX_COMPLETION_TOKENS="${SYMBOLIC_MAX_COMPLETION_TOKENS:-3200}"

EPOCHS="${EPOCHS:-1}"
PATIENCE="${PATIENCE:-1}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_NEIGHBORS="${NUM_NEIGHBORS:-10,5}"
GNN_LAYERS="${GNN_LAYERS:-2}"
DEVICE="${DEVICE:-cuda:0}"
DEBUG_MAX_TRAIN_EDGES="${DEBUG_MAX_TRAIN_EDGES:-1024}"
DEBUG_MAX_VAL_EDGES="${DEBUG_MAX_VAL_EDGES:-512}"
DEBUG_MAX_TEST_EDGES="${DEBUG_MAX_TEST_EDGES:-512}"

GRAPH_DIR="${GRAPH_DIR:-}"
TABLE_ROOT="${TABLE_ROOT:-}"

if ! command -v "${PYTHON_BIN}" &>/dev/null; then
  echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -z "${GRAPH_DIR}" ]]; then
  echo "[ERROR] Set GRAPH_DIR to the graph data directory (e.g. /path/to/${DATASET}_040303_no_token)" >&2
  exit 1
fi
if [[ -z "${TABLE_ROOT}" ]]; then
  echo "[ERROR] Set TABLE_ROOT to the datalake directory (e.g. /path/to/${DATASET}_040303/datalake_plus)" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}"

if [[ "${FORCE_REGEN}" == "1" || ! -f "${GEN_SPECS_PATH}" ]]; then
  echo "[INFO] generating feature bundle at ${GEN_SPECS_PATH}"
  gen_args=(
    --task entity_matching
    --output "${GEN_SPECS_PATH}"
    --feature-request "Starting from three human-selected exemplar atoms plus the task and data descriptions, generate complementary EM atom features that help complete a compact 10-feature pool."
    --num-features "${NUM_GENERATED_FEATURES}"
    --target-total-pool-size "${TARGET_TOTAL_POOL_SIZE}"
    --teacher-feature-pool "${TEACHER_FEATURE_POOL}"
    --preserved-features "${GENERATOR_EXEMPLAR_FEATURES}"
    --task-description "${TASK_DESCRIPTION}"
    --data-description "${DATA_DESCRIPTION}"
    --selection-notes "${SELECTION_NOTES}"
    --dump-prompt "${ARTIFACT_DIR}/smoke_prompt.json"
    --dump-validation "${ARTIFACT_DIR}/smoke_validation.json"
  )
  if [[ "${GEN_MODE}" == "dryrun" ]]; then
    gen_args+=(--dry-run 1 --dump-response "${ARTIFACT_DIR}/smoke_response.json")
  else
    gen_args+=(--api-key-file "${API_KEY_FILE}" --api-key-label "${API_KEY_LABEL}" --dump-response "${ARTIFACT_DIR}/smoke_response.txt")
  fi
  "${PYTHON_BIN}" "${GEN_SCRIPT}" "${gen_args[@]}"
fi

SYMBOLIC_FEATURE_POOL="$("${PYTHON_BIN}" - "${GEN_SPECS_PATH}" "${EM_SYMBOLIC_SOURCE_FEATURES}" <<'PY'
import json
import sys
from pathlib import Path

spec_path = Path(sys.argv[1])
base = [token.strip() for token in sys.argv[2].split(",") if token.strip()]
payload = json.loads(spec_path.read_text(encoding="utf-8"))
features = payload.get("features", [])
names = [str(item.get("feature_name", "")).strip() for item in features if isinstance(item, dict)]
ordered = []
seen = set()
for name in base + names:
    if name and name not in seen:
        seen.add(name)
        ordered.append(name)
print(",".join(ordered))
PY
)"

if [[ "${ENABLE_SYMBOLIC}" == "1" ]]; then
  if [[ "${FORCE_SYMBOLIC_REGEN}" == "1" || ! -f "${SYMBOLIC_SPEC_PATH}" ]]; then
    echo "[INFO] generating symbolic spec at ${SYMBOLIC_SPEC_PATH}"
    sym_args=(
      --task entity_matching
      --feature-pool "${SYMBOLIC_FEATURE_POOL}"
      --spec-version v2
      --min-num-channels "${SYMBOLIC_MIN_CHANNELS}"
      --model "${SYMBOLIC_MODEL}"
      --reasoning-effort "${SYMBOLIC_REASONING_EFFORT}"
      --max-completion-tokens "${SYMBOLIC_MAX_COMPLETION_TOKENS}"
      --task-hint-strong-atoms-split "${SYMBOLIC_TASK_HINT_STRONG_ATOMS_SPLIT}"
      --task-hint-strong-atoms-topk "${SYMBOLIC_TASK_HINT_STRONG_ATOMS_TOPK}"
      --feature-cards-file "${REPO_DIR}/symbolic_feature_cards.json"
      --output "${SYMBOLIC_SPEC_PATH}"
      --summary-output "${ARTIFACT_DIR}/smoke_symbolic_summary.json"
      --dump-prompt "${ARTIFACT_DIR}/smoke_symbolic_prompt.json"
    )
    if [[ -n "${NUM_SYMBOLIC_CHANNELS}" && "${NUM_SYMBOLIC_CHANNELS}" != "auto" ]]; then
      sym_args+=(--num-channels "${NUM_SYMBOLIC_CHANNELS}")
    fi
    if [[ "${SYMBOLIC_GEN_MODE}" == "dryrun" ]]; then
      sym_args+=(--dry-run 1)
    else
      sym_args+=(--api-key-file "${API_KEY_FILE}" --api-key-label "${API_KEY_LABEL}")
    fi
    "${PYTHON_BIN}" "${SYMBOLIC_GEN_SCRIPT}" "${sym_args[@]}"
  fi
fi

echo "[INFO] dataset=${DATASET}"
echo "[INFO] gpu=${GPU_ID}"
echo "[INFO] generated_specs=${GEN_SPECS_PATH}"
echo "[INFO] generator_exemplars=${GENERATOR_EXEMPLAR_FEATURES}"
echo "[INFO] em_symbolic_source_features=${EM_SYMBOLIC_SOURCE_FEATURES}"
echo "[INFO] em_decoder_static_features=${EM_DECODER_STATIC_FEATURES}"
echo "[INFO] symbolic_feature_pool=${SYMBOLIC_FEATURE_POOL}"
echo "[INFO] symbolic_enabled=${ENABLE_SYMBOLIC}"
if [[ "${ENABLE_SYMBOLIC}" == "1" ]]; then
  echo "[INFO] symbolic_spec=${SYMBOLIC_SPEC_PATH}"
fi
echo "[INFO] graph_dir=${GRAPH_DIR}"
echo "[INFO] table_root=${TABLE_ROOT}"

train_args=(
  --dataset "${DATASET}"
  --graph_data_dir "${GRAPH_DIR}"
  --em_table_root "${TABLE_ROOT}"
  --jts_table_root "${TABLE_ROOT}"
  --sm_table_root "${TABLE_ROOT}"
  --uts_table_root "${TABLE_ROOT}"
  --task_permutation 0
  --limit_tasks 1
  --epochs "${EPOCHS}"
  --early_stopping_patience "${PATIENCE}"
  --lr "${LR}"
  --batch_size "${BATCH_SIZE}"
  --hidden_dim "${HIDDEN_DIM}"
  --num_neighbors "${NUM_NEIGHBORS}"
  --gnn_layers "${GNN_LAYERS}"
  --gnn_type our
  --device "${DEVICE}"
  --num_workers 0
  --seed 0
  --drop_cell_edges 1
  --em_pair_feat_norm 1
  --em_pair_cache_mode off
  --em_symbolic_source_features "${EM_SYMBOLIC_SOURCE_FEATURES}"
  --em_decoder_static_features "${EM_DECODER_STATIC_FEATURES}"
  --em_generated_feature_specs_path "${GEN_SPECS_PATH}"
  --feature_wiring_mode decoupled
  --allow_empty_decoder_static_features 0
  --jts_pair_features jaccard_containment,value_profile,header_similarity
  --jts_decoder_static_features jaccard_containment,value_profile,header_similarity
  --sm_pair_features header_similarity,value_stats
  --sm_decoder_static_features value_stats
  --uts_pair_features column_overlap,header_jaccard
  --uts_decoder_static_features column_overlap,header_jaccard
  --em_row_stats_mode required
  --debug_max_train_edges "${DEBUG_MAX_TRAIN_EDGES}"
  --debug_max_val_edges "${DEBUG_MAX_VAL_EDGES}"
  --debug_max_test_edges "${DEBUG_MAX_TEST_EDGES}"
  --run_tag "${RUN_TAG}"
)

if [[ "${ENABLE_SYMBOLIC}" == "1" ]]; then
  train_args+=(
    --online_symbolic_spec_path "${SYMBOLIC_SPEC_PATH}"
    --online_symbolic_repr "${SYMBOLIC_REPR}"
    --online_symbolic_normalize "${SYMBOLIC_NORMALIZE}"
    --online_symbolic_tile_repeat "${SYMBOLIC_TILE_REPEAT}"
  )
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  "${train_args[@]}" \
  "$@"
