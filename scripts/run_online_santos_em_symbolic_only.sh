#!/usr/bin/env bash
set -euo pipefail

# Fixed online single-task verify run on santos_benchmark with:
# source static groups computed for all tasks, decoder static subset by task, plus symbolic concat.
#
# Examples:
#   bash scripts/run_online_santos_em_symbolic_only.sh \
#     --spec /abs/path/to/cand_06.json \
#     --gpu 0 \
#     --run-tag santos_em_cand06 \
#     --task em
#
#   bash scripts/run_online_santos_em_symbolic_only.sh \
#     --spec-dir /abs/path/to/entity_matching \
#     --cand 08 \
#     --task em
#
#   bash scripts/run_online_santos_em_symbolic_only.sh --task jts
#   bash scripts/run_online_santos_em_symbolic_only.sh --task sm
#   bash scripts/run_online_santos_em_symbolic_only.sh --task uts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/ajoint_0318_v3c.py}"

# Fixed dataset paths for santos 040303 no_token.
DATASET="santos_benchmark"
GRAPH_DATA_DIR="${GRAPH_DATA_DIR:-}"
TABLE_ROOT="${TABLE_ROOT:-}"
PAIR_CACHE_ROOT="${REPO_DIR}/outputs/em_pair_cache/santos_benchmark"

# Core training defaults (same family as prior reproducible runs).
GPU="0"
SEED="0"
EPOCHS="120"
PATIENCE="20"
LR="0.001"
BATCH_SIZE="192"
HIDDEN_DIM="256"
NUM_NEIGHBORS="10,5"
GNN_LAYERS="2"
GNN_TYPE="our"
NUM_WORKERS="0"
DROP_CELL_EDGES="1"

# Symbolic defaults.
SPEC_PATH=""
SPEC_DIR=""
CAND_ID=""
TASK_ALIAS="em"   # em | jts | sm | uts
TASK_NAME="entity_matching"
TASK_PERMUTATION="0"
DEFAULT_SPEC_ROOT="${REPO_DIR}/symbolic_specs/batches/v3_tasklevel_nocontext_trainhint_gpt5_20260417_180446/c12"
RUN_TAG=""
ONLINE_SYMBOLIC_REPR="concat"
ONLINE_SYMBOLIC_NORMALIZE="none"
ONLINE_SYMBOLIC_TILE_REPEAT="1"
EM_PAIR_FEAT_NORM="1"
EM_PAIR_CACHE_MODE="readwrite"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_online_santos_em_symbolic_only.sh [options]

Optional:
  --task <em|jts|sm|uts>        Default: em
  (No spec args)                Use built-in default cand (em->02, others->01)
    OR
  --spec <path>                 Absolute spec json path
    OR
  --spec-dir <dir> --cand <id>  Resolve as <dir>/cand_<id>.json
  --gpu <id>                    Default: 0
  --seed <int>                  Default: 0
  --epochs <int>                Default: 120
  --patience <int>              Default: 20
  --lr <float>                  Default: 0.001
  --batch-size <int>            Default: 192
  --run-tag <str>               Default: santos_em_symbolic_only
  --symbolic-repr <mode>        Default: concat
  --symbolic-normalize <mode>   Default: none
  --tile-repeat <int>           Default: 1
  --em-pair-feat-norm <0|1>     Default: 1
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK_ALIAS="$2"; shift 2 ;;
    --spec) SPEC_PATH="$2"; shift 2 ;;
    --spec-dir) SPEC_DIR="$2"; shift 2 ;;
    --cand) CAND_ID="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --symbolic-repr) ONLINE_SYMBOLIC_REPR="$2"; shift 2 ;;
    --symbolic-normalize) ONLINE_SYMBOLIC_NORMALIZE="$2"; shift 2 ;;
    --tile-repeat) ONLINE_SYMBOLIC_TILE_REPEAT="$2"; shift 2 ;;
    --em-pair-feat-norm) EM_PAIR_FEAT_NORM="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

case "${TASK_ALIAS}" in
  em|entity_matching)
    TASK_ALIAS="em"
    TASK_NAME="entity_matching"
    TASK_PERMUTATION="0"
    [[ -z "${RUN_TAG}" ]] && RUN_TAG="santos_em_static3_plus_symbolic"
    ;;
  jts|joinable_table_search)
    TASK_ALIAS="jts"
    TASK_NAME="joinable_table_search"
    TASK_PERMUTATION="1"
    [[ -z "${RUN_TAG}" ]] && RUN_TAG="santos_jts_symbolic_verify"
    ;;
  sm|schema_matching)
    TASK_ALIAS="sm"
    TASK_NAME="schema_matching"
    TASK_PERMUTATION="2"
    [[ -z "${RUN_TAG}" ]] && RUN_TAG="santos_sm_symbolic_verify"
    ;;
  uts|union_table_search)
    TASK_ALIAS="uts"
    TASK_NAME="union_table_search"
    TASK_PERMUTATION="3"
    [[ -z "${RUN_TAG}" ]] && RUN_TAG="santos_uts_symbolic_verify"
    ;;
  *)
    echo "[ERROR] --task must be one of: em,jts,sm,uts"
    exit 1
    ;;
esac

if [[ -n "${SPEC_DIR}" || -n "${CAND_ID}" ]]; then
  if [[ -z "${SPEC_DIR}" || -z "${CAND_ID}" ]]; then
    echo "[ERROR] --spec-dir and --cand must be provided together."
    exit 1
  fi
  SPEC_PATH="${SPEC_DIR}/cand_${CAND_ID}.json"
fi

if [[ -z "${SPEC_PATH}" ]]; then
  if [[ "${TASK_ALIAS}" == "em" ]]; then
    SPEC_PATH="${DEFAULT_SPEC_ROOT}/${TASK_NAME}/cand_02.json"
  else
    SPEC_PATH="${DEFAULT_SPEC_ROOT}/${TASK_NAME}/cand_01.json"
  fi
fi

if [[ ! -f "${SPEC_PATH}" ]]; then
  echo "[ERROR] Spec not found: ${SPEC_PATH}"
  exit 1
fi
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
  echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -z "${GRAPH_DATA_DIR}" ]]; then
  echo "[ERROR] Set GRAPH_DATA_DIR to the graph data directory (e.g. /path/to/santos_benchmark_040303_no_token)" >&2
  exit 1
fi
if [[ -z "${TABLE_ROOT}" ]]; then
  echo "[ERROR] Set TABLE_ROOT to the datalake directory (e.g. /path/to/santos_benchmark_040303/datalake_plus)" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Train script not found: ${TRAIN_SCRIPT}"
  exit 1
fi

echo "[INFO] Running santos task verify: task=${TASK_NAME}"
echo "[INFO] spec=${SPEC_PATH}"
echo "[INFO] gpu=${GPU} seed=${SEED} epochs=${EPOCHS} patience=${PATIENCE} batch=${BATCH_SIZE}"
echo "[INFO] symbolic_repr=${ONLINE_SYMBOLIC_REPR} symbolic_norm=${ONLINE_SYMBOLIC_NORMALIZE} tile=${ONLINE_SYMBOLIC_TILE_REPEAT}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --dataset "${DATASET}" \
  --graph_data_dir "${GRAPH_DATA_DIR}" \
  --em_table_root "${TABLE_ROOT}" \
  --jts_table_root "${TABLE_ROOT}" \
  --sm_table_root "${TABLE_ROOT}" \
  --uts_table_root "${TABLE_ROOT}" \
  --task_permutation "${TASK_PERMUTATION}" \
  --limit_tasks 1 \
  --epochs "${EPOCHS}" \
  --early_stopping_patience "${PATIENCE}" \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_neighbors "${NUM_NEIGHBORS}" \
  --gnn_layers "${GNN_LAYERS}" \
  --gnn_type "${GNN_TYPE}" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --drop_cell_edges "${DROP_CELL_EDGES}" \
  --em_pair_feat_norm "${EM_PAIR_FEAT_NORM}" \
  --em_pair_cache_mode "${EM_PAIR_CACHE_MODE}" \
  --em_pair_cache_root "${PAIR_CACHE_ROOT}" \
  --em_pair_features embedding_similarity,row_value_overlap,row_profile,serial_value_alignment,serial_lexical_plus \
  --em_decoder_static_features serial_value_alignment \
  --feature_wiring_mode decoupled \
  --allow_empty_decoder_static_features 1 \
  --jts_pair_features jaccard_containment,value_distribution,overlap_coverage,value_profile,header_similarity \
  --jts_decoder_static_features none \
  --sm_pair_features header_similarity,value_stats,value_overlap \
  --sm_decoder_static_features value_stats \
  --uts_pair_features column_overlap,header_jaccard,table_size_ratio \
  --uts_decoder_static_features none \
  --run_tag "${RUN_TAG}" \
  --online_symbolic_spec_template "${SPEC_PATH}" \
  --online_symbolic_repr "${ONLINE_SYMBOLIC_REPR}" \
  --online_symbolic_normalize "${ONLINE_SYMBOLIC_NORMALIZE}" \
  --online_symbolic_tile_repeat "${ONLINE_SYMBOLIC_TILE_REPEAT}"
