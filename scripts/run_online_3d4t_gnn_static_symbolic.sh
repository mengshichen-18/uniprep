#!/usr/bin/env bash
set -euo pipefail

# Run online training for 3 datasets x 4 tasks with:
# GNN + optional static features + optional symbolic features.
#
# Defaults are chosen to match our recent 040303 online runs.
#
# Example 1 (static selected + symbolic c12 cand_01):
#   bash scripts/run_online_3d4t_gnn_static_symbolic.sh \
#     --static-preset selected \
#     --symbolic on \
#     --symbolic-template "/path/to/symbolic_specs/batches/<batch_name>/c12/{task}/cand_01.json"
#
# Example 2 (static only):
#   bash scripts/run_online_3d4t_gnn_static_symbolic.sh \
#     --symbolic off
#
# Example 3 (full static + symbolic off, single dataset):
#   bash scripts/run_online_3d4t_gnn_static_symbolic.sh \
#     --datasets santos_benchmark \
#     --static-preset full \
#     --symbolic off
#
# Example 4 (symbolic suite: c4,c8,c12 x cand01..cand05):
#   bash scripts/run_online_3d4t_gnn_static_symbolic.sh \
#     --symbolic on \
#     --symbolic-suite 1 \
#     --symbolic-dims c4,c8,c12 \
#     --symbolic-cands 01,02,03,04,05 \
#     --parallel 1 --gpu-ids 0,1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/ajoint_0318_v3c.py}"

# ----------------------------
# Defaults
# ----------------------------
DATASETS_CSV="magellan,santos_benchmark,wikidbs"
GPU_IDS_CSV="0"
PARALLEL=0

STATIC_PRESET="selected"   # off | selected | full | custom
SYMBOLIC_MODE="on"        # off | on
SYMBOLIC_TEMPLATE=""       # e.g. /.../c12/{task}/cand_01.json
SYMBOLIC_REPR="concat"
SYMBOLIC_NORMALIZE="zscore"
SYMBOLIC_TILE_REPEAT=1
SYMBOLIC_SUITE=0
SYMBOLIC_DIMS_CSV="c4,c8,c12"
SYMBOLIC_CANDS_CSV="01"
# NOTE: this root currently stores c4 (and legacy c1/c2).
SYMBOLIC_SPEC_ROOT_C124="${REPO_DIR}/symbolic_specs/batches/v2_tasklevel_nocontext_c124_gpt5_20260414_215520"
SYMBOLIC_SPEC_ROOT_C8C12="${REPO_DIR}/symbolic_specs/batches/v2_tasklevel_nocontext_t1_gpt5_20260414_173616"

EPOCHS=120
PATIENCE=20
LR=0.001
BATCH_SIZE=192
HIDDEN_DIM=256
NUM_NEIGHBORS="10,5"
GNN_LAYERS=2
GNN_TYPE="our"
NUM_WORKERS=4
SEED=0
DROP_CELL_EDGES=1
TASK_PERMUTATION=0
LIMIT_TASKS=0
RUN_TAG_PREFIX="online_3d4t"

# Custom static (used when --static-preset custom)
EM_GROUPS="${EM_GROUPS:-row_value_overlap,serial_value_alignment,serial_lexical_plus}"
EM_DECODER_GROUPS="${EM_DECODER_GROUPS:-serial_value_alignment}"
JTS_GROUPS="${JTS_GROUPS:-jaccard_containment,value_profile,header_similarity}"
JTS_DECODER_GROUPS="${JTS_DECODER_GROUPS:-}"
SM_GROUPS="${SM_GROUPS:-header_similarity,value_stats}"
SM_DECODER_GROUPS="${SM_DECODER_GROUPS:-value_stats}"
UTS_GROUPS="${UTS_GROUPS:-column_overlap,header_jaccard}"
UTS_DECODER_GROUPS="${UTS_DECODER_GROUPS:-}"
ALLOW_EMPTY_DECODER_GROUPS="${ALLOW_EMPTY_DECODER_GROUPS:-0}"
FEATURE_WIRING_MODE="${FEATURE_WIRING_MODE:-decoupled}"
EM_GENERATED_FEATURE_SPECS_PATH="${EM_GENERATED_FEATURE_SPECS_PATH:-}"
JTS_GENERATED_FEATURE_SPECS_PATH="${JTS_GENERATED_FEATURE_SPECS_PATH:-}"
SM_GENERATED_FEATURE_SPECS_PATH="${SM_GENERATED_FEATURE_SPECS_PATH:-}"
UTS_GENERATED_FEATURE_SPECS_PATH="${UTS_GENERATED_FEATURE_SPECS_PATH:-}"
DEBUG_MAX_TRAIN_EDGES="${DEBUG_MAX_TRAIN_EDGES:-0}"
DEBUG_MAX_VAL_EDGES="${DEBUG_MAX_VAL_EDGES:-0}"
DEBUG_MAX_TEST_EDGES="${DEBUG_MAX_TEST_EDGES:-0}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_online_3d4t_gnn_static_symbolic.sh [options]

Options:
  --datasets <csv>              Dataset list, default: ${DATASETS_CSV}
  --gpu-ids <csv>               Physical GPU ids, default: ${GPU_IDS_CSV}
  --parallel <0|1>              Run datasets in parallel, default: ${PARALLEL}

  --static-preset <mode>        off | selected | full | custom
  --symbolic <on|off>           Enable/disable symbolic
  --symbolic-template <path>    Template path supports {task},{dataset}
  --symbolic-repr <mode>        default: ${SYMBOLIC_REPR}
  --symbolic-normalize <mode>   default: ${SYMBOLIC_NORMALIZE}
  --symbolic-tile-repeat <int>  default: ${SYMBOLIC_TILE_REPEAT}
  --symbolic-suite <0|1>        Enable c4/c8/c12 suite runner (legacy c1/c2 still supported)
  --symbolic-dims <csv>         default: ${SYMBOLIC_DIMS_CSV}
  --symbolic-cands <csv>        default: ${SYMBOLIC_CANDS_CSV}
  --symbolic-spec-root-c124 <p> default: ${SYMBOLIC_SPEC_ROOT_C124}
  --symbolic-spec-root-c8c12 <p> default: ${SYMBOLIC_SPEC_ROOT_C8C12}

  --epochs <int>                default: ${EPOCHS}
  --patience <int>              default: ${PATIENCE}
  --lr <float>                  default: ${LR}
  --batch-size <int>            default: ${BATCH_SIZE}
  --hidden-dim <int>            default: ${HIDDEN_DIM}
  --num-neighbors <csv>         default: ${NUM_NEIGHBORS}
  --gnn-layers <int>            default: ${GNN_LAYERS}
  --gnn-type <name>             default: ${GNN_TYPE}
  --num-workers <int>           default: ${NUM_WORKERS}
  --seed <int>                  default: ${SEED}
  --drop-cell-edges <0|1>       default: ${DROP_CELL_EDGES}
  --task-permutation <int>      default: ${TASK_PERMUTATION}
  --limit-tasks <int>           default: ${LIMIT_TASKS}
  --run-tag-prefix <str>        default: ${RUN_TAG_PREFIX}

  --em-groups <csv>             only when --static-preset custom
  --em-decoder-groups <csv>     optional EM decoder subset; empty => same as --em-groups
  --jts-decoder-groups <csv>    optional JTS decoder subset; empty => same as --jts-groups
  --sm-decoder-groups <csv>     optional SM decoder subset; empty => same as --sm-groups
  --uts-decoder-groups <csv>    optional UTS decoder subset; empty => same as --uts-groups
  --allow-empty-decoder-groups <0|1>  if 1, keep empty decoder groups as empty (no fallback)
  --feature-wiring-mode <mode>  decoupled | coupled, default: ${FEATURE_WIRING_MODE}
  --em-generated-feature-specs-path <path>
                                Optional JSON file or dir of generated EM feature functions.
  --debug-max-train-edges <int> Optional debug cap for train supervision edges.
  --debug-max-val-edges <int>   Optional debug cap for val supervision edges.
  --debug-max-test-edges <int>  Optional debug cap for test supervision edges.
  --jts-groups <csv>            only when --static-preset custom
  --sm-groups <csv>             only when --static-preset custom
  --uts-groups <csv>            only when --static-preset custom
  --em-generated-feature-specs-path <path>
  --jts-generated-feature-specs-path <path>
  --sm-generated-feature-specs-path <path>
  --uts-generated-feature-specs-path <path>

  -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;

    --static-preset) STATIC_PRESET="$2"; shift 2 ;;
    --symbolic) SYMBOLIC_MODE="$2"; shift 2 ;;
    --symbolic-template) SYMBOLIC_TEMPLATE="$2"; shift 2 ;;
    --symbolic-repr) SYMBOLIC_REPR="$2"; shift 2 ;;
    --symbolic-normalize) SYMBOLIC_NORMALIZE="$2"; shift 2 ;;
    --symbolic-tile-repeat) SYMBOLIC_TILE_REPEAT="$2"; shift 2 ;;
    --symbolic-suite) SYMBOLIC_SUITE="$2"; shift 2 ;;
    --symbolic-dims) SYMBOLIC_DIMS_CSV="$2"; shift 2 ;;
    --symbolic-cands) SYMBOLIC_CANDS_CSV="$2"; shift 2 ;;
    --symbolic-spec-root-c124) SYMBOLIC_SPEC_ROOT_C124="$2"; shift 2 ;;
    --symbolic-spec-root-c8c12) SYMBOLIC_SPEC_ROOT_C8C12="$2"; shift 2 ;;

    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
    --num-neighbors) NUM_NEIGHBORS="$2"; shift 2 ;;
    --gnn-layers) GNN_LAYERS="$2"; shift 2 ;;
    --gnn-type) GNN_TYPE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --drop-cell-edges) DROP_CELL_EDGES="$2"; shift 2 ;;
    --task-permutation) TASK_PERMUTATION="$2"; shift 2 ;;
    --limit-tasks) LIMIT_TASKS="$2"; shift 2 ;;
    --run-tag-prefix) RUN_TAG_PREFIX="$2"; shift 2 ;;

    --em-groups) EM_GROUPS="$2"; shift 2 ;;
    --em-decoder-groups) EM_DECODER_GROUPS="$2"; shift 2 ;;
    --jts-decoder-groups) JTS_DECODER_GROUPS="$2"; shift 2 ;;
    --sm-decoder-groups) SM_DECODER_GROUPS="$2"; shift 2 ;;
    --uts-decoder-groups) UTS_DECODER_GROUPS="$2"; shift 2 ;;
    --allow-empty-decoder-groups) ALLOW_EMPTY_DECODER_GROUPS="$2"; shift 2 ;;
    --feature-wiring-mode) FEATURE_WIRING_MODE="$2"; shift 2 ;;
    --em-generated-feature-specs-path) EM_GENERATED_FEATURE_SPECS_PATH="$2"; shift 2 ;;
    --jts-generated-feature-specs-path) JTS_GENERATED_FEATURE_SPECS_PATH="$2"; shift 2 ;;
    --sm-generated-feature-specs-path) SM_GENERATED_FEATURE_SPECS_PATH="$2"; shift 2 ;;
    --uts-generated-feature-specs-path) UTS_GENERATED_FEATURE_SPECS_PATH="$2"; shift 2 ;;
    --debug-max-train-edges) DEBUG_MAX_TRAIN_EDGES="$2"; shift 2 ;;
    --debug-max-val-edges) DEBUG_MAX_VAL_EDGES="$2"; shift 2 ;;
    --debug-max-test-edges) DEBUG_MAX_TEST_EDGES="$2"; shift 2 ;;
    --jts-groups) JTS_GROUPS="$2"; shift 2 ;;
    --sm-groups) SM_GROUPS="$2"; shift 2 ;;
    --uts-groups) UTS_GROUPS="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Train script not found: ${TRAIN_SCRIPT}"
  exit 1
fi
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
  echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

case "${STATIC_PRESET}" in
  off)
    EM_GROUPS=""
    EM_DECODER_GROUPS=""
    JTS_GROUPS=""
    JTS_DECODER_GROUPS=""
    SM_GROUPS=""
    SM_DECODER_GROUPS=""
    UTS_GROUPS=""
    UTS_DECODER_GROUPS=""
    ;;
  selected)
    # Current selected static setup in our recent runs.
    EM_GROUPS="row_value_overlap,serial_value_alignment,serial_lexical_plus"
    JTS_GROUPS="jaccard_containment,value_profile,header_similarity"
    SM_GROUPS="header_similarity,value_stats"
    UTS_GROUPS="column_overlap,header_jaccard"
    ;;
  full)
    EM_GROUPS="embedding_similarity,row_value_overlap,row_profile,serial_value_alignment,serial_lexical_plus"
    JTS_GROUPS="jaccard_containment,value_distribution,overlap_coverage,value_profile,header_similarity"
    SM_GROUPS="header_similarity,value_stats,value_overlap"
    UTS_GROUPS="column_overlap,header_jaccard,table_size_ratio"
    ;;
  custom)
    if [[ -z "${EM_GROUPS}" || -z "${JTS_GROUPS}" || -z "${SM_GROUPS}" || -z "${UTS_GROUPS}" ]]; then
      echo "[ERROR] static-preset=custom requires --em-groups/--jts-groups/--sm-groups/--uts-groups"
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] Invalid --static-preset: ${STATIC_PRESET}"
    exit 1
    ;;
esac

if [[ "${ALLOW_EMPTY_DECODER_GROUPS}" != "1" ]]; then
  if [[ -z "${EM_DECODER_GROUPS}" ]]; then
    EM_DECODER_GROUPS="${EM_GROUPS}"
  fi
  if [[ -z "${JTS_DECODER_GROUPS}" ]]; then
    JTS_DECODER_GROUPS="${JTS_GROUPS}"
  fi
  if [[ -z "${SM_DECODER_GROUPS}" ]]; then
    SM_DECODER_GROUPS="${SM_GROUPS}"
  fi
  if [[ -z "${UTS_DECODER_GROUPS}" ]]; then
    UTS_DECODER_GROUPS="${UTS_GROUPS}"
  fi
fi

case "${SYMBOLIC_MODE}" in
  on)
    if [[ "${SYMBOLIC_SUITE}" != "1" && -z "${SYMBOLIC_TEMPLATE}" ]]; then
      echo "[ERROR] --symbolic on requires --symbolic-template"
      exit 1
    fi
    ;;
  off) ;;
  *)
    echo "[ERROR] Invalid --symbolic: ${SYMBOLIC_MODE}"
    exit 1
    ;;
esac

if [[ "${SYMBOLIC_SUITE}" != "0" && "${SYMBOLIC_SUITE}" != "1" ]]; then
  echo "[ERROR] --symbolic-suite must be 0 or 1"
  exit 1
fi
if [[ "${ALLOW_EMPTY_DECODER_GROUPS}" != "0" && "${ALLOW_EMPTY_DECODER_GROUPS}" != "1" ]]; then
  echo "[ERROR] --allow-empty-decoder-groups must be 0 or 1"
  exit 1
fi
case "${FEATURE_WIRING_MODE}" in
  decoupled|coupled) ;;
  *)
    echo "[ERROR] --feature-wiring-mode must be decoupled or coupled"
    exit 1
    ;;
esac
if [[ "${SYMBOLIC_SUITE}" == "1" && "${SYMBOLIC_MODE}" != "on" ]]; then
  echo "[ERROR] --symbolic-suite 1 requires --symbolic on"
  exit 1
fi

# Decoder input is fixed to base(static-selected) + symbolic.
echo "[INFO] decoder input policy fixed: base(static-selected) + symbolic"

IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "[ERROR] No GPU ids provided"
  exit 1
fi

IFS=',' read -r -a SYMBOLIC_DIMS <<< "${SYMBOLIC_DIMS_CSV}"
IFS=',' read -r -a SYMBOLIC_CANDS_RAW <<< "${SYMBOLIC_CANDS_CSV}"

normalize_cand() {
  local raw="$1"
  if [[ "${raw}" =~ ^[0-9]+$ ]]; then
    printf "%02d" "${raw}"
  else
    printf "%s" "${raw}"
  fi
}

SYMBOLIC_CANDS=()
for c in "${SYMBOLIC_CANDS_RAW[@]}"; do
  c="$(echo "${c}" | xargs)"
  [[ -z "${c}" ]] && continue
  SYMBOLIC_CANDS+=("$(normalize_cand "${c}")")
done

if [[ "${SYMBOLIC_SUITE}" == "1" ]]; then
  if [[ "${#SYMBOLIC_DIMS[@]}" -eq 0 || "${#SYMBOLIC_CANDS[@]}" -eq 0 ]]; then
    echo "[ERROR] symbolic suite needs non-empty --symbolic-dims and --symbolic-cands"
    exit 1
  fi
fi

resolve_suite_template() {
  local dim="$1"
  local cand="$2"
  local root=""
  case "${dim}" in
    c1|c2|c4) root="${SYMBOLIC_SPEC_ROOT_C124}" ;;
    c8|c12) root="${SYMBOLIC_SPEC_ROOT_C8C12}" ;;
    *)
      echo "[ERROR] Unsupported symbolic dim in suite: ${dim}" >&2
      return 1
      ;;
  esac
  printf "%s/%s/{task}/cand_%s.json" "${root}" "${dim}" "${cand}"
}

check_template_exists() {
  local tpl="$1"
  local -a tasks=("entity_matching" "joinable_table_search" "schema_matching" "union_table_search")
  local probe_ds="magellan"
  local t p
  for t in "${tasks[@]}"; do
    p="${tpl//\{task\}/${t}}"
    p="${p//\{dataset\}/${probe_ds}}"
    if [[ ! -f "${p}" ]]; then
      echo "[ERROR] symbolic spec not found: ${p}" >&2
      return 1
    fi
  done
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_DIR}/outputs/online_symbolic_verify/${RUN_TAG_PREFIX}_${RUN_STAMP}"
mkdir -p "${LOG_DIR}"

echo "[INFO] run_stamp=${RUN_STAMP}"
echo "[INFO] log_dir=${LOG_DIR}"
echo "[INFO] datasets=${DATASETS_CSV}"
echo "[INFO] static_preset=${STATIC_PRESET}"
if [[ -n "${EM_GENERATED_FEATURE_SPECS_PATH}" ]]; then
  echo "[INFO] em_generated_feature_specs_path=${EM_GENERATED_FEATURE_SPECS_PATH}"
fi
if [[ -n "${JTS_GENERATED_FEATURE_SPECS_PATH}" ]]; then
  echo "[INFO] jts_generated_feature_specs_path=${JTS_GENERATED_FEATURE_SPECS_PATH}"
fi
if [[ -n "${SM_GENERATED_FEATURE_SPECS_PATH}" ]]; then
  echo "[INFO] sm_generated_feature_specs_path=${SM_GENERATED_FEATURE_SPECS_PATH}"
fi
if [[ -n "${UTS_GENERATED_FEATURE_SPECS_PATH}" ]]; then
  echo "[INFO] uts_generated_feature_specs_path=${UTS_GENERATED_FEATURE_SPECS_PATH}"
fi
if [[ "${DEBUG_MAX_TRAIN_EDGES}" != "0" || "${DEBUG_MAX_VAL_EDGES}" != "0" || "${DEBUG_MAX_TEST_EDGES}" != "0" ]]; then
  echo "[INFO] debug_edge_caps train=${DEBUG_MAX_TRAIN_EDGES} val=${DEBUG_MAX_VAL_EDGES} test=${DEBUG_MAX_TEST_EDGES}"
fi
echo "[INFO] symbolic=${SYMBOLIC_MODE}"
if [[ "${SYMBOLIC_MODE}" == "on" ]]; then
  if [[ "${SYMBOLIC_SUITE}" == "1" ]]; then
    echo "[INFO] symbolic_suite=1 dims=${SYMBOLIC_DIMS_CSV} cands=${SYMBOLIC_CANDS_CSV}"
    echo "[INFO] symbolic_spec_root_c124=${SYMBOLIC_SPEC_ROOT_C124}"
    echo "[INFO] symbolic_spec_root_c8c12=${SYMBOLIC_SPEC_ROOT_C8C12}"
  else
    echo "[INFO] symbolic_template=${SYMBOLIC_TEMPLATE}"
  fi
fi

run_one_dataset() {
  local dataset="$1"
  local gpu_id="$2"
  local run_suffix="$3"
  local symbolic_template="$4"

  local graph_dir="${GRAPH_DIR_BASE:?Set GRAPH_DIR_BASE to the graph data root}/${dataset}_040303_no_token"
  local table_root="${TABLE_ROOT_BASE:?Set TABLE_ROOT_BASE to the datasets root}/${dataset}_040303/datalake_plus"
  local run_tag=""
  local log_file=""
  if [[ -n "${run_suffix}" ]]; then
    mkdir -p "${LOG_DIR}/${run_suffix}"
    run_tag="${RUN_TAG_PREFIX}_${run_suffix}_${dataset}_seed${SEED}"
    log_file="${LOG_DIR}/${run_suffix}/${dataset}.log"
  else
    run_tag="${RUN_TAG_PREFIX}_${dataset}_seed${SEED}"
    log_file="${LOG_DIR}/${dataset}.log"
  fi

  if [[ ! -d "${graph_dir}" ]]; then
    echo "[WARN] graph_dir not found: ${graph_dir}"
  fi
  if [[ ! -d "${table_root}" ]]; then
    echo "[WARN] table_root not found: ${table_root}"
  fi

  local -a cmd
  cmd=(
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
    --dataset "${dataset}"
    --graph_data_dir "${graph_dir}"
    --em_table_root "${table_root}"
    --jts_table_root "${table_root}"
    --sm_table_root "${table_root}"
    --uts_table_root "${table_root}"
    --task_permutation "${TASK_PERMUTATION}"
    --limit_tasks "${LIMIT_TASKS}"
    --epochs "${EPOCHS}"
    --early_stopping_patience "${PATIENCE}"
    --lr "${LR}"
    --batch_size "${BATCH_SIZE}"
    --hidden_dim "${HIDDEN_DIM}"
    --num_neighbors "${NUM_NEIGHBORS}"
    --gnn_layers "${GNN_LAYERS}"
    --gnn_type "${GNN_TYPE}"
    --device cuda:0
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --drop_cell_edges "${DROP_CELL_EDGES}"
    --em_pair_feat_norm 1
    --em_pair_cache_mode readwrite
    --em_pair_cache_root "${REPO_DIR}/outputs/em_pair_cache/${dataset}"
    --em_pair_features "${EM_GROUPS}"
    --em_decoder_static_features "${EM_DECODER_GROUPS}"
    --em_generated_feature_specs_path "${EM_GENERATED_FEATURE_SPECS_PATH}"
    --debug_max_train_edges "${DEBUG_MAX_TRAIN_EDGES}"
    --debug_max_val_edges "${DEBUG_MAX_VAL_EDGES}"
    --debug_max_test_edges "${DEBUG_MAX_TEST_EDGES}"
    --feature_wiring_mode "${FEATURE_WIRING_MODE}"
    --allow_empty_decoder_static_features "${ALLOW_EMPTY_DECODER_GROUPS}"
    --jts_pair_features "${JTS_GROUPS}"
    --jts_decoder_static_features "${JTS_DECODER_GROUPS}"
    --jts_generated_feature_specs_path "${JTS_GENERATED_FEATURE_SPECS_PATH}"
    --sm_pair_features "${SM_GROUPS}"
    --sm_decoder_static_features "${SM_DECODER_GROUPS}"
    --sm_generated_feature_specs_path "${SM_GENERATED_FEATURE_SPECS_PATH}"
    --uts_pair_features "${UTS_GROUPS}"
    --uts_decoder_static_features "${UTS_DECODER_GROUPS}"
    --uts_generated_feature_specs_path "${UTS_GENERATED_FEATURE_SPECS_PATH}"
    --run_tag "${run_tag}"
  )

  if [[ "${SYMBOLIC_MODE}" == "on" ]]; then
    cmd+=(
      --online_symbolic_spec_template "${symbolic_template}"
      --online_symbolic_repr "${SYMBOLIC_REPR}"
      --online_symbolic_normalize "${SYMBOLIC_NORMALIZE}"
      --online_symbolic_tile_repeat "${SYMBOLIC_TILE_REPEAT}"
    )
  fi

  {
    echo "[RUN] dataset=${dataset} gpu=${gpu_id}"
    printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
  } | tee "${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee -a "${log_file}"
}

run_variant() {
  local variant_label="$1"
  local variant_template="$2"
  if [[ "${SYMBOLIC_MODE}" == "on" ]]; then
    check_template_exists "${variant_template}"
    echo "[INFO] variant=${variant_label:-single} symbolic_template=${variant_template}"
  else
    echo "[INFO] variant=${variant_label:-single} symbolic=off"
  fi

  if [[ "${PARALLEL}" == "1" ]]; then
    echo "[INFO] parallel mode on"
    pids=()
    for i in "${!DATASETS[@]}"; do
      ds="${DATASETS[$i]}"
      gpu="${GPU_IDS[$(( i % ${#GPU_IDS[@]} ))]}"
      run_one_dataset "${ds}" "${gpu}" "${variant_label}" "${variant_template}" &
      pids+=("$!")
    done
    for p in "${pids[@]}"; do
      wait "${p}"
    done
  else
    echo "[INFO] sequential mode"
    for i in "${!DATASETS[@]}"; do
      ds="${DATASETS[$i]}"
      gpu="${GPU_IDS[$(( i % ${#GPU_IDS[@]} ))]}"
      run_one_dataset "${ds}" "${gpu}" "${variant_label}" "${variant_template}"
    done
  fi
}

if [[ "${SYMBOLIC_SUITE}" == "1" ]]; then
  for d in "${SYMBOLIC_DIMS[@]}"; do
    d="$(echo "${d}" | xargs)"
    [[ -z "${d}" ]] && continue
    for cand in "${SYMBOLIC_CANDS[@]}"; do
      tpl="$(resolve_suite_template "${d}" "${cand}")"
      run_variant "${d}_cand${cand}" "${tpl}"
    done
  done
else
  run_variant "" "${SYMBOLIC_TEMPLATE}"
fi

echo "[DONE] logs saved to ${LOG_DIR}"
