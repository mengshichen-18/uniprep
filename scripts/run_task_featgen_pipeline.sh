#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/ajoint_0318_v3c.py}"
ATOM_GEN_SCRIPT="${ATOM_GEN_SCRIPT:-${REPO_DIR}/scripts/generate_task_feature_functions_gpt5.py}"
SYMBOLIC_GEN_WRAPPER="${SYMBOLIC_GEN_WRAPPER:-${REPO_DIR}/scripts/run_generate_symbolic_spec_v3.sh}"
API_KEY_FILE="${API_KEY_FILE:-${REPO_DIR}/../0325_policy_pro/LIGHTNING_API_KEY.md}"
API_KEY_LABEL="${API_KEY_LABEL:-0428_KEY}"

TASK="${TASK:-entity_matching}"
DATASET="${DATASET:-wikidbs}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-featgen_${TASK}_${DATASET}_c12}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_DIR}/outputs/featgen_pipeline/${TASK}/${DATASET}}"
CONTEXTS_DIR="${CONTEXTS_DIR:-${REPO_DIR}/outputs/featgen_contexts}"

ATOM_GEN_MODE="${ATOM_GEN_MODE:-real}"   # dryrun | real
FORCE_ATOM_REGEN="${FORCE_ATOM_REGEN:-0}"
NUM_GENERATED_FEATURES="${NUM_GENERATED_FEATURES:-auto}"
TARGET_TOTAL_POOL_SIZE="${TARGET_TOTAL_POOL_SIZE:-12}"
ATOM_MODEL="${ATOM_MODEL:-gpt-5}"
ATOM_BASE_URL="${ATOM_BASE_URL:-}"
ATOM_REASONING_EFFORT="${ATOM_REASONING_EFFORT:-low}"
ATOM_MAX_COMPLETION_TOKENS="${ATOM_MAX_COMPLETION_TOKENS:-2200}"
ATOM_TIMEOUT_SEC="${ATOM_TIMEOUT_SEC:-120}"
ATOM_MAX_REPAIR_ATTEMPTS="${ATOM_MAX_REPAIR_ATTEMPTS:-2}"
ATOM_ALLOW_EXTRA_FEATURES_TRUNCATE="${ATOM_ALLOW_EXTRA_FEATURES_TRUNCATE:-0}"

ENABLE_SYMBOLIC="${ENABLE_SYMBOLIC:-1}"
SYMBOLIC_GEN_MODE="${SYMBOLIC_GEN_MODE:-real}"   # dryrun | real
FORCE_SYMBOLIC_REGEN="${FORCE_SYMBOLIC_REGEN:-0}"
NUM_SYMBOLIC_CHANNELS="${NUM_SYMBOLIC_CHANNELS:-auto}"
SYMBOLIC_MIN_CHANNELS="${SYMBOLIC_MIN_CHANNELS:-auto}"
SYMBOLIC_MODEL="${SYMBOLIC_MODEL:-gpt-5}"
SYMBOLIC_BASE_URL="${SYMBOLIC_BASE_URL:-}"
SYMBOLIC_REASONING_EFFORT="${SYMBOLIC_REASONING_EFFORT:-low}"
SYMBOLIC_MAX_COMPLETION_TOKENS="${SYMBOLIC_MAX_COMPLETION_TOKENS:-3200}"
SYMBOLIC_TIMEOUT_SEC="${SYMBOLIC_TIMEOUT_SEC:-120}"
SYMBOLIC_REPR="${SYMBOLIC_REPR:-concat}"
SYMBOLIC_NORMALIZE="${SYMBOLIC_NORMALIZE:-none}"
SYMBOLIC_TILE_REPEAT="${SYMBOLIC_TILE_REPEAT:-1}"
SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO="${SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO:--1}"
SYMBOLIC_MAX_REPAIR_ATTEMPTS="${SYMBOLIC_MAX_REPAIR_ATTEMPTS:-1}"
ALLOW_DATASET_CONTEXT="${ALLOW_DATASET_CONTEXT:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-192}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_NEIGHBORS="${NUM_NEIGHBORS:-10,5}"
GNN_LAYERS="${GNN_LAYERS:-2}"
GNN_TYPE="${GNN_TYPE:-our}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEBUG_MAX_TRAIN_EDGES="${DEBUG_MAX_TRAIN_EDGES:-0}"
DEBUG_MAX_VAL_EDGES="${DEBUG_MAX_VAL_EDGES:-0}"
DEBUG_MAX_TEST_EDGES="${DEBUG_MAX_TEST_EDGES:-0}"
EM_PAIR_CACHE_MODE="${EM_PAIR_CACHE_MODE:-readwrite}"
EM_PAIR_CACHE_ROOT="${EM_PAIR_CACHE_ROOT:-${REPO_DIR}/outputs/em_pair_cache/${DATASET}}"
EM_ROW_STATS_MODE="${EM_ROW_STATS_MODE:-full}"

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
if [[ "${EM_PAIR_CACHE_MODE}" == "readwrite" && -n "${EM_PAIR_CACHE_ROOT}" ]]; then
  mkdir -p "${EM_PAIR_CACHE_ROOT}"
fi

cfg_json="$("${PYTHON_BIN}" - "${TASK}" <<'PY'
import json
import sys
from task_featgen_config import TASK_FEATGEN_CONFIGS

task = sys.argv[1]
cfg = TASK_FEATGEN_CONFIGS[task]
print(json.dumps(cfg, ensure_ascii=False))
PY
)"

read_cfg() {
  local key="$1"
  CFG_JSON="${cfg_json}" CFG_KEY="${key}" "${PYTHON_BIN}" - <<'PY'
import json
import os

cfg = json.loads(os.environ["CFG_JSON"])
key = os.environ["CFG_KEY"]
value = cfg.get(key, "")
if isinstance(value, list):
    print(",".join(str(x) for x in value))
else:
    print(str(value))
PY
}

case "${TASK}" in
  entity_matching) TASK_PERMUTATION=0 ;;
  joinable_table_search) TASK_PERMUTATION=1 ;;
  schema_matching) TASK_PERMUTATION=2 ;;
  union_table_search) TASK_PERMUTATION=3 ;;
  *)
    echo "[ERROR] unsupported TASK=${TASK}" >&2
    exit 2
    ;;
esac

SHORT_NAME="$(read_cfg short_name)"
TASK_SCOPE="$(read_cfg task_scope)"
TEACHER_FEATURE_POOL="${TEACHER_FEATURE_POOL:-$(read_cfg teacher_pool_atoms)}"
GENERATOR_EXEMPLAR_FEATURES="${GENERATOR_EXEMPLAR_FEATURES:-$(read_cfg human_exemplar_atoms)}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-$(read_cfg task_description)}"
DATA_DESCRIPTION="${DATA_DESCRIPTION:-$(read_cfg data_description)}"
SELECTION_NOTES="${SELECTION_NOTES:-$(read_cfg selection_notes)}"
TARGET_SOURCE_GROUPS="${TARGET_SOURCE_GROUPS:-$(read_cfg symbolic_source_groups_csv)}"
TARGET_DECODER_STATIC_GROUPS="${TARGET_DECODER_STATIC_GROUPS:-$(read_cfg decoder_static_groups_csv)}"
TARGET_DECODER_STATIC_ATOMS="${TARGET_DECODER_STATIC_ATOMS:-$(read_cfg decoder_static_atoms)}"
SYMBOLIC_BASE_ATOMS="${SYMBOLIC_BASE_ATOMS:-}"
PROTECTED_GENERATOR_ATOMS="${PROTECTED_GENERATOR_ATOMS:-}"

GEN_SPECS_PATH="${GEN_SPECS_PATH:-${ARTIFACT_DIR}/${TASK}__${DATASET}__generated_atoms.json}"
DATASET_CONTEXT_JSON="${DATASET_CONTEXT_JSON:-${CONTEXTS_DIR}/${TASK}__${DATASET}.json}"

EXEMPLAR_COUNT="$("${PYTHON_BIN}" - "${GENERATOR_EXEMPLAR_FEATURES}" <<'PY'
import sys
items = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(len(items))
PY
)"

if [[ -z "${SYMBOLIC_BASE_ATOMS}" ]]; then
  SYMBOLIC_BASE_ATOMS="$("${PYTHON_BIN}" - "${GENERATOR_EXEMPLAR_FEATURES}" "${TARGET_DECODER_STATIC_ATOMS}" <<'PY'
import sys

ordered = []
seen = set()
for raw in sys.argv[1:]:
    for item in str(raw).split(","):
        token = item.strip()
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
print(",".join(ordered))
PY
)"
fi

if [[ -z "${PROTECTED_GENERATOR_ATOMS}" ]]; then
  PROTECTED_GENERATOR_ATOMS="$("${PYTHON_BIN}" - "${GENERATOR_EXEMPLAR_FEATURES}" "${TARGET_DECODER_STATIC_ATOMS}" <<'PY'
import sys

ordered = []
seen = set()
for raw in sys.argv[1:]:
    for item in str(raw).split(","):
        token = item.strip()
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
print(",".join(ordered))
PY
)"
fi

SYMBOLIC_BASE_COUNT="$("${PYTHON_BIN}" - "${SYMBOLIC_BASE_ATOMS}" <<'PY'
import sys
items = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(len(items))
PY
)"
if [[ -z "${NUM_GENERATED_FEATURES}" || "${NUM_GENERATED_FEATURES}" == "auto" ]]; then
  NUM_GENERATED_FEATURES=$(( TARGET_TOTAL_POOL_SIZE - SYMBOLIC_BASE_COUNT ))
fi
if (( NUM_GENERATED_FEATURES < 0 )); then
  echo "[ERROR] NUM_GENERATED_FEATURES became negative: target_total_pool_size=${TARGET_TOTAL_POOL_SIZE}, symbolic_base_count=${SYMBOLIC_BASE_COUNT}" >&2
  echo "[ERROR] symbolic base atoms=${SYMBOLIC_BASE_ATOMS}" >&2
  exit 2
fi

if [[ "${FORCE_ATOM_REGEN}" == "1" || ! -f "${GEN_SPECS_PATH}" ]]; then
  echo "[INFO] generating atoms for task=${TASK} dataset=${DATASET} -> ${GEN_SPECS_PATH}"
  atom_args=(
    --task "${TASK}"
    --output "${GEN_SPECS_PATH}"
    --num-features "${NUM_GENERATED_FEATURES}"
    --target-total-pool-size "${TARGET_TOTAL_POOL_SIZE}"
    --teacher-feature-pool "${TEACHER_FEATURE_POOL}"
    --preserved-features "${GENERATOR_EXEMPLAR_FEATURES}"
    --protected-feature-names "${PROTECTED_GENERATOR_ATOMS}"
    --task-description "${TASK_DESCRIPTION}"
    --data-description "${DATA_DESCRIPTION}"
    --selection-notes "${SELECTION_NOTES}"
    --model "${ATOM_MODEL}"
    --reasoning-effort "${ATOM_REASONING_EFFORT}"
    --max-completion-tokens "${ATOM_MAX_COMPLETION_TOKENS}"
    --timeout-sec "${ATOM_TIMEOUT_SEC}"
    --max-repair-attempts "${ATOM_MAX_REPAIR_ATTEMPTS}"
    --allow-extra-features-truncate "${ATOM_ALLOW_EXTRA_FEATURES_TRUNCATE}"
    --api-key-file "${API_KEY_FILE}"
    --api-key-label "${API_KEY_LABEL}"
    --dump-prompt "${ARTIFACT_DIR}/${TASK}__${DATASET}__atom_prompt.json"
    --dump-validation "${ARTIFACT_DIR}/${TASK}__${DATASET}__atom_validation.json"
  )
  if [[ -n "${ATOM_BASE_URL}" ]]; then
    atom_args+=(--base-url "${ATOM_BASE_URL}")
  fi
  if [[ -f "${DATASET_CONTEXT_JSON}" ]]; then
    atom_args+=(--dataset-context-json "${DATASET_CONTEXT_JSON}")
  fi
  if [[ "${ATOM_GEN_MODE}" == "dryrun" ]]; then
    atom_args+=(--dry-run 1 --dump-response "${ARTIFACT_DIR}/${TASK}__${DATASET}__atom_response.json")
  else
    atom_args+=(--dump-response "${ARTIFACT_DIR}/${TASK}__${DATASET}__atom_response.txt")
  fi
  "${PYTHON_BIN}" "${ATOM_GEN_SCRIPT}" "${atom_args[@]}"
fi

SYMBOLIC_FEATURE_POOL="$("${PYTHON_BIN}" - "${GEN_SPECS_PATH}" "${SYMBOLIC_BASE_ATOMS}" <<'PY'
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

SYMBOLIC_POOL_COUNT="$("${PYTHON_BIN}" - "${SYMBOLIC_FEATURE_POOL}" <<'PY'
import sys
items = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(len(items))
PY
)"
if (( SYMBOLIC_POOL_COUNT > TARGET_TOTAL_POOL_SIZE )); then
  echo "[ERROR] symbolic feature pool size overflow: got ${SYMBOLIC_POOL_COUNT}, expected at most ${TARGET_TOTAL_POOL_SIZE}" >&2
  echo "[ERROR] symbolic base atoms=${SYMBOLIC_BASE_ATOMS}" >&2
  echo "[ERROR] symbolic feature pool=${SYMBOLIC_FEATURE_POOL}" >&2
  exit 2
fi
if (( SYMBOLIC_POOL_COUNT <= 0 )); then
  echo "[ERROR] symbolic feature pool is empty after dedup." >&2
  exit 2
fi
if (( SYMBOLIC_POOL_COUNT < TARGET_TOTAL_POOL_SIZE )); then
  echo "[WARN] symbolic feature pool shrank after dedup: got ${SYMBOLIC_POOL_COUNT}, target=${TARGET_TOTAL_POOL_SIZE}" >&2
  echo "[WARN] This usually means one or more generated atom names matched existing atoms in the base pool." >&2
fi

SYMBOLIC_EXACT_CHANNELS=""
if [[ -n "${NUM_SYMBOLIC_CHANNELS}" && "${NUM_SYMBOLIC_CHANNELS}" != "auto" ]]; then
  SYMBOLIC_EXACT_CHANNELS="${NUM_SYMBOLIC_CHANNELS}"
fi
if [[ -z "${SYMBOLIC_MIN_CHANNELS}" || "${SYMBOLIC_MIN_CHANNELS}" == "auto" ]]; then
  if [[ -n "${SYMBOLIC_EXACT_CHANNELS}" ]]; then
    SYMBOLIC_MIN_CHANNELS="${SYMBOLIC_EXACT_CHANNELS}"
  else
    SYMBOLIC_MIN_CHANNELS="${TARGET_TOTAL_POOL_SIZE}"
  fi
fi

if [[ -n "${SYMBOLIC_EXACT_CHANNELS}" ]]; then
  SYMBOLIC_SPEC_BASENAME="${TASK}__${DATASET}__symbolic_c${SYMBOLIC_EXACT_CHANNELS}.json"
else
  SYMBOLIC_SPEC_BASENAME="${TASK}__${DATASET}__symbolic_auto_min${SYMBOLIC_MIN_CHANNELS}.json"
fi
SYMBOLIC_SPEC_PATH="${SYMBOLIC_SPEC_PATH:-${ARTIFACT_DIR}/${SYMBOLIC_SPEC_BASENAME}}"

if [[ "${ENABLE_SYMBOLIC}" == "1" ]]; then
  if [[ "${FORCE_SYMBOLIC_REGEN}" == "1" || ! -f "${SYMBOLIC_SPEC_PATH}" ]]; then
    echo "[INFO] generating symbolic spec for task=${TASK} dataset=${DATASET} -> ${SYMBOLIC_SPEC_PATH}"
    sym_args=(
      --task "${TASK}"
      --feature-pool "${SYMBOLIC_FEATURE_POOL}"
      --output "${SYMBOLIC_SPEC_PATH}"
      --min-num-channels "${SYMBOLIC_MIN_CHANNELS}"
      --model "${SYMBOLIC_MODEL}"
      --reasoning-effort "${SYMBOLIC_REASONING_EFFORT}"
      --timeout-sec "${SYMBOLIC_TIMEOUT_SEC}"
      --max-completion-tokens "${SYMBOLIC_MAX_COMPLETION_TOKENS}"
      --feature-cards-file "${REPO_DIR}/symbolic_feature_cards.json"
      --summary-output "${ARTIFACT_DIR}/${TASK}__${DATASET}__symbolic_summary.json"
      --dump-prompt "${ARTIFACT_DIR}/${TASK}__${DATASET}__symbolic_prompt.json"
      --api-key-file "${API_KEY_FILE}"
      --api-key-label "${API_KEY_LABEL}"
      --allow-dataset-context "${ALLOW_DATASET_CONTEXT}"
      --max-audit-passthrough-ratio "${SYMBOLIC_MAX_AUDIT_PASSTHROUGH_RATIO}"
      --max-repair-attempts "${SYMBOLIC_MAX_REPAIR_ATTEMPTS}"
    )
    if [[ -n "${SYMBOLIC_BASE_URL}" ]]; then
      sym_args+=(--base-url "${SYMBOLIC_BASE_URL}")
    fi
    if [[ -f "${DATASET_CONTEXT_JSON}" ]]; then
      sym_args+=(--dataset-context-file "${DATASET_CONTEXT_JSON}")
    fi
    if [[ -n "${SYMBOLIC_EXACT_CHANNELS}" ]]; then
      sym_args+=(--num-channels "${SYMBOLIC_EXACT_CHANNELS}")
    fi
    if [[ "${SYMBOLIC_GEN_MODE}" == "dryrun" ]]; then
      sym_args+=(--dry-run 1)
    fi
    bash "${SYMBOLIC_GEN_WRAPPER}" "${sym_args[@]}"
  fi
fi

MANIFEST_PATH="${MANIFEST_PATH:-${ARTIFACT_DIR}/${TASK}__${DATASET}__generation_manifest.json}"

EM_GROUPS="$("${PYTHON_BIN}" - <<'PY'
from task_featgen_config import TASK_FEATGEN_CONFIGS
print(TASK_FEATGEN_CONFIGS["entity_matching"]["symbolic_source_groups_csv"])
PY
)"
JTS_GROUPS="$("${PYTHON_BIN}" - <<'PY'
from task_featgen_config import TASK_FEATGEN_CONFIGS
print(TASK_FEATGEN_CONFIGS["joinable_table_search"]["symbolic_source_groups_csv"])
PY
)"
SM_GROUPS="$("${PYTHON_BIN}" - <<'PY'
from task_featgen_config import TASK_FEATGEN_CONFIGS
print(TASK_FEATGEN_CONFIGS["schema_matching"]["symbolic_source_groups_csv"])
PY
)"
UTS_GROUPS="$("${PYTHON_BIN}" - <<'PY'
from task_featgen_config import TASK_FEATGEN_CONFIGS
print(TASK_FEATGEN_CONFIGS["union_table_search"]["symbolic_source_groups_csv"])
PY
)"

EM_DECODER_GROUPS="$("${PYTHON_BIN}" - <<'PY'
from task_featgen_config import TASK_FEATGEN_CONFIGS
print(TASK_FEATGEN_CONFIGS["entity_matching"]["decoder_static_groups_csv"])
PY
)"
JTS_DECODER_GROUPS="$("${PYTHON_BIN}" - <<'PY'
from task_featgen_config import TASK_FEATGEN_CONFIGS
print(TASK_FEATGEN_CONFIGS["joinable_table_search"]["decoder_static_groups_csv"])
PY
)"
SM_DECODER_GROUPS="$("${PYTHON_BIN}" - <<'PY'
from task_featgen_config import TASK_FEATGEN_CONFIGS
print(TASK_FEATGEN_CONFIGS["schema_matching"]["decoder_static_groups_csv"])
PY
)"
UTS_DECODER_GROUPS="$("${PYTHON_BIN}" - <<'PY'
from task_featgen_config import TASK_FEATGEN_CONFIGS
print(TASK_FEATGEN_CONFIGS["union_table_search"]["decoder_static_groups_csv"])
PY
)"

EM_GENERATED_FEATURE_SPECS_PATH=""
JTS_GENERATED_FEATURE_SPECS_PATH=""
SM_GENERATED_FEATURE_SPECS_PATH=""
UTS_GENERATED_FEATURE_SPECS_PATH=""
ONLINE_SYMBOLIC_SPEC_PATH=""

case "${TASK}" in
  entity_matching)
    EM_GROUPS="${SYMBOLIC_FEATURE_POOL}"
    EM_DECODER_GROUPS="${TARGET_DECODER_STATIC_GROUPS}"
    EM_GENERATED_FEATURE_SPECS_PATH="${GEN_SPECS_PATH}"
    ;;
  joinable_table_search)
    JTS_GROUPS="${SYMBOLIC_FEATURE_POOL}"
    JTS_DECODER_GROUPS="${TARGET_DECODER_STATIC_GROUPS}"
    JTS_GENERATED_FEATURE_SPECS_PATH="${GEN_SPECS_PATH}"
    ;;
  schema_matching)
    SM_GROUPS="${SYMBOLIC_FEATURE_POOL}"
    SM_DECODER_GROUPS="${TARGET_DECODER_STATIC_GROUPS}"
    SM_GENERATED_FEATURE_SPECS_PATH="${GEN_SPECS_PATH}"
    ;;
  union_table_search)
    UTS_GROUPS="${SYMBOLIC_FEATURE_POOL}"
    UTS_DECODER_GROUPS="${TARGET_DECODER_STATIC_GROUPS}"
    UTS_GENERATED_FEATURE_SPECS_PATH="${GEN_SPECS_PATH}"
    ;;
esac

if [[ "${ENABLE_SYMBOLIC}" == "1" ]]; then
  ONLINE_SYMBOLIC_SPEC_PATH="${SYMBOLIC_SPEC_PATH}"
fi

echo "[INFO] task=${TASK} dataset=${DATASET} gpu=${GPU_ID}"
echo "[INFO] generator_exemplars=${GENERATOR_EXEMPLAR_FEATURES}"
echo "[INFO] exemplar_count=${EXEMPLAR_COUNT}"
echo "[INFO] symbolic_base_count=${SYMBOLIC_BASE_COUNT}"
echo "[INFO] num_generated_features=${NUM_GENERATED_FEATURES}"
echo "[INFO] target_total_pool_size=${TARGET_TOTAL_POOL_SIZE}"
echo "[INFO] symbolic_exact_channels=${SYMBOLIC_EXACT_CHANNELS:-auto}"
echo "[INFO] symbolic_min_channels=${SYMBOLIC_MIN_CHANNELS}"
echo "[INFO] target_decoder_static_atoms=${TARGET_DECODER_STATIC_ATOMS}"
echo "[INFO] symbolic_base_atoms=${SYMBOLIC_BASE_ATOMS}"
echo "[INFO] symbolic_feature_pool=${SYMBOLIC_FEATURE_POOL}"
echo "[INFO] training_source_pool=${SYMBOLIC_FEATURE_POOL}"
echo "[INFO] generated_specs=${GEN_SPECS_PATH}"
if [[ -f "${DATASET_CONTEXT_JSON}" ]]; then
  echo "[INFO] dataset_context=${DATASET_CONTEXT_JSON}"
fi
if [[ "${ENABLE_SYMBOLIC}" == "1" ]]; then
  echo "[INFO] symbolic_spec=${SYMBOLIC_SPEC_PATH}"
fi

"${PYTHON_BIN}" - \
  "${MANIFEST_PATH}" \
  "${TASK}" \
  "${DATASET}" \
  "${RUN_TAG}" \
  "${SEED}" \
  "${GENERATOR_EXEMPLAR_FEATURES}" \
  "${EXEMPLAR_COUNT}" \
  "${NUM_GENERATED_FEATURES}" \
  "${TARGET_TOTAL_POOL_SIZE}" \
  "${SYMBOLIC_EXACT_CHANNELS}" \
  "${SYMBOLIC_MIN_CHANNELS}" \
  "${TARGET_DECODER_STATIC_ATOMS}" \
  "${SYMBOLIC_FEATURE_POOL}" \
  "${GEN_SPECS_PATH}" \
  "${ENABLE_SYMBOLIC}" \
  "${SYMBOLIC_SPEC_PATH}" \
  "${DATASET_CONTEXT_JSON}" \
  "${SKIP_TRAIN}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
task = sys.argv[2]
dataset = sys.argv[3]
run_tag = sys.argv[4]
seed = sys.argv[5]
generator_exemplars = sys.argv[6]
exemplar_count = sys.argv[7]
num_generated_features = sys.argv[8]
target_total_pool_size = sys.argv[9]
symbolic_exact_channels = sys.argv[10]
symbolic_min_channels = sys.argv[11]
target_decoder_static_atoms = sys.argv[12]
symbolic_feature_pool = sys.argv[13]
gen_specs_path = sys.argv[14]
enable_symbolic = sys.argv[15]
symbolic_spec_path = sys.argv[16]
dataset_context_path = sys.argv[17]
skip_train = sys.argv[18]
payload = {
    "task": task,
    "dataset": dataset,
    "run_tag": run_tag,
    "seed": int(seed or 0),
    "generator_exemplars": [x.strip() for x in generator_exemplars.split(",") if x.strip()],
    "exemplar_count": int(exemplar_count or 0),
    "num_generated_features": int(num_generated_features or 0),
    "target_total_pool_size": int(target_total_pool_size or 0),
    "symbolic_exact_channels": (int(symbolic_exact_channels) if str(symbolic_exact_channels).strip() else None),
    "symbolic_min_channels": int(symbolic_min_channels or 0),
    "target_decoder_static_atoms": [x.strip() for x in target_decoder_static_atoms.split(",") if x.strip()],
    "symbolic_feature_pool": [x.strip() for x in symbolic_feature_pool.split(",") if x.strip()],
    "generated_specs_path": gen_specs_path,
    "generated_specs_exists": Path(gen_specs_path).is_file(),
    "symbolic_enabled": bool(int(enable_symbolic or 0)),
    "symbolic_spec_path": symbolic_spec_path,
    "symbolic_spec_exists": Path(symbolic_spec_path).is_file(),
    "dataset_context_path": dataset_context_path,
    "dataset_context_exists": Path(dataset_context_path).is_file(),
    "skip_train": bool(int(skip_train or 0)),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] generation_manifest={out}")
PY

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "[INFO] SKIP_TRAIN=1 -> generation stage completed, skipping downstream training."
  exit 0
fi

train_args=(
  --dataset "${DATASET}"
  --graph_data_dir "${GRAPH_DIR}"
  --em_table_root "${TABLE_ROOT}"
  --jts_table_root "${TABLE_ROOT}"
  --sm_table_root "${TABLE_ROOT}"
  --uts_table_root "${TABLE_ROOT}"
  --task_permutation "${TASK_PERMUTATION}"
  --limit_tasks 1
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
  --drop_cell_edges 1
  --em_pair_feat_norm 1
  --em_row_stats_mode "${EM_ROW_STATS_MODE}"
  --em_pair_cache_mode "${EM_PAIR_CACHE_MODE}"
  --feature_wiring_mode decoupled
  --allow_empty_decoder_static_features 1
  --em_pair_features "${EM_GROUPS}"
  --em_decoder_static_features "${EM_DECODER_GROUPS}"
  --em_generated_feature_specs_path "${EM_GENERATED_FEATURE_SPECS_PATH}"
  --jts_pair_features "${JTS_GROUPS}"
  --jts_decoder_static_features "${JTS_DECODER_GROUPS}"
  --jts_generated_feature_specs_path "${JTS_GENERATED_FEATURE_SPECS_PATH}"
  --sm_pair_features "${SM_GROUPS}"
  --sm_decoder_static_features "${SM_DECODER_GROUPS}"
  --sm_generated_feature_specs_path "${SM_GENERATED_FEATURE_SPECS_PATH}"
  --uts_pair_features "${UTS_GROUPS}"
  --uts_decoder_static_features "${UTS_DECODER_GROUPS}"
  --uts_generated_feature_specs_path "${UTS_GENERATED_FEATURE_SPECS_PATH}"
  --debug_max_train_edges "${DEBUG_MAX_TRAIN_EDGES}"
  --debug_max_val_edges "${DEBUG_MAX_VAL_EDGES}"
  --debug_max_test_edges "${DEBUG_MAX_TEST_EDGES}"
  --run_tag "${RUN_TAG}"
)

if [[ -n "${EM_PAIR_CACHE_ROOT}" ]]; then
  train_args+=(--em_pair_cache_root "${EM_PAIR_CACHE_ROOT}")
fi

if [[ "${ENABLE_SYMBOLIC}" == "1" ]]; then
  train_args+=(
    --online_symbolic_spec_path "${ONLINE_SYMBOLIC_SPEC_PATH}"
    --online_symbolic_repr "${SYMBOLIC_REPR}"
    --online_symbolic_normalize "${SYMBOLIC_NORMALIZE}"
    --online_symbolic_tile_repeat "${SYMBOLIC_TILE_REPEAT}"
  )
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${train_args[@]}" "$@"
