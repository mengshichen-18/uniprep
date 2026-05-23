#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${REPO_DIR}/scripts/run_online_3d4t_gnn_static_symbolic.sh"

if [[ ! -x "${RUNNER}" ]]; then
  echo "[ERROR] runner not found or not executable: ${RUNNER}"
  exit 1
fi

SPEC_ROOT="${SPEC_ROOT:-}"
DATASETS_CSV="${DATASETS_CSV:-magellan,santos_benchmark,wikidbs}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1}"
PARALLEL="${PARALLEL:-1}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
DIMS_CSV="${DIMS_CSV:-c4,c8,c12}"
TASKS_CSV="${TASKS_CSV:-em,jts,sm,uts}"

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

SYMBOLIC_REPR="${SYMBOLIC_REPR:-concat}"
SYMBOLIC_NORMALIZE="${SYMBOLIC_NORMALIZE:-zscore}"
SYMBOLIC_TILE_REPEAT="${SYMBOLIC_TILE_REPEAT:-1}"
STATIC_PRESET="${STATIC_PRESET:-full}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-seedmap_c4c8c12_3d4t}"
MAX_JOBS="${MAX_JOBS:-0}" # 0 => run all

# Locked decoder defaults for this experiment:
EM_DECODER_GROUPS="${EM_DECODER_GROUPS:-serial_value_alignment}"
SM_DECODER_GROUPS="${SM_DECODER_GROUPS:-value_stats}"
JTS_DECODER_GROUPS="${JTS_DECODER_GROUPS:-}"
UTS_DECODER_GROUPS="${UTS_DECODER_GROUPS:-}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_seedmap_c4c8c12_3d4t.sh [options]

Options:
  --spec-root <path>         default: ${SPEC_ROOT}
  --datasets <csv>           default: ${DATASETS_CSV}
  --gpu-ids <csv>            default: ${GPU_IDS_CSV}
  --parallel <0|1>           default: ${PARALLEL}
  --seeds <csv>              default: ${SEEDS_CSV}
  --dims <csv>               default: ${DIMS_CSV} (e.g. c4,c6,c8,c10,c12)
  --tasks <csv>              default: ${TASKS_CSV} (em,jts,sm,uts)

  --epochs <int>             default: ${EPOCHS}
  --patience <int>           default: ${PATIENCE}
  --batch-size <int>         default: ${BATCH_SIZE}
  --lr <float>               default: ${LR}
  --hidden-dim <int>         default: ${HIDDEN_DIM}
  --num-neighbors <csv>      default: ${NUM_NEIGHBORS}
  --gnn-layers <int>         default: ${GNN_LAYERS}
  --gnn-type <str>           default: ${GNN_TYPE}
  --num-workers <int>        default: ${NUM_WORKERS}
  --drop-cell-edges <0|1>    default: ${DROP_CELL_EDGES}

  --symbolic-repr <str>      default: ${SYMBOLIC_REPR}
  --symbolic-normalize <str> default: ${SYMBOLIC_NORMALIZE}
  --symbolic-tile-repeat <n> default: ${SYMBOLIC_TILE_REPEAT}
  --static-preset <mode>     default: ${STATIC_PRESET}
  --run-tag-prefix <str>     default: ${RUN_TAG_PREFIX}
  --max-jobs <int>           default: ${MAX_JOBS} (0 means all jobs)

  -h, --help                 show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec-root) SPEC_ROOT="$2"; shift 2 ;;
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --dims) DIMS_CSV="$2"; shift 2 ;;
    --tasks) TASKS_CSV="$2"; shift 2 ;;

    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
    --num-neighbors) NUM_NEIGHBORS="$2"; shift 2 ;;
    --gnn-layers) GNN_LAYERS="$2"; shift 2 ;;
    --gnn-type) GNN_TYPE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --drop-cell-edges) DROP_CELL_EDGES="$2"; shift 2 ;;

    --symbolic-repr) SYMBOLIC_REPR="$2"; shift 2 ;;
    --symbolic-normalize) SYMBOLIC_NORMALIZE="$2"; shift 2 ;;
    --symbolic-tile-repeat) SYMBOLIC_TILE_REPEAT="$2"; shift 2 ;;
    --static-preset) STATIC_PRESET="$2"; shift 2 ;;
    --run-tag-prefix) RUN_TAG_PREFIX="$2"; shift 2 ;;
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${SPEC_ROOT}" ]]; then
  echo "[ERROR] spec root not found: ${SPEC_ROOT}"
  exit 1
fi

IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
IFS=',' read -r -a DIMS <<< "${DIMS_CSV}"
IFS=',' read -r -a TASK_KEYS <<< "${TASKS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

task_name_of() {
  case "$1" in
    em) echo "entity_matching" ;;
    jts) echo "joinable_table_search" ;;
    sm) echo "schema_matching" ;;
    uts) echo "union_table_search" ;;
    *) return 1 ;;
  esac
}

task_perm_of() {
  case "$1" in
    em) echo "0" ;;
    jts) echo "1" ;;
    sm) echo "2" ;;
    uts) echo "3" ;;
    *) return 1 ;;
  esac
}

for t in "${TASK_KEYS[@]}"; do
  t="$(echo "${t}" | xargs)"
  if ! task_name_of "${t}" >/dev/null; then
    echo "[ERROR] invalid task key: ${t}"
    exit 1
  fi
done

for d in "${DIMS[@]}"; do
  d="$(echo "${d}" | xargs)"
  if [[ ! "${d}" =~ ^c[0-9]+$ ]]; then
    echo "[ERROR] invalid dim: ${d} (expected pattern c<number>, e.g. c6/c10)"
    exit 1
  fi
done

for s in "${SEEDS[@]}"; do
  s="$(echo "${s}" | xargs)"
  if [[ ! "${s}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] seed must be integer, got: ${s}"
    exit 1
  fi
done

# Preflight: fail-fast spec existence
for s in "${SEEDS[@]}"; do
  s="$(echo "${s}" | xargs)"
  cand=$((10#${s} + 1))
  cand2="$(printf '%02d' "${cand}")"
  for d in "${DIMS[@]}"; do
    d="$(echo "${d}" | xargs)"
    for t in "${TASK_KEYS[@]}"; do
      t="$(echo "${t}" | xargs)"
      task_name="$(task_name_of "${t}")"
      spec_path="${SPEC_ROOT}/${d}/${task_name}/cand_${cand2}.json"
      if [[ ! -f "${spec_path}" ]]; then
        echo "[ERROR] missing spec (fail-fast): ${spec_path}"
        exit 1
      fi
    done
  done
done

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_DIR}/outputs/online_symbolic_verify/${RUN_TAG_PREFIX}_${TS}"
LOG_DIR="${OUT_DIR}/runner_logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

cat > "${OUT_DIR}/config.env" <<EOF
SPEC_ROOT=${SPEC_ROOT}
DATASETS_CSV=${DATASETS_CSV}
GPU_IDS_CSV=${GPU_IDS_CSV}
PARALLEL=${PARALLEL}
SEEDS_CSV=${SEEDS_CSV}
DIMS_CSV=${DIMS_CSV}
TASKS_CSV=${TASKS_CSV}
EPOCHS=${EPOCHS}
PATIENCE=${PATIENCE}
BATCH_SIZE=${BATCH_SIZE}
LR=${LR}
HIDDEN_DIM=${HIDDEN_DIM}
NUM_NEIGHBORS=${NUM_NEIGHBORS}
GNN_LAYERS=${GNN_LAYERS}
GNN_TYPE=${GNN_TYPE}
NUM_WORKERS=${NUM_WORKERS}
DROP_CELL_EDGES=${DROP_CELL_EDGES}
SYMBOLIC_REPR=${SYMBOLIC_REPR}
SYMBOLIC_NORMALIZE=${SYMBOLIC_NORMALIZE}
SYMBOLIC_TILE_REPEAT=${SYMBOLIC_TILE_REPEAT}
STATIC_PRESET=${STATIC_PRESET}
RUN_TAG_PREFIX=${RUN_TAG_PREFIX}
EM_DECODER_GROUPS=${EM_DECODER_GROUPS}
SM_DECODER_GROUPS=${SM_DECODER_GROUPS}
JTS_DECODER_GROUPS=${JTS_DECODER_GROUPS}
UTS_DECODER_GROUPS=${UTS_DECODER_GROUPS}
EOF

MANIFEST="${OUT_DIR}/runs_manifest.csv"
echo "seed,cand,dim,task_key,task_name,task_permutation,spec_path,runner_log,run_log_dir,runner_status,runner_rc,epochs,patience,batch_size,lr,hidden_dim,num_neighbors,gnn_layers,gnn_type,num_workers,drop_cell_edges" > "${MANIFEST}"

echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] mapping: seed0->cand01, seed1->cand02, seed2->cand03, seed3->cand04, seed4->cand05"

job_idx=0
for s in "${SEEDS[@]}"; do
  s="$(echo "${s}" | xargs)"
  cand=$((10#${s} + 1))
  cand2="$(printf '%02d' "${cand}")"

  for d in "${DIMS[@]}"; do
    d="$(echo "${d}" | xargs)"
    for t in "${TASK_KEYS[@]}"; do
      t="$(echo "${t}" | xargs)"
      task_name="$(task_name_of "${t}")"
      task_perm="$(task_perm_of "${t}")"
      spec_path="${SPEC_ROOT}/${d}/${task_name}/cand_${cand2}.json"
      run_suffix="s${s}_${d}_${t}_cand${cand2}"
      run_tag="${RUN_TAG_PREFIX}_${run_suffix}"
      runner_log="${LOG_DIR}/${run_suffix}.log"
      run_log_dir=""
      runner_status="ok"
      runner_rc=0

      job_idx=$((job_idx + 1))
      if [[ "${MAX_JOBS}" =~ ^[0-9]+$ ]] && [[ "${MAX_JOBS}" -gt 0 ]] && [[ "${job_idx}" -gt "${MAX_JOBS}" ]]; then
        echo "[INFO] MAX_JOBS=${MAX_JOBS} reached, skipping remaining jobs."
        break 3
      fi

      echo "[RUN] seed=${s} cand=${cand2} dim=${d} task=${t} spec=$(basename "${spec_path}")"
      set +e
      bash "${RUNNER}" \
        --datasets "${DATASETS_CSV}" \
        --gpu-ids "${GPU_IDS_CSV}" \
        --parallel "${PARALLEL}" \
        --task-permutation "${task_perm}" \
        --limit-tasks 1 \
        --static-preset "${STATIC_PRESET}" \
        --symbolic on \
        --symbolic-suite 0 \
        --symbolic-template "${spec_path}" \
        --symbolic-repr "${SYMBOLIC_REPR}" \
        --symbolic-normalize "${SYMBOLIC_NORMALIZE}" \
        --symbolic-tile-repeat "${SYMBOLIC_TILE_REPEAT}" \
        --allow-empty-decoder-groups 1 \
        --em-decoder-groups "${EM_DECODER_GROUPS}" \
        --jts-decoder-groups "${JTS_DECODER_GROUPS}" \
        --sm-decoder-groups "${SM_DECODER_GROUPS}" \
        --uts-decoder-groups "${UTS_DECODER_GROUPS}" \
        --epochs "${EPOCHS}" \
        --patience "${PATIENCE}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --hidden-dim "${HIDDEN_DIM}" \
        --num-neighbors "${NUM_NEIGHBORS}" \
        --gnn-layers "${GNN_LAYERS}" \
        --gnn-type "${GNN_TYPE}" \
        --num-workers "${NUM_WORKERS}" \
        --drop-cell-edges "${DROP_CELL_EDGES}" \
        --seed "${s}" \
        --run-tag-prefix "${run_tag}" 2>&1 | tee "${runner_log}"
      runner_rc=$?
      set -e

      run_log_dir="$(grep -m1 'log_dir=' "${runner_log}" | sed 's/.*log_dir=//' || true)"
      if [[ "${runner_rc}" -ne 0 ]]; then
        runner_status="runner_failed"
      elif [[ -z "${run_log_dir}" || ! -d "${run_log_dir}" ]]; then
        runner_status="missing_run_log_dir"
      fi

      echo "${s},${cand2},${d},${t},${task_name},${task_perm},${spec_path},${runner_log},${run_log_dir},${runner_status},${runner_rc},${EPOCHS},${PATIENCE},${BATCH_SIZE},${LR},${HIDDEN_DIM},${NUM_NEIGHBORS},${GNN_LAYERS},${GNN_TYPE},${NUM_WORKERS},${DROP_CELL_EDGES}" >> "${MANIFEST}"
    done
  done
done

python - "${OUT_DIR}" "${DATASETS_CSV}" <<'PY'
import ast
import csv
import math
import pathlib
import re
import shlex
import statistics
import sys
from collections import defaultdict

out_dir = pathlib.Path(sys.argv[1]).resolve()
datasets = [x.strip() for x in sys.argv[2].split(",") if x.strip()]

manifest_path = out_dir / "runs_manifest.csv"
raw_csv = out_dir / "results_raw.csv"
summary_csv = out_dir / "results_summary.csv"
audit_csv = out_dir / "decoder_audit.csv"
pivot_md = out_dir / "results_pivot.md"
overview_md = out_dir / "summary.md"

re_val = re.compile(r"\[Validation\] selected threshold=([0-9.]+)\s+f1=([0-9.]+)")
re_test_thr = re.compile(r"Test Metrics \(threshold=([0-9.]+)\): (\{.*\})")
re_test_05 = re.compile(r"Test Metrics \(threshold=0\.5\): (\{.*\})")
re_cmd = re.compile(r"^\[CMD\]\s+(.*)$", re.MULTILINE)

expected_decoder = {
    "em_decoder_static_features": "serial_value_alignment",
    "jts_decoder_static_features": "",
    "sm_decoder_static_features": "value_stats",
    "uts_decoder_static_features": "",
}

def parse_cmd_value(cmd_text: str, flag_name: str) -> str:
    try:
        tokens = shlex.split(cmd_text)
    except Exception:
        return ""
    key = f"--{flag_name}"
    for idx, tok in enumerate(tokens):
        if tok == key:
            if idx + 1 < len(tokens):
                return tokens[idx + 1]
            return ""
    return ""

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan

def mean_std(vals):
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)

runs = []
with manifest_path.open("r", encoding="utf-8") as f:
    runs = list(csv.DictReader(f))

rows = []
audit_rows = []
for run in runs:
    run_status = run.get("runner_status", "")
    run_log_dir = pathlib.Path(run.get("run_log_dir", ""))
    for ds in datasets:
        row = {
            "seed": run.get("seed", ""),
            "cand": run.get("cand", ""),
            "dim": run.get("dim", ""),
            "task_key": run.get("task_key", ""),
            "task": run.get("task_name", ""),
            "dataset": ds,
            "spec_path": run.get("spec_path", ""),
            "runner_log": run.get("runner_log", ""),
            "run_log_dir": run.get("run_log_dir", ""),
            "dataset_log": "",
            "status": "",
            "threshold_valbest": "",
            "val_f1_best": "",
            "test_f1_valbest": "",
            "test_f1_at05": "",
            "em_decoder_static_features": "",
            "jts_decoder_static_features": "",
            "sm_decoder_static_features": "",
            "uts_decoder_static_features": "",
        }
        if run_status != "ok":
            row["status"] = run_status
            rows.append(row)
            continue

        log_path = run_log_dir / f"{ds}.log"
        row["dataset_log"] = str(log_path)
        if not log_path.is_file():
            row["status"] = "missing_dataset_log"
            rows.append(row)
            continue

        text = log_path.read_text(encoding="utf-8", errors="ignore")
        val_hits = re_val.findall(text)
        thr_hits = re_test_thr.findall(text)
        at05_hits = re_test_05.findall(text)
        cmd_match = re_cmd.search(text)
        cmd_text = cmd_match.group(1) if cmd_match else ""
        for k in expected_decoder:
            row[k] = parse_cmd_value(cmd_text, k)

        if cmd_text:
            audit_rows.append(
                {
                    "seed": row["seed"],
                    "cand": row["cand"],
                    "dim": row["dim"],
                    "task_key": row["task_key"],
                    "dataset": row["dataset"],
                    "em_decoder_static_features": row["em_decoder_static_features"],
                    "jts_decoder_static_features": row["jts_decoder_static_features"],
                    "sm_decoder_static_features": row["sm_decoder_static_features"],
                    "uts_decoder_static_features": row["uts_decoder_static_features"],
                }
            )

        if not val_hits or not thr_hits or not at05_hits:
            row["status"] = "missing_metrics"
            rows.append(row)
            continue

        threshold_valbest, val_f1_best = val_hits[-1]
        thr_pairs = []
        for thr_s, metrics_s in thr_hits:
            try:
                thr_f = float(thr_s)
            except Exception:
                continue
            try:
                parsed_metrics = ast.literal_eval(metrics_s)
            except Exception:
                continue
            if isinstance(parsed_metrics, dict):
                thr_pairs.append((thr_f, parsed_metrics))

        selected_thr = float(threshold_valbest)
        metrics = None
        for thr_f, m in thr_pairs:
            if abs(thr_f - selected_thr) < 1e-6:
                metrics = m
                break
        if metrics is None:
            non_05 = [m for thr_f, m in thr_pairs if abs(thr_f - 0.5) > 1e-9]
            if non_05:
                metrics = non_05[-1]
            elif thr_pairs:
                metrics = thr_pairs[-1][1]
            else:
                metrics = {}

        metrics05 = None
        for thr_f, m in thr_pairs:
            if abs(thr_f - 0.5) < 1e-9:
                metrics05 = m
                break
        if metrics05 is None:
            try:
                metrics05 = ast.literal_eval(at05_hits[-1])
            except Exception:
                metrics05 = {}

        row["threshold_valbest"] = float(threshold_valbest)
        row["val_f1_best"] = float(val_f1_best)
        row["test_f1_valbest"] = safe_float(metrics.get("link_f1", math.nan))
        row["test_f1_at05"] = safe_float(metrics05.get("link_f1", math.nan))
        row["status"] = "ok"
        rows.append(row)

raw_fields = [
    "seed",
    "cand",
    "dim",
    "task_key",
    "task",
    "dataset",
    "spec_path",
    "runner_log",
    "run_log_dir",
    "dataset_log",
    "status",
    "threshold_valbest",
    "val_f1_best",
    "test_f1_valbest",
    "test_f1_at05",
    "em_decoder_static_features",
    "jts_decoder_static_features",
    "sm_decoder_static_features",
    "uts_decoder_static_features",
]
with raw_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=raw_fields)
    w.writeheader()
    w.writerows(rows)

audit_fields = [
    "seed",
    "cand",
    "dim",
    "task_key",
    "dataset",
    "em_decoder_static_features",
    "jts_decoder_static_features",
    "sm_decoder_static_features",
    "uts_decoder_static_features",
]
with audit_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=audit_fields)
    w.writeheader()
    for r in audit_rows:
        w.writerow(r)

ok_rows = [r for r in rows if r["status"] == "ok"]
group_vals = defaultdict(lambda: {"valbest": [], "at05": []})
for r in ok_rows:
    key = (r["dataset"], r["task"], r["dim"])
    group_vals[key]["valbest"].append(float(r["test_f1_valbest"]))
    group_vals[key]["at05"].append(float(r["test_f1_at05"]))

summary_rows = []
for key in sorted(group_vals.keys()):
    ds, task, dim = key
    m1, s1 = mean_std(group_vals[key]["valbest"])
    m2, s2 = mean_std(group_vals[key]["at05"])
    summary_rows.append(
        {
            "dataset": ds,
            "task": task,
            "dim": dim,
            "n_ok": len(group_vals[key]["valbest"]),
            "valbest_mean": m1,
            "valbest_std": s1,
            "valbest_mean_std": f"{m1:.4f}±{s1:.4f}" if not math.isnan(m1) else "N/A",
            "at05_mean": m2,
            "at05_std": s2,
            "at05_mean_std": f"{m2:.4f}±{s2:.4f}" if not math.isnan(m2) else "N/A",
        }
    )

with summary_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "dataset",
            "task",
            "dim",
            "n_ok",
            "valbest_mean",
            "valbest_std",
            "valbest_mean_std",
            "at05_mean",
            "at05_std",
            "at05_mean_std",
        ],
    )
    w.writeheader()
    w.writerows(summary_rows)

# Pivot table: dataset x task, dim columns
dim_order = ["c4", "c8", "c12"]
task_order = ["entity_matching", "joinable_table_search", "schema_matching", "union_table_search"]
dataset_order = ["magellan", "santos_benchmark", "wikidbs"]
summary_map = {(r["dataset"], r["task"], r["dim"]): r for r in summary_rows}

lines = []
lines.append("# Seed→Candidate 对齐实验汇总")
lines.append("")
lines.append("- 映射: seed0->cand01, seed1->cand02, seed2->cand03, seed3->cand04, seed4->cand05")
lines.append(f"- 原始结果: `{raw_csv}`")
lines.append(f"- 汇总CSV: `{summary_csv}`")
lines.append("")
lines.append("## Mean±Std 大表（跨 seed0~4）")
lines.append("")
header = "| dataset | task | c4 valbest | c4 @0.5 | c8 valbest | c8 @0.5 | c12 valbest | c12 @0.5 |"
sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
lines.extend([header, sep])
for ds in dataset_order:
    for task in task_order:
        cells = []
        for dim in dim_order:
            item = summary_map.get((ds, task, dim))
            if item is None:
                cells.extend(["N/A", "N/A"])
            else:
                cells.extend([item["valbest_mean_std"], item["at05_mean_std"]])
        lines.append(f"| {ds} | {task} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} | {cells[5]} |")

# Optional seed-cand detail pivot
detail_map = defaultdict(dict)
for r in ok_rows:
    key = (r["dataset"], r["task"], int(r["seed"]), r["cand"])
    detail_map[key][r["dim"]] = (
        float(r["test_f1_valbest"]),
        float(r["test_f1_at05"]),
    )

lines.append("")
lines.append("## Seed-Cand 逐点位明细（按 dim 展开）")
lines.append("")
lines.append("| dataset | task | seed | cand | c4 valbest | c4 @0.5 | c8 valbest | c8 @0.5 | c12 valbest | c12 @0.5 |")
lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for ds in dataset_order:
    for task in task_order:
        for seed in [0, 1, 2, 3, 4]:
            cand = f"{seed + 1:02d}"
            key = (ds, task, seed, cand)
            row = detail_map.get(key, {})
            vals = []
            for dim in dim_order:
                if dim in row:
                    vals.append(f"{row[dim][0]:.4f}")
                    vals.append(f"{row[dim][1]:.4f}")
                else:
                    vals.append("N/A")
                    vals.append("N/A")
            lines.append(
                f"| {ds} | {task} | {seed} | {cand} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | {vals[5]} |"
            )

pivot_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

expected_rows = len(runs) * len(datasets)
actual_rows = len(rows)
ok_count = len(ok_rows)
failed_rows = [r for r in rows if r["status"] != "ok"]

# Decoder audit summary
decoder_mismatch = []
for r in ok_rows:
    for k, v in expected_decoder.items():
        actual = str(r.get(k, ""))
        if actual != v:
            decoder_mismatch.append(
                {
                    "dataset": r["dataset"],
                    "task_key": r["task_key"],
                    "seed": r["seed"],
                    "cand": r["cand"],
                    "dim": r["dim"],
                    "field": k,
                    "expected": v,
                    "actual": actual,
                    "log": r["dataset_log"],
                }
            )

ov = []
ov.append("# 运行总览")
ov.append("")
ov.append(f"- expected_rows: {expected_rows}")
ov.append(f"- actual_rows: {actual_rows}")
ov.append(f"- ok_rows: {ok_count}")
ov.append(f"- failed_rows: {len(failed_rows)}")
ov.append(f"- raw_csv: `{raw_csv}`")
ov.append(f"- summary_csv: `{summary_csv}`")
ov.append(f"- pivot_md: `{pivot_md}`")
ov.append(f"- decoder_audit_csv: `{audit_csv}`")
ov.append("")
ov.append("## Decoder 参数审计")
ov.append("")
ov.append(f"- expected: EM=serial_value_alignment, SM=value_stats, JTS='', UTS=''")
ov.append(f"- mismatches: {len(decoder_mismatch)}")
if decoder_mismatch:
    ov.append("")
    ov.append("| dataset | task | seed | cand | dim | field | expected | actual |")
    ov.append("| --- | --- | ---: | --- | --- | --- | --- | --- |")
    for r in decoder_mismatch[:50]:
        ov.append(
            f"| {r['dataset']} | {r['task_key']} | {r['seed']} | {r['cand']} | {r['dim']} | {r['field']} | {r['expected']} | {r['actual']} |"
        )
    if len(decoder_mismatch) > 50:
        ov.append(f"- ... and {len(decoder_mismatch) - 50} more mismatches")

if failed_rows:
    ov.append("")
    ov.append("## 失败清单（前100行）")
    ov.append("")
    ov.append("| seed | cand | dim | dataset | task | status | log |")
    ov.append("| ---: | --- | --- | --- | --- | --- | --- |")
    for r in failed_rows[:100]:
        ov.append(
            f"| {r['seed']} | {r['cand']} | {r['dim']} | {r['dataset']} | {r['task_key']} | {r['status']} | {r['dataset_log'] or r['runner_log']} |"
        )
    if len(failed_rows) > 100:
        ov.append(f"- ... and {len(failed_rows) - 100} more")

overview_md.write_text("\n".join(ov) + "\n", encoding="utf-8")
print(f"[DONE] raw={raw_csv}")
print(f"[DONE] summary={summary_csv}")
print(f"[DONE] pivot={pivot_md}")
print(f"[DONE] overview={overview_md}")

if actual_rows != expected_rows:
    print(f"[WARN] row_count mismatch: expected={expected_rows} actual={actual_rows}")
    sys.exit(2)
if failed_rows:
    print(f"[WARN] failed rows: {len(failed_rows)}")
    sys.exit(3)
PY

echo "[DONE] seedmap pipeline finished. out_dir=${OUT_DIR}"
