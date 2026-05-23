#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${REPO_DIR}/scripts/run_online_3d4t_gnn_static_symbolic.sh"

if [[ ! -x "${RUNNER}" ]]; then
  echo "[ERROR] runner not found or not executable: ${RUNNER}"
  exit 1
fi

DATASETS_CSV="${DATASETS_CSV:-magellan,santos_benchmark,wikidbs}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0}"
PARALLEL="${PARALLEL:-0}"
TASKS_CSV="${TASKS_CSV:-jts,sm,uts}"  # supported: em,jts,sm,uts
SYMBOLIC_DIMS_CSV="${SYMBOLIC_DIMS_CSV:-c4,c8,c12}"
SYMBOLIC_CANDS_CSV="${SYMBOLIC_CANDS_CSV:-01,02,03,04,05}"
EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-192}"
SEED="${SEED:-0}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-online_symbolic_c4c8c12_3d3t}"

SPEC_ROOT="${SPEC_ROOT:-}"
if [[ -z "${SPEC_ROOT}" ]]; then
  SPEC_ROOT="$(ls -td "${REPO_DIR}"/symbolic_specs/batches/v3_tasklevel_nocontext_trainhint_gpt5_* 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${SPEC_ROOT}" || ! -d "${SPEC_ROOT}" ]]; then
  echo "[ERROR] cannot resolve SPEC_ROOT. Set SPEC_ROOT=/abs/path/to/spec_batch."
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_DIR}/outputs/online_symbolic_verify/${RUN_TAG_PREFIX}_${TS}"
mkdir -p "${OUT_DIR}"

declare -A TASK_TO_PERM=(
  ["em"]="0"
  ["jts"]="1"
  ["sm"]="2"
  ["uts"]="3"
)
declare -A TASK_TO_NAME=(
  ["em"]="entity_matching"
  ["jts"]="joinable_table_search"
  ["sm"]="schema_matching"
  ["uts"]="union_table_search"
)

echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] datasets=${DATASETS_CSV}"
echo "[INFO] tasks=${TASKS_CSV}"
echo "[INFO] dims=${SYMBOLIC_DIMS_CSV} cands=${SYMBOLIC_CANDS_CSV}"
echo "[INFO] spec_root=${SPEC_ROOT}"
echo "task_key,task_name,run_log_dir" > "${OUT_DIR}/run_dirs.csv"

IFS=',' read -r -a TASK_KEYS <<< "${TASKS_CSV}"
for raw_key in "${TASK_KEYS[@]}"; do
  key="$(echo "${raw_key}" | xargs)"
  [[ -z "${key}" ]] && continue
  perm="${TASK_TO_PERM[$key]:-}"
  task_name="${TASK_TO_NAME[$key]:-}"
  if [[ -z "${perm}" || -z "${task_name}" ]]; then
    echo "[ERROR] unsupported task key: ${key} (supported: em,jts,sm,uts)"
    exit 1
  fi

  task_runner_log="${OUT_DIR}/run_${key}.log"
  echo "[INFO] running task=${key}(${task_name}) permutation=${perm}"
  bash "${RUNNER}" \
    --datasets "${DATASETS_CSV}" \
    --gpu-ids "${GPU_IDS_CSV}" \
    --parallel "${PARALLEL}" \
    --task-permutation "${perm}" \
    --limit-tasks 1 \
    --static-preset full \
    --symbolic on \
    --symbolic-suite 1 \
    --symbolic-dims "${SYMBOLIC_DIMS_CSV}" \
    --symbolic-cands "${SYMBOLIC_CANDS_CSV}" \
    --symbolic-spec-root-c124 "${SPEC_ROOT}" \
    --symbolic-spec-root-c8c12 "${SPEC_ROOT}" \
    --symbolic-repr concat \
    --symbolic-normalize zscore \
    --symbolic-tile-repeat 1 \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --run-tag-prefix "${RUN_TAG_PREFIX}_${key}" 2>&1 | tee "${task_runner_log}"

  run_log_dir="$(grep -m1 'log_dir=' "${task_runner_log}" | sed 's/.*log_dir=//')"
  if [[ -z "${run_log_dir}" || ! -d "${run_log_dir}" ]]; then
    echo "[ERROR] failed to parse run_log_dir for task=${key} from ${task_runner_log}"
    exit 1
  fi
  echo "${key},${task_name},${run_log_dir}" >> "${OUT_DIR}/run_dirs.csv"
done

python - "${OUT_DIR}" <<'PY'
import ast
import csv
import math
import pathlib
import re
import statistics
import sys

out_dir = pathlib.Path(sys.argv[1]).resolve()
run_dirs_csv = out_dir / "run_dirs.csv"
raw_csv = out_dir / "results_raw.csv"
point_csv = out_dir / "results_point_mean_std.csv"
macro_cand_csv = out_dir / "results_macro_by_candidate.csv"
macro_dim_csv = out_dir / "results_macro_mean_std.csv"
summary_md = out_dir / "summary.md"

re_val = re.compile(r"\[Validation\] selected threshold=([0-9.]+)\s+f1=([0-9.]+)")
re_test_thr = re.compile(r"Test Metrics \(threshold=([0-9.]+)\): (\{.*\})")
re_test_05 = re.compile(r"Test Metrics \(threshold=0\.5\): (\{.*\})")

rows = []
with run_dirs_csv.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    task_entries = list(reader)

for item in task_entries:
    task_key = item["task_key"]
    task_name = item["task_name"]
    run_dir = pathlib.Path(item["run_log_dir"]).resolve()
    if not run_dir.is_dir():
        continue
    for variant_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        variant = variant_dir.name
        m = re.match(r"^(c[0-9]+)_cand([0-9]{2})$", variant)
        if not m:
            continue
        dim = m.group(1)
        cand = m.group(2)
        for log_path in sorted(variant_dir.glob("*.log")):
            dataset = log_path.stem
            text = log_path.read_text(encoding="utf-8", errors="ignore")

            val_hits = re_val.findall(text)
            thr_hits = re_test_thr.findall(text)
            at05_hits = re_test_05.findall(text)
            if not val_hits or not thr_hits or not at05_hits:
                rows.append(
                    {
                        "task_key": task_key,
                        "task": task_name,
                        "dataset": dataset,
                        "variant": variant,
                        "dim": dim,
                        "cand": cand,
                        "status": "missing_metrics",
                        "threshold_valbest": "",
                        "val_f1_best": "",
                        "test_f1_valbest": "",
                        "test_f1_at05": "",
                        "log_path": str(log_path),
                    }
                )
                continue

            threshold_valbest, val_f1_best = val_hits[-1]
            thr_val, metrics_str = thr_hits[-1]
            metrics = ast.literal_eval(metrics_str)
            test_f1_valbest = float(metrics.get("link_f1", float("nan")))
            metrics05 = ast.literal_eval(at05_hits[-1])
            test_f1_at05 = float(metrics05.get("link_f1", float("nan")))

            rows.append(
                {
                    "task_key": task_key,
                    "task": task_name,
                    "dataset": dataset,
                    "variant": variant,
                    "dim": dim,
                    "cand": cand,
                    "status": "ok",
                    "threshold_valbest": float(threshold_valbest),
                    "val_f1_best": float(val_f1_best),
                    "test_f1_valbest": test_f1_valbest,
                    "test_f1_at05": test_f1_at05,
                    "log_path": str(log_path),
                }
            )

raw_fields = [
    "task_key",
    "task",
    "dataset",
    "variant",
    "dim",
    "cand",
    "status",
    "threshold_valbest",
    "val_f1_best",
    "test_f1_valbest",
    "test_f1_at05",
    "log_path",
]
with raw_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=raw_fields)
    w.writeheader()
    w.writerows(rows)

ok_rows = [r for r in rows if r["status"] == "ok"]

def mean_std(vals):
    if not vals:
        return (float("nan"), float("nan"))
    if len(vals) == 1:
        return (vals[0], 0.0)
    return (statistics.mean(vals), statistics.stdev(vals))

# point-wise mean/std: (dataset,task,dim) over 5 candidates
point_map = {}
for r in ok_rows:
    k = (r["dataset"], r["task"], r["dim"])
    point_map.setdefault(k, []).append(float(r["test_f1_valbest"]))

point_rows = []
for (dataset, task, dim), vals in sorted(point_map.items()):
    m, s = mean_std(vals)
    point_rows.append(
        {
            "dataset": dataset,
            "task": task,
            "dim": dim,
            "n": len(vals),
            "test_f1_valbest_mean": m,
            "test_f1_valbest_std": s,
            "mean_std": f"{m:.4f}±{s:.4f}",
        }
    )
with point_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "dataset",
            "task",
            "dim",
            "n",
            "test_f1_valbest_mean",
            "test_f1_valbest_std",
            "mean_std",
        ],
    )
    w.writeheader()
    w.writerows(point_rows)

# macro by candidate: each (dim,cand) over all dataset-task points
macro_map = {}
for r in ok_rows:
    k = (r["dim"], r["cand"])
    macro_map.setdefault(k, []).append(float(r["test_f1_valbest"]))

macro_rows = []
for (dim, cand), vals in sorted(macro_map.items()):
    macro_rows.append(
        {
            "dim": dim,
            "cand": cand,
            "n_points": len(vals),
            "macro_test_f1_valbest": statistics.mean(vals) if vals else float("nan"),
        }
    )
with macro_cand_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dim", "cand", "n_points", "macro_test_f1_valbest"])
    w.writeheader()
    w.writerows(macro_rows)

# macro mean/std over cands for each dim
dim_map = {}
for r in macro_rows:
    dim_map.setdefault(r["dim"], []).append(float(r["macro_test_f1_valbest"]))

macro_dim_rows = []
for dim, vals in sorted(dim_map.items()):
    m, s = mean_std(vals)
    macro_dim_rows.append(
        {
            "dim": dim,
            "n_cands": len(vals),
            "macro_mean": m,
            "macro_std": s,
            "mean_std": f"{m:.4f}±{s:.4f}",
        }
    )
with macro_dim_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dim", "n_cands", "macro_mean", "macro_std", "mean_std"])
    w.writeheader()
    w.writerows(macro_dim_rows)

def md_table(headers, rows2d):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows2d:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)

point_headers = ["dataset", "task", "c4", "c8", "c12"]
point_rows_md = []
tasks_order = [
    "joinable_table_search",
    "schema_matching",
    "union_table_search",
    "entity_matching",
]
datasets_order = ["magellan", "santos_benchmark", "wikidbs"]
point_lookup = {(r["dataset"], r["task"], r["dim"]): r["mean_std"] for r in point_rows}
for ds in datasets_order:
    for tk in tasks_order:
        if (ds, tk, "c4") in point_lookup or (ds, tk, "c8") in point_lookup or (ds, tk, "c12") in point_lookup:
            point_rows_md.append(
                [
                    ds,
                    tk,
                    point_lookup.get((ds, tk, "c4"), "N/A"),
                    point_lookup.get((ds, tk, "c8"), "N/A"),
                    point_lookup.get((ds, tk, "c12"), "N/A"),
                ]
            )

macro_headers = ["dim", "macro(valbest) mean±std"]
macro_rows_md = [[r["dim"], r["mean_std"]] for r in macro_dim_rows]

summary_lines = []
summary_lines.append("# Symbolic-only C4/C8/C12 (3D3T) Summary")
summary_lines.append("")
summary_lines.append(f"- run_dirs: `{run_dirs_csv}`")
summary_lines.append(f"- raw rows: `{raw_csv}`")
summary_lines.append(f"- point mean/std: `{point_csv}`")
summary_lines.append(f"- macro by cand: `{macro_cand_csv}`")
summary_lines.append(f"- macro mean/std: `{macro_dim_csv}`")
summary_lines.append("")
summary_lines.append("## 9-point Table (test_f1(valbest), mean±std over 5 cands)")
summary_lines.append(md_table(point_headers, point_rows_md))
summary_lines.append("")
summary_lines.append("## Macro (over all available points)")
summary_lines.append(md_table(macro_headers, macro_rows_md))
summary_lines.append("")
summary_md.write_text("\n".join(summary_lines), encoding="utf-8")

print(f"[DONE] wrote: {raw_csv}")
print(f"[DONE] wrote: {point_csv}")
print(f"[DONE] wrote: {macro_cand_csv}")
print(f"[DONE] wrote: {macro_dim_csv}")
print(f"[DONE] wrote: {summary_md}")
PY

echo "[DONE] all runs + summary complete."
echo "[DONE] out_dir=${OUT_DIR}"
