#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="${REPO_DIR}/scripts/run_online_3d4t_gnn_static_symbolic.sh"
OUT_ROOT="${REPO_DIR}/outputs/online_symbolic_verify"
STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${OUT_ROOT}/seed0to4_jts_uts_gnnonly_fix_${STAMP}"
MANIFEST="${WORK_DIR}/manifest.tsv"
RESULT_JU_CSV="${WORK_DIR}/results_jts_uts_30rows.csv"
RESULT_MERGED_CSV="${WORK_DIR}/results_45rows_merged.csv"
SUMMARY_MD="${WORK_DIR}/summary_3tasks.md"

mkdir -p "${WORK_DIR}"
echo -e "seed\ttask\tdataset\tstatus\trun_dir" > "${MANIFEST}"

DATASETS="magellan,santos_benchmark,wikidbs"

# 默认从最近一次三任务结果里提 EM（本轮不重跑 EM）
BASE_EM_CSV_DEFAULT="$(ls -1dt "${OUT_ROOT}"/seed0to4_gnn_em_jts_uts_no_symbolic_*/results_45rows.csv 2>/dev/null | head -n1 || true)"
BASE_EM_CSV="${BASE_EM_CSV:-${BASE_EM_CSV_DEFAULT}}"

if [[ -z "${BASE_EM_CSV}" || ! -f "${BASE_EM_CSV}" ]]; then
  echo "[ERROR] BASE_EM_CSV not found. Please set BASE_EM_CSV=/abs/path/results_45rows.csv"
  exit 1
fi

run_one() {
  local seed="$1"
  local task_name="$2"
  local perm="$3"
  local gpu="$4"
  local prefix="seed${seed}_${task_name}_gnnonly_fix"

  (
    cd "${REPO_DIR}"
    bash "${RUNNER}" \
      --run-tag-prefix "${prefix}" \
      --datasets "${DATASETS}" \
      --gpu-ids "${gpu}" \
      --parallel 0 \
      --seed "${seed}" \
      --task-permutation "${perm}" \
      --limit-tasks 1 \
      --symbolic off \
      --num-workers 0 \
      --allow-empty-decoder-groups 1 \
      --feature-wiring-mode decoupled \
      --jts-decoder-groups "" \
      --uts-decoder-groups ""
  )

  local run_dir
  run_dir="$(ls -1dt "${OUT_ROOT}/${prefix}_"* 2>/dev/null | head -n1 || true)"
  if [[ -z "${run_dir}" ]]; then
    echo -e "${seed}\t${task_name}\t-\tmissing_run_dir\t-" >> "${MANIFEST}"
    return 1
  fi

  local ds
  IFS=',' read -r -a ds_arr <<< "${DATASETS}"
  for ds in "${ds_arr[@]}"; do
    if [[ -f "${run_dir}/${ds}.log" ]] && tail -n 3 "${run_dir}/${ds}.log" | rg -q "All tasks completed\\."; then
      echo -e "${seed}\t${task_name}\t${ds}\tok\t${run_dir}" >> "${MANIFEST}"
    else
      echo -e "${seed}\t${task_name}\t${ds}\tlog_incomplete\t${run_dir}" >> "${MANIFEST}"
    fi
  done
}

# task_permutation: 1->JTS first, 3->UTS first
for seed in 0 1 2 3 4; do
  echo "[BATCH] seed=${seed} start"
  run_one "${seed}" "jts" 1 0 &
  pid_jts=$!
  run_one "${seed}" "uts" 3 1 &
  pid_uts=$!
  wait "${pid_jts}"
  wait "${pid_uts}"
  echo "[BATCH] seed=${seed} done"
done

python - <<'PY' "${MANIFEST}" "${RESULT_JU_CSV}" "${BASE_EM_CSV}" "${RESULT_MERGED_CSV}" "${SUMMARY_MD}"
import csv
import re
import sys
from pathlib import Path
from statistics import mean, stdev

manifest = Path(sys.argv[1])
out_ju = Path(sys.argv[2])
base_em_csv = Path(sys.argv[3])
out_merged = Path(sys.argv[4])
summary_md = Path(sys.argv[5])

task_map = {"jts": "joinable_table_search", "uts": "union_table_search"}

rows_ju = []
with manifest.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        if r["status"] != "ok":
            continue
        seed = int(r["seed"])
        task_short = r["task"]
        dataset = r["dataset"]
        log_path = Path(r["run_dir"]) / f"{dataset}.log"
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        m_thr = re.findall(r"\[Validation\] selected threshold=([0-9.]+)", text)
        thr = float(m_thr[-1]) if m_thr else float("nan")
        m_metrics = re.findall(r"Test Metrics \(threshold=[0-9.]+\): \{[^\n]*'link_f1': ([0-9.]+)\}", text)
        if len(m_metrics) >= 2:
            f1_valbest = float(m_metrics[-2])
            f1_at05 = float(m_metrics[-1])
        elif len(m_metrics) == 1:
            f1_valbest = float(m_metrics[-1])
            f1_at05 = float("nan")
        else:
            f1_valbest = float("nan")
            f1_at05 = float("nan")
        rows_ju.append(
            {
                "seed": seed,
                "task": task_map[task_short],
                "dataset": dataset,
                "test_f1_valbest": f1_valbest,
                "test_f1_at05": f1_at05,
                "threshold_valbest": thr,
                "run_dir": r["run_dir"],
            }
        )

rows_ju.sort(key=lambda x: (x["seed"], x["task"], x["dataset"]))
fields = ["seed", "task", "dataset", "test_f1_valbest", "test_f1_at05", "threshold_valbest", "run_dir"]
with out_ju.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_ju)

# merge with EM rows from previous table
rows_em = []
with base_em_csv.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get("task") == "entity_matching":
            rows_em.append(
                {
                    "seed": int(r["seed"]),
                    "task": r["task"],
                    "dataset": r["dataset"],
                    "test_f1_valbest": float(r["test_f1_valbest"]),
                    "test_f1_at05": float(r["test_f1_at05"]),
                    "threshold_valbest": float(r["threshold_valbest"]),
                    "run_dir": r.get("run_dir", str(base_em_csv)),
                }
            )

rows_all = rows_em + rows_ju
rows_all.sort(key=lambda x: (x["seed"], x["task"], x["dataset"]))
with out_merged.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_all)

def ms(vals):
    if len(vals) <= 1:
        return f"{vals[0]:.4f}±0.0000" if vals else "N/A"
    return f"{mean(vals):.4f}±{stdev(vals):.4f}"

group = {}
for r in rows_all:
    k = (r["dataset"], r["task"])
    group.setdefault(k, []).append(r["test_f1_valbest"])

with summary_md.open("w", encoding="utf-8") as f:
    f.write("# Seed0-4 三任务汇总（JTS/UTS 已按 gnn_only 重跑）\n\n")
    f.write(f"- EM 来源（未重跑）：`{base_em_csv}`\n")
    f.write("- JTS/UTS 口径：`--allow-empty-decoder-groups 1 --jts-decoder-groups \"\" --uts-decoder-groups \"\"`\n")
    f.write("- 任务：entity_matching / joinable_table_search / union_table_search\n\n")
    f.write("## Mean±Std (5 seeds)\n\n")
    f.write("| dataset | task | test_f1(valbest) mean±std |\n")
    f.write("| --- | --- | ---: |\n")
    for (dataset, task), vals in sorted(group.items()):
        f.write(f"| {dataset} | {task} | {ms(vals)} |\n")
    f.write("\n## Full 45 Rows\n\n")
    f.write("| seed | dataset | task | test_f1(valbest) | test_f1@0.5 | threshold_valbest |\n")
    f.write("| --- | --- | --- | ---: | ---: | ---: |\n")
    for r in rows_all:
        f.write(
            f"| {r['seed']} | {r['dataset']} | {r['task']} | "
            f"{r['test_f1_valbest']:.4f} | {r['test_f1_at05']:.4f} | {r['threshold_valbest']:.4f} |\n"
        )

print(f"RESULT_JTS_UTS={out_ju}")
print(f"RESULT_MERGED={out_merged}")
print(f"SUMMARY_MD={summary_md}")
print(f"N_JTS_UTS_ROWS={len(rows_ju)}")
print(f"N_MERGED_ROWS={len(rows_all)}")
PY

echo "WORK_DIR=${WORK_DIR}"
echo "MANIFEST=${MANIFEST}"
echo "RESULT_JTS_UTS=${RESULT_JU_CSV}"
echo "RESULT_MERGED=${RESULT_MERGED_CSV}"
echo "SUMMARY_MD=${SUMMARY_MD}"
