#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="${REPO_DIR}/scripts/run_online_3d4t_gnn_static_symbolic.sh"
OUT_ROOT="${REPO_DIR}/outputs/online_symbolic_verify"
STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${OUT_ROOT}/seed0to4_gnn_em_jts_uts_no_symbolic_${STAMP}"
MANIFEST="${WORK_DIR}/manifest.tsv"
RESULT_CSV="${WORK_DIR}/results_45rows.csv"
SUMMARY_MD="${WORK_DIR}/summary.md"

mkdir -p "${WORK_DIR}"
echo -e "seed\ttask\tdataset\tstatus\trun_dir" > "${MANIFEST}"

DATASETS="magellan,santos_benchmark,wikidbs"

run_one() {
  local seed="$1"
  local task_name="$2"
  local perm="$3"
  local gpu="$4"
  local prefix="seed${seed}_${task_name}_gnn_nosym"

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
      --num-workers 0
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

# TASK mapping via v3c permutation:
# perm0->EM first, perm1->JTS first, perm3->UTS first
for seed in 0 1 2 3 4; do
  echo "[BATCH] seed=${seed} start"

  run_one "${seed}" "em" 0 0 &
  pid_em=$!
  run_one "${seed}" "jts" 1 1 &
  pid_jts=$!

  wait "${pid_em}"
  wait "${pid_jts}"

  run_one "${seed}" "uts" 3 1

  echo "[BATCH] seed=${seed} done"
done

python - <<'PY' "${MANIFEST}" "${RESULT_CSV}" "${SUMMARY_MD}"
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

manifest = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
summary_md = Path(sys.argv[3])

task_map = {
    "em": "entity_matching",
    "jts": "joinable_table_search",
    "uts": "union_table_search",
}

rows = []
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
        threshold = float(m_thr[-1]) if m_thr else float("nan")

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

        rows.append(
            {
                "seed": seed,
                "task": task_map.get(task_short, task_short),
                "dataset": dataset,
                "test_f1_valbest": f1_valbest,
                "test_f1_at05": f1_at05,
                "threshold_valbest": threshold,
                "run_dir": r["run_dir"],
            }
        )

rows.sort(key=lambda x: (x["seed"], x["task"], x["dataset"]))
fields = ["seed", "task", "dataset", "test_f1_valbest", "test_f1_at05", "threshold_valbest", "run_dir"]
with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

# aggregate mean±std for valbest by dataset-task
group = {}
for r in rows:
    k = (r["dataset"], r["task"])
    group.setdefault(k, []).append(r["test_f1_valbest"])

def fmt_ms(vals):
    if not vals:
        return "N/A"
    if len(vals) == 1:
        return f"{vals[0]:.4f}±0.0000"
    return f"{mean(vals):.4f}±{stdev(vals):.4f}"

table_rows = []
for (dataset, task), vals in sorted(group.items()):
    table_rows.append((dataset, task, fmt_ms(vals)))

with summary_md.open("w", encoding="utf-8") as f:
    f.write("# Seed0-4 GNN(no symbolic) Summary\n\n")
    f.write("- Config: EM_DECODER_GROUPS=serial_value_alignment, SM_DECODER_GROUPS=value_stats (SM not rerun)\n")
    f.write("- Tasks rerun: entity_matching, joinable_table_search, union_table_search\n")
    f.write("- Metric: test_f1(valbest), with test_f1@0.5 also kept in csv\n\n")
    f.write("## Mean±Std (5 seeds)\n\n")
    f.write("| dataset | task | test_f1(valbest) mean±std |\n")
    f.write("| --- | --- | ---: |\n")
    for dataset, task, val in table_rows:
        f.write(f"| {dataset} | {task} | {val} |\n")
    f.write("\n## Full 45 Rows\n\n")
    f.write("| seed | dataset | task | test_f1(valbest) | test_f1@0.5 | threshold_valbest |\n")
    f.write("| --- | --- | --- | ---: | ---: | ---: |\n")
    for r in rows:
        f.write(
            f"| {r['seed']} | {r['dataset']} | {r['task']} | "
            f"{r['test_f1_valbest']:.4f} | {r['test_f1_at05']:.4f} | {r['threshold_valbest']:.4f} |\n"
        )

print(f"RESULT_CSV={out_csv}")
print(f"SUMMARY_MD={summary_md}")
print(f"N_ROWS={len(rows)}")
PY

echo "WORK_DIR=${WORK_DIR}"
echo "MANIFEST=${MANIFEST}"
echo "RESULT_CSV=${RESULT_CSV}"
echo "SUMMARY_MD=${SUMMARY_MD}"
