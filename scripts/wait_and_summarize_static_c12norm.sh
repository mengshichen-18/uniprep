#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

STATIC_PREFIX=""
C12_PREFIX=""
EXPECT_LOGS=3
POLL_SEC=20
OUT_DIR=""

usage() {
  cat <<EOF
Usage:
  bash scripts/wait_and_summarize_static_c12norm.sh \\
    --static-prefix <prefix> \\
    --c12-prefix <prefix> \\
    [--expect-logs <n>] [--poll-sec <n>] [--out-dir <path>]

This script waits until the latest run dirs of both prefixes have all logs
containing "All tasks completed", then emits comparison summary.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --static-prefix) STATIC_PREFIX="$2"; shift 2 ;;
    --c12-prefix) C12_PREFIX="$2"; shift 2 ;;
    --expect-logs) EXPECT_LOGS="$2"; shift 2 ;;
    --poll-sec) POLL_SEC="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${STATIC_PREFIX}" || -z "${C12_PREFIX}" ]]; then
  echo "[ERROR] --static-prefix and --c12-prefix are required"
  exit 1
fi

latest_dir() {
  local prefix="$1"
  ls -1dt "${REPO_DIR}/outputs/online_symbolic_verify/${prefix}_"* 2>/dev/null | head -n 1 || true
}

is_done_dir() {
  local dir="$1"
  local expect="$2"
  [[ -d "${dir}" ]] || return 1
  local logs
  logs=$(find "${dir}" -maxdepth 1 -type f -name '*.log' | wc -l | tr -d ' ')
  [[ "${logs}" -ge "${expect}" ]] || return 1
  local done_count
  done_count=$( (grep -h "All tasks completed" "${dir}"/*.log 2>/dev/null || true) | wc -l | tr -d ' ')
  [[ "${done_count}" -ge "${expect}" ]]
}

echo "[WAIT] static_prefix=${STATIC_PREFIX}"
echo "[WAIT] c12_prefix=${C12_PREFIX}"
echo "[WAIT] expect_logs=${EXPECT_LOGS}"

while true; do
  STATIC_DIR="$(latest_dir "${STATIC_PREFIX}")"
  C12_DIR="$(latest_dir "${C12_PREFIX}")"
  echo "[POLL] static_dir=${STATIC_DIR:-N/A}"
  echo "[POLL] c12_dir=${C12_DIR:-N/A}"
  if [[ -n "${STATIC_DIR}" && -d "${STATIC_DIR}" ]]; then
    s_logs=$(find "${STATIC_DIR}" -maxdepth 1 -type f -name '*.log' | wc -l | tr -d ' ')
    s_done=$( (grep -h "All tasks completed" "${STATIC_DIR}"/*.log 2>/dev/null || true) | wc -l | tr -d ' ')
    echo "[POLL] static_progress=${s_done}/${s_logs}"
  fi
  if [[ -n "${C12_DIR}" && -d "${C12_DIR}" ]]; then
    c_logs=$(find "${C12_DIR}" -maxdepth 1 -type f -name '*.log' | wc -l | tr -d ' ')
    c_done=$( (grep -h "All tasks completed" "${C12_DIR}"/*.log 2>/dev/null || true) | wc -l | tr -d ' ')
    echo "[POLL] c12_progress=${c_done}/${c_logs}"
  fi
  if [[ -n "${STATIC_DIR}" && -n "${C12_DIR}" ]] \
    && is_done_dir "${STATIC_DIR}" "${EXPECT_LOGS}" \
    && is_done_dir "${C12_DIR}" "${EXPECT_LOGS}"; then
    break
  fi
  sleep "${POLL_SEC}"
done

if [[ -z "${OUT_DIR}" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="${REPO_DIR}/outputs/online_symbolic_verify/summaries/${STATIC_PREFIX}_vs_${C12_PREFIX}_${stamp}"
fi

mkdir -p "${OUT_DIR}"
"${PYTHON_BIN}" "${REPO_DIR}/scripts/summarize_static_vs_c12norm.py" \
  --static-dir "${STATIC_DIR}" \
  --c12-dir "${C12_DIR}" \
  --output-dir "${OUT_DIR}"

echo "[DONE] summary: ${OUT_DIR}"
