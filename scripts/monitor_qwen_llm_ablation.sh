#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-}"
LOG_ROOT="${LOG_ROOT:-}"
OUT_LOG="${OUT_LOG:-}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"

if [[ -z "${RUN_ROOT}" || -z "${LOG_ROOT}" || -z "${OUT_LOG}" ]]; then
  echo "[ERROR] RUN_ROOT, LOG_ROOT, and OUT_LOG are required." >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT_LOG}")"

interesting_rg='generated_atoms\.json|generating symbolic spec|\{"ok":|AuthenticationError|invalid_api_key|APITimeoutError|Request timed out|All tasks completed\.|Test Metrics \(threshold='

while true; do
  {
    echo "=== $(date '+%F %T') ==="
    echo "[run_root] ${RUN_ROOT}"
    echo "[log_root] ${LOG_ROOT}"
    echo "[files]"
    find "${RUN_ROOT}" -type f | sort || true
    echo "[log_events]"
    rg -n "${interesting_rg}" "${LOG_ROOT}" -g '*.log' || true
    echo
  } >> "${OUT_LOG}"
  sleep "${INTERVAL_SEC}"
done
