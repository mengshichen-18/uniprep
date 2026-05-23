#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for generating v2-style symbolic specs with the v3 prompt.
# Example:
#   bash scripts/run_generate_symbolic_spec_v3.sh \
#     --task entity_matching \
#     --feature-pool-file /path/to/feature_pool.json \
#     --output /path/to/spec.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GEN_SCRIPT="${REPO_DIR}/scripts/generate_symbolic_spec_gpt5_v3.py"

DEFAULT_MODEL="${DEFAULT_MODEL:-gpt-5-mini}"
DEFAULT_REASONING_EFFORT="${DEFAULT_REASONING_EFFORT:-low}"
DEFAULT_TIMEOUT_SEC="${DEFAULT_TIMEOUT_SEC:-120}"
DEFAULT_MAX_COMPLETION_TOKENS="${DEFAULT_MAX_COMPLETION_TOKENS:-1800}"
DEFAULT_FEATURE_CARDS_MAX="${DEFAULT_FEATURE_CARDS_MAX:-80}"
DEFAULT_ENABLE_SINGLE_ATOM_HINT="${DEFAULT_ENABLE_SINGLE_ATOM_HINT:-1}"
DEFAULT_DISALLOW_GROUP_TOKENS="${DEFAULT_DISALLOW_GROUP_TOKENS:-1}"
DEFAULT_GROUP_TOKEN_PROMPT_MAX="${DEFAULT_GROUP_TOKEN_PROMPT_MAX:-32}"
DEFAULT_ALLOW_DATASET_CONTEXT="${DEFAULT_ALLOW_DATASET_CONTEXT:-0}"
DEFAULT_PASSTHROUGH_RATIO="${DEFAULT_PASSTHROUGH_RATIO:-0}"
DEFAULT_MAX_AUDIT_PASSTHROUGH_RATIO="${DEFAULT_MAX_AUDIT_PASSTHROUGH_RATIO:--1}"
DEFAULT_MAX_REPAIR_ATTEMPTS="${DEFAULT_MAX_REPAIR_ATTEMPTS:-1}"
DEFAULT_FEATURE_CARDS_FILE="${DEFAULT_FEATURE_CARDS_FILE:-${REPO_DIR}/symbolic_feature_cards.json}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_generate_symbolic_spec_v3.sh --task <task> --output <spec.json> \
    [--feature-pool <csv> | --feature-pool-file <file> | --replay-npz <file>] [extra args...]

Behavior:
  - This wrapper calls scripts/generate_symbolic_spec_gpt5_v3.py.
  - Output schema stays v2 style (--spec-version v2).
  - If you don't pass some options, defaults are injected:
      --model gpt-5-mini
      --reasoning-effort low
      --disallow-group-tokens 1
      --group-token-prompt-max 32
      --enable-single-atom-hint 1
      --allow-dataset-context 0
      --passthrough-ratio 0
      --max-audit-passthrough-ratio -1
      --max-repair-attempts 1
  - Channel count is flexible:
      * pass --num-channels N for a fixed size
      * or pass --min-num-channels N to let LLM decide a size >= N
      * if neither is passed, generator defaults to min = atom-pool size
      * if --channel-roles is omitted, LLM defines roles by itself

Examples:
  bash scripts/run_generate_symbolic_spec_v3.sh \
    --task entity_matching \
    --feature-pool-file ./symbolic_specs/feature_pools/entity_matching.json \
    --output ./symbolic_specs/tmp/entity_matching_v2.json

  OPENAI_API_KEY=... bash scripts/run_generate_symbolic_spec_v3.sh \
    --task joinable_table_search \
    --replay-npz ./outputs/replay/example.npz \
    --min-num-channels 8 \
    --channel-roles precision,recall,coverage,robustness,boundary,calibration,noise,semantic \
    --output ./symbolic_specs/tmp/jts_v2.json
USAGE
}

if [[ ! -f "${GEN_SCRIPT}" ]]; then
  echo "[ERROR] generator not found: ${GEN_SCRIPT}" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

ARGS=("$@")

has_arg() {
  local key="$1"
  local item
  for item in "${ARGS[@]}"; do
    if [[ "${item}" == "${key}" || "${item}" == "${key}="* ]]; then
      return 0
    fi
  done
  return 1
}

get_arg_value() {
  local key="$1"
  local i item next_idx
  for ((i=0; i<${#ARGS[@]}; i++)); do
    item="${ARGS[$i]}"
    if [[ "${item}" == "${key}" ]]; then
      next_idx=$((i + 1))
      if [[ "${next_idx}" -lt "${#ARGS[@]}" ]]; then
        printf "%s" "${ARGS[$next_idx]}"
      fi
      return 0
    fi
    if [[ "${item}" == "${key}="* ]]; then
      printf "%s" "${item#*=}"
      return 0
    fi
  done
  return 1
}

if has_arg "--help" || has_arg "-h"; then
  usage
  exit 0
fi

if ! has_arg "--task"; then
  echo "[ERROR] missing required arg: --task" >&2
  usage
  exit 2
fi
if ! has_arg "--output"; then
  echo "[ERROR] missing required arg: --output" >&2
  usage
  exit 2
fi
if ! has_arg "--feature-pool" && ! has_arg "--feature-pool-file" && ! has_arg "--replay-npz"; then
  echo "[ERROR] provide one feature source: --feature-pool / --feature-pool-file / --replay-npz" >&2
  usage
  exit 2
fi

cmd=(
  "${PYTHON_BIN}" "${GEN_SCRIPT}"
  "${ARGS[@]}"
)

if ! has_arg "--spec-version"; then
  cmd+=(--spec-version v2)
fi
if ! has_arg "--model"; then
  cmd+=(--model "${DEFAULT_MODEL}")
fi
if ! has_arg "--reasoning-effort"; then
  cmd+=(--reasoning-effort "${DEFAULT_REASONING_EFFORT}")
fi
if ! has_arg "--timeout-sec"; then
  cmd+=(--timeout-sec "${DEFAULT_TIMEOUT_SEC}")
fi
if ! has_arg "--max-completion-tokens"; then
  cmd+=(--max-completion-tokens "${DEFAULT_MAX_COMPLETION_TOKENS}")
fi
if ! has_arg "--feature-cards-file"; then
  cmd+=(--feature-cards-file "${DEFAULT_FEATURE_CARDS_FILE}")
fi
if ! has_arg "--feature-cards-max"; then
  cmd+=(--feature-cards-max "${DEFAULT_FEATURE_CARDS_MAX}")
fi
if ! has_arg "--disallow-group-tokens"; then
  cmd+=(--disallow-group-tokens "${DEFAULT_DISALLOW_GROUP_TOKENS}")
fi
if ! has_arg "--group-token-prompt-max"; then
  cmd+=(--group-token-prompt-max "${DEFAULT_GROUP_TOKEN_PROMPT_MAX}")
fi
if ! has_arg "--enable-single-atom-hint"; then
  cmd+=(--enable-single-atom-hint "${DEFAULT_ENABLE_SINGLE_ATOM_HINT}")
fi
if ! has_arg "--allow-dataset-context"; then
  cmd+=(--allow-dataset-context "${DEFAULT_ALLOW_DATASET_CONTEXT}")
fi
if ! has_arg "--passthrough-ratio"; then
  cmd+=(--passthrough-ratio "${DEFAULT_PASSTHROUGH_RATIO}")
fi
if ! has_arg "--max-audit-passthrough-ratio"; then
  cmd+=(--max-audit-passthrough-ratio "${DEFAULT_MAX_AUDIT_PASSTHROUGH_RATIO}")
fi
if ! has_arg "--max-repair-attempts"; then
  cmd+=(--max-repair-attempts "${DEFAULT_MAX_REPAIR_ATTEMPTS}")
fi

printf '[RUN]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
