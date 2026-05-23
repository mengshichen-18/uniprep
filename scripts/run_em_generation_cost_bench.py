#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task_featgen_config import TASK_FEATGEN_CONFIGS  # noqa: E402


DATASETS = ["wikidbs", "santos_benchmark", "magellan"]
PYTHON_BIN_DEFAULT = os.environ.get("PYTHON_BIN", "python3")
ATOM_GEN_SCRIPT = _ROOT / "scripts" / "generate_task_feature_functions_gpt5.py"
SYMBOLIC_GEN_SCRIPT = _ROOT / "scripts" / "generate_symbolic_spec_gpt5_v3.py"
DEFAULT_API_KEY_FILE = _ROOT.parent / "0325_policy_pro" / "LIGHTNING_API_KEY.md"
DEFAULT_CONTEXTS_DIR = _ROOT / "outputs" / "featgen_contexts"
DEFAULT_OUTPUT_ROOT = _ROOT / "outputs" / "em_generation_costbench_20260430"
DEFAULT_FEATURE_CARDS = _ROOT / "symbolic_feature_cards.json"


def _ordered_unique(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        token = str(item).strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _run_cmd(cmd: List[str], *, cwd: Path, env: Dict[str, str]) -> float:
    started = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)
    return float(time.perf_counter() - started)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run EM-only generation cost bench over three datasets.")
    ap.add_argument("--task", type=str, default="entity_matching")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--python-bin", type=str, default=PYTHON_BIN_DEFAULT)
    ap.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--contexts-dir", type=str, default=str(DEFAULT_CONTEXTS_DIR))
    ap.add_argument("--api-key-file", type=str, default=str(DEFAULT_API_KEY_FILE))
    ap.add_argument("--api-key-label", type=str, default="0428_KEY")
    ap.add_argument("--atom-model", type=str, default="gpt-5")
    ap.add_argument("--symbolic-model", type=str, default="gpt-5")
    ap.add_argument("--atom-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    ap.add_argument("--symbolic-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    ap.add_argument("--atom-timeout-sec", type=float, default=120.0)
    ap.add_argument("--symbolic-timeout-sec", type=float, default=120.0)
    ap.add_argument("--atom-max-completion-tokens", type=int, default=2200)
    ap.add_argument("--symbolic-max-completion-tokens", type=int, default=3200)
    ap.add_argument("--atom-reasoning-effort", type=str, default="low")
    ap.add_argument("--symbolic-reasoning-effort", type=str, default="low")
    ap.add_argument("--atom-max-repair-attempts", type=int, default=2)
    ap.add_argument("--symbolic-max-repair-attempts", type=int, default=1)
    ap.add_argument("--target-total-pool-size", type=int, default=12)
    ap.add_argument("--allow-dataset-context", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    task = str(args.task).strip()
    if task != "entity_matching":
        raise ValueError("This benchmark script is currently scoped to entity_matching only.")

    cfg = TASK_FEATGEN_CONFIGS[task]
    output_root = Path(str(args.output_root)).expanduser().resolve()
    contexts_dir = Path(str(args.contexts_dir)).expanduser().resolve()
    python_bin = Path(str(args.python_bin)).expanduser().resolve()
    if not python_bin.exists():
        raise FileNotFoundError(f"python bin not found: {python_bin}")

    base_atoms = _ordered_unique(
        [str(x) for x in cfg.get("human_exemplar_atoms", [])]
        + [str(x) for x in cfg.get("decoder_static_atoms", [])]
    )
    target_total_pool_size = int(args.target_total_pool_size)
    num_generated_features = target_total_pool_size - len(base_atoms)
    if num_generated_features <= 0:
        raise ValueError(
            f"target_total_pool_size={target_total_pool_size} leaves no room for generated atoms; "
            f"base_atom_count={len(base_atoms)}"
        )

    datasets = [token.strip() for token in str(args.datasets).split(",") if token.strip()]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")

    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        dataset_dir = output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_context_json = contexts_dir / f"{task}__{dataset}.json"
        if not dataset_context_json.is_file():
            raise FileNotFoundError(f"dataset context file not found: {dataset_context_json}")

        atom_output = dataset_dir / f"{task}__{dataset}__generated_atoms.json"
        atom_prompt = dataset_dir / f"{task}__{dataset}__atom_prompt.json"
        atom_validation = dataset_dir / f"{task}__{dataset}__atom_validation.json"
        atom_response = dataset_dir / f"{task}__{dataset}__atom_response.txt"
        atom_summary = dataset_dir / f"{task}__{dataset}__atom_summary.json"

        atom_cmd = [
            str(python_bin),
            str(ATOM_GEN_SCRIPT),
            "--task",
            task,
            "--output",
            str(atom_output),
            "--num-features",
            str(num_generated_features),
            "--target-total-pool-size",
            str(target_total_pool_size),
            "--teacher-feature-pool",
            ",".join(str(x) for x in cfg.get("teacher_pool_atoms", [])),
            "--preserved-features",
            ",".join(str(x) for x in cfg.get("human_exemplar_atoms", [])),
            "--protected-feature-names",
            ",".join(base_atoms),
            "--task-description",
            str(cfg.get("task_description", "")),
            "--data-description",
            str(cfg.get("data_description", "")),
            "--selection-notes",
            str(cfg.get("selection_notes", "")),
            "--dataset-context-json",
            str(dataset_context_json),
            "--model",
            str(args.atom_model),
            "--reasoning-effort",
            str(args.atom_reasoning_effort),
            "--max-completion-tokens",
            str(int(args.atom_max_completion_tokens)),
            "--timeout-sec",
            str(float(args.atom_timeout_sec)),
            "--max-repair-attempts",
            str(int(args.atom_max_repair_attempts)),
            "--api-key-file",
            str(Path(str(args.api_key_file)).expanduser().resolve()),
            "--api-key-label",
            str(args.api_key_label),
            "--dump-prompt",
            str(atom_prompt),
            "--dump-validation",
            str(atom_validation),
            "--dump-response",
            str(atom_response),
            "--summary-output",
            str(atom_summary),
        ]
        if str(args.atom_base_url).strip():
            atom_cmd.extend(["--base-url", str(args.atom_base_url).strip()])

        atom_wall_sec = _run_cmd(atom_cmd, cwd=_ROOT, env=env)
        atom_summary_payload = _load_json(atom_summary)
        atom_specs_payload = _load_json(atom_output)
        generated_features = atom_specs_payload.get("features", [])
        if not isinstance(generated_features, list):
            raise ValueError(f"Unexpected atom output format: {atom_output}")
        generated_names = [
            str(item.get("feature_name", "")).strip()
            for item in generated_features
            if isinstance(item, dict) and str(item.get("feature_name", "")).strip()
        ]
        symbolic_feature_pool = _ordered_unique(base_atoms + generated_names)

        symbolic_output = dataset_dir / f"{task}__{dataset}__symbolic_auto_min{target_total_pool_size}.json"
        symbolic_prompt = dataset_dir / f"{task}__{dataset}__symbolic_prompt.json"
        symbolic_summary = dataset_dir / f"{task}__{dataset}__symbolic_summary.json"

        symbolic_cmd = [
            str(python_bin),
            str(SYMBOLIC_GEN_SCRIPT),
            "--task",
            task,
            "--feature-pool",
            ",".join(symbolic_feature_pool),
            "--output",
            str(symbolic_output),
            "--summary-output",
            str(symbolic_summary),
            "--feature-cards-file",
            str(DEFAULT_FEATURE_CARDS.resolve()),
            "--feature-cards-max",
            "80",
            "--disallow-group-tokens",
            "1",
            "--group-token-prompt-max",
            "32",
            "--enable-single-atom-hint",
            "1",
            "--passthrough-ratio",
            "0",
            "--max-audit-passthrough-ratio",
            "-1",
            "--max-repair-attempts",
            str(int(args.symbolic_max_repair_attempts)),
            "--allow-dataset-context",
            str(int(args.allow_dataset_context)),
            "--dataset-context-file",
            str(dataset_context_json),
            "--min-num-channels",
            str(target_total_pool_size),
            "--model",
            str(args.symbolic_model),
            "--reasoning-effort",
            str(args.symbolic_reasoning_effort),
            "--timeout-sec",
            str(float(args.symbolic_timeout_sec)),
            "--max-completion-tokens",
            str(int(args.symbolic_max_completion_tokens)),
            "--api-key-file",
            str(Path(str(args.api_key_file)).expanduser().resolve()),
            "--api-key-label",
            str(args.api_key_label),
            "--dump-prompt",
            str(symbolic_prompt),
        ]
        if str(args.symbolic_base_url).strip():
            symbolic_cmd.extend(["--base-url", str(args.symbolic_base_url).strip()])

        symbolic_wall_sec = _run_cmd(symbolic_cmd, cwd=_ROOT, env=env)
        symbolic_summary_payload = _load_json(symbolic_summary)

        row = {
            "dataset": dataset,
            "atom": {
                "requests": _int(atom_summary_payload.get("llm_api_request_count")),
                "prompt_tokens": _int(atom_summary_payload.get("llm_prompt_tokens_total")),
                "completion_tokens": _int(atom_summary_payload.get("llm_completion_tokens_total")),
                "total_tokens": _int(atom_summary_payload.get("llm_total_tokens_total")),
                "llm_elapsed_sec": _float(atom_summary_payload.get("llm_elapsed_sec_total")),
                "wall_elapsed_sec": atom_wall_sec,
                "generated_feature_count": _int(atom_summary_payload.get("feature_count")),
            },
            "symbolic": {
                "requests": _int(symbolic_summary_payload.get("llm_api_request_count")),
                "prompt_tokens": _int(symbolic_summary_payload.get("llm_prompt_tokens_total")),
                "completion_tokens": _int(symbolic_summary_payload.get("llm_completion_tokens_total")),
                "total_tokens": _int(symbolic_summary_payload.get("llm_total_tokens_total")),
                "llm_elapsed_sec": _float(symbolic_summary_payload.get("llm_elapsed_sec_total")),
                "wall_elapsed_sec": symbolic_wall_sec,
                "channel_count": _int(symbolic_summary_payload.get("channel_count")),
            },
        }
        row["total"] = {
            "requests": int(row["atom"]["requests"] + row["symbolic"]["requests"]),
            "prompt_tokens": int(row["atom"]["prompt_tokens"] + row["symbolic"]["prompt_tokens"]),
            "completion_tokens": int(row["atom"]["completion_tokens"] + row["symbolic"]["completion_tokens"]),
            "total_tokens": int(row["atom"]["total_tokens"] + row["symbolic"]["total_tokens"]),
            "llm_elapsed_sec": float(row["atom"]["llm_elapsed_sec"] + row["symbolic"]["llm_elapsed_sec"]),
            "wall_elapsed_sec": float(row["atom"]["wall_elapsed_sec"] + row["symbolic"]["wall_elapsed_sec"]),
        }
        rows.append(row)

    avg = {
        "atom": {},
        "symbolic": {},
        "total": {},
    }
    count = max(1, len(rows))
    for section in ("atom", "symbolic", "total"):
        numeric_keys = list(rows[0][section].keys())
        for key in numeric_keys:
            avg[section][key] = sum(_float(row[section][key]) for row in rows) / float(count)

    summary = {
        "task": task,
        "datasets": datasets,
        "target_total_pool_size": target_total_pool_size,
        "base_atoms": base_atoms,
        "num_generated_features": num_generated_features,
        "rows": rows,
        "average": avg,
    }

    summary_path = output_root / "em_generation_cost_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
