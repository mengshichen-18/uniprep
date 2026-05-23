#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task_featgen_config import DATASET_CSV_DEFAULT, TASK_FEATGEN_CONFIGS, TASK_ORDER_DEFAULT


DEFAULT_ATOM_DICT_PATHS: Dict[str, str] = {}


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _resolve_atom_dict(dataset: str, override_root: str) -> Optional[Path]:
    token = str(dataset).strip()
    if str(override_root).strip():
        root = Path(str(override_root)).expanduser()
        candidate = root / f"{token}_040303" / "atom_dictionary.json"
        if candidate.is_file():
            return candidate
        candidate = root / token / "atom_dictionary.json"
        if candidate.is_file():
            return candidate
    default = DEFAULT_ATOM_DICT_PATHS.get(token, "")
    if default:
        p = Path(default)
        if p.is_file():
            return p
    return None


def _build_dataset_context(
    *,
    dataset_name: str,
    task: str,
    teacher_feature_pool: Sequence[str],
    atom_dict_path: Optional[Path],
    top_atoms: int,
    split: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "task": task,
        "split": split,
        "source_file": str(atom_dict_path.resolve()) if atom_dict_path else "",
        "selected_atoms": [],
        "selection_policy": (
            f"{split}-only atom stats ranked by auc then best_f1 within task teacher pool"
        ),
    }
    if atom_dict_path is None:
        return payload

    raw = json.loads(atom_dict_path.read_text(encoding="utf-8"))
    tasks = raw.get("tasks", {}) if isinstance(raw, Mapping) else {}
    task_obj = tasks.get(task, {}) if isinstance(tasks, Mapping) else {}
    atoms = task_obj.get("atoms", {}) if isinstance(task_obj, Mapping) else {}
    if not isinstance(atoms, Mapping):
        return payload

    feature_pool_set = {str(x).strip() for x in teacher_feature_pool if str(x).strip()}
    ranked: List[Tuple[float, float, str, Dict[str, Any]]] = []
    for atom_name, atom_payload in atoms.items():
        if str(atom_name) not in feature_pool_set:
            continue
        if not isinstance(atom_payload, Mapping):
            continue
        split_stats = atom_payload.get(split, {})
        if not isinstance(split_stats, Mapping):
            continue

        auc = _safe_float(split_stats.get("auc", None))
        best_f1 = _safe_float(split_stats.get("best_f1", None))
        row = {
            "name": str(atom_name),
            "auc": auc,
            "best_f1": best_f1,
            "best_f1_threshold": _safe_float(split_stats.get("best_f1_threshold", None)),
            "missing_rate": _safe_float(split_stats.get("missing_rate", None)),
            "mean": _safe_float(split_stats.get("mean", None)),
            "std": _safe_float(split_stats.get("std", None)),
            "p25": _safe_float(split_stats.get("p25", None)),
            "p50": _safe_float(split_stats.get("p50", None)),
            "p75": _safe_float(split_stats.get("p75", None)),
            "p95": _safe_float(split_stats.get("p95", None)),
        }
        auc_rank = auc if auc is not None else -1.0
        f1_rank = best_f1 if best_f1 is not None else -1.0
        ranked.append((auc_rank, f1_rank, str(atom_name), row))

    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    payload["selected_atoms"] = [item[3] for item in ranked[: max(0, int(top_atoms))]]
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Build task x dataset context JSONs for 0428 featgen.")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--datasets", type=str, default=DATASET_CSV_DEFAULT)
    ap.add_argument("--tasks", type=str, default=",".join(TASK_ORDER_DEFAULT))
    ap.add_argument("--top-atoms", type=int, default=8)
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--atom-dict-root", type=str, default="")
    args = ap.parse_args()

    out_dir = Path(str(args.output_dir)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = _parse_csv(args.datasets)
    tasks = _parse_csv(args.tasks)

    summary: Dict[str, Any] = {
        "output_dir": str(out_dir.resolve()),
        "datasets": datasets,
        "tasks": tasks,
        "top_atoms": int(args.top_atoms),
        "split": str(args.split),
        "written": [],
    }

    for dataset in datasets:
        atom_dict_path = _resolve_atom_dict(dataset, str(args.atom_dict_root))
        for task in tasks:
            cfg = TASK_FEATGEN_CONFIGS[str(task)]
            payload = _build_dataset_context(
                dataset_name=str(dataset),
                task=str(task),
                teacher_feature_pool=list(cfg.get("teacher_pool_atoms", [])),
                atom_dict_path=atom_dict_path,
                top_atoms=int(args.top_atoms),
                split=str(args.split),
            )
            out_path = out_dir / f"{task}__{dataset}.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["written"].append(
                {
                    "task": str(task),
                    "dataset": str(dataset),
                    "path": str(out_path.resolve()),
                    "selected_atoms": int(len(payload.get("selected_atoms", []))),
                    "source_file": str(payload.get("source_file", "")),
                }
            )

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
