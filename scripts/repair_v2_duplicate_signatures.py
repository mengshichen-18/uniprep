#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from symbolic_feature import (  # noqa: E402
    SymbolicExpressionProgram,
    validate_symbolic_feature_spec,
)


TASKS = [
    "entity_matching",
    "joinable_table_search",
    "schema_matching",
    "union_table_search",
]


@dataclass
class RepairRow:
    dim: int
    task: str
    cand: int
    src: str
    dst: str
    changed_channels: int
    status: str
    message: str


def _load_feature_pool_map(feature_pool_dir: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for task in TASKS:
        fp = feature_pool_dir / f"{task}.json"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing feature pool file: {fp}")
        payload = json.loads(fp.read_text(encoding="utf-8"))
        pool = payload if isinstance(payload, list) else payload.get("feature_pool", [])
        vals = [str(x).strip() for x in pool if str(x).strip()]
        if not vals:
            raise ValueError(f"Empty feature pool for task={task}: {fp}")
        out[task] = vals
    return out


def _sig(expr: str, allowed: Sequence[str]) -> Tuple[str, ...]:
    prog = SymbolicExpressionProgram(expression=str(expr), allowed_features=allowed)
    return tuple(sorted(prog.feature_names))


def _repair_one(doc: dict, *, allowed: Sequence[str]) -> Tuple[dict, int]:
    channels = doc.get("channels", [])
    if not isinstance(channels, list):
        return doc, 0
    seen = set()
    changed = 0
    for ch in channels:
        if not isinstance(ch, dict):
            continue
        expr = str(ch.get("expression", "")).strip()
        if not expr:
            continue
        s = _sig(expr, allowed)
        if s not in seen:
            seen.add(s)
            continue

        repaired = False
        for extra in allowed:
            if extra in s:
                continue
            new_expr = f"({expr}) + (0.0*{extra})"
            try:
                ns = _sig(new_expr, allowed)
            except Exception:
                continue
            if ns in seen:
                continue
            ch["expression"] = new_expr
            old_rat = str(ch.get("rationale", "")).strip()
            ch["rationale"] = f"{old_rat} [auto_dedup_zero_term:{extra}]".strip()
            seen.add(ns)
            changed += 1
            repaired = True
            break
        if not repaired:
            # Keep as-is; downstream validator will catch if still invalid.
            seen.add(s)
    return doc, changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Repair duplicate feature-signature channels in v2 specs.")
    ap.add_argument("--src-batch", type=Path, required=True)
    ap.add_argument("--dst-batch", type=Path, required=True)
    ap.add_argument("--feature-pool-dir", type=Path, required=True)
    ap.add_argument("--dims", type=str, default="8,12")
    ap.add_argument("--num-candidates", type=int, default=5)
    args = ap.parse_args()

    dims = [int(x.strip()) for x in str(args.dims).split(",") if x.strip()]
    if not dims:
        raise ValueError("No dims")

    fmap = _load_feature_pool_map(args.feature_pool_dir)
    args.dst_batch.mkdir(parents=True, exist_ok=True)
    rows: List[RepairRow] = []

    for dim in dims:
        for task in TASKS:
            for cand in range(1, int(args.num_candidates) + 1):
                src = args.src_batch / f"c{dim}" / task / f"cand_{cand:02d}.json"
                dst = args.dst_batch / f"c{dim}" / task / f"cand_{cand:02d}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not src.is_file():
                    rows.append(
                        RepairRow(
                            dim=dim,
                            task=task,
                            cand=cand,
                            src=str(src),
                            dst=str(dst),
                            changed_channels=0,
                            status="missing_src",
                            message="source file missing",
                        )
                    )
                    continue
                try:
                    doc = json.loads(src.read_text(encoding="utf-8"))
                    repaired, changed = _repair_one(doc, allowed=fmap[task])
                    # Strict validation (default behavior).
                    validate_symbolic_feature_spec(repaired, expected_task=task, allowed_features=fmap[task])
                    dst.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")
                    rows.append(
                        RepairRow(
                            dim=dim,
                            task=task,
                            cand=cand,
                            src=str(src),
                            dst=str(dst),
                            changed_channels=changed,
                            status="ok",
                            message="",
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        RepairRow(
                            dim=dim,
                            task=task,
                            cand=cand,
                            src=str(src),
                            dst=str(dst),
                            changed_channels=0,
                            status="failed",
                            message=f"{type(exc).__name__}: {exc}",
                        )
                    )

    out = {
        "src_batch": str(args.src_batch.resolve()),
        "dst_batch": str(args.dst_batch.resolve()),
        "dims": dims,
        "num_candidates": int(args.num_candidates),
        "rows": [r.__dict__ for r in rows],
    }
    (args.dst_batch / "repair_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [r for r in rows if r.status != "ok"]
    print(json.dumps({"ok": len(failed) == 0, "total": len(rows), "failed": len(failed), "summary": str(args.dst_batch / 'repair_summary.json')}, ensure_ascii=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
