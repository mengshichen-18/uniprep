#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from symbolic_feature import load_symbolic_feature_spec, save_symbolic_feature_spec  # noqa: E402


def _parse_feature_pool(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _collect_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate symbolic spec JSON files.")
    parser.add_argument("--path", type=str, required=True, help="Spec file or directory.")
    parser.add_argument("--task", type=str, default="", help="Optional expected task.")
    parser.add_argument(
        "--feature-pool",
        type=str,
        default="",
        help="Optional allowed feature names (comma-separated).",
    )
    parser.add_argument(
        "--rewrite-normalized",
        type=int,
        default=0,
        choices=[0, 1],
        help="Rewrite each valid file using normalized canonical payload.",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise FileNotFoundError(f"path not found: {root}")

    files = _collect_files(root)
    if not files:
        raise FileNotFoundError(f"No .json files found under: {root}")

    expected_task = str(args.task).strip() or None
    allowed_features = _parse_feature_pool(args.feature_pool)
    if not allowed_features:
        allowed_features = None

    ok_rows = []
    fail_rows = []

    for path in files:
        try:
            spec = load_symbolic_feature_spec(
                path,
                expected_task=expected_task,
                allowed_features=allowed_features,
            )
            if int(args.rewrite_normalized) == 1:
                save_symbolic_feature_spec(spec, path)
            ok_rows.append(
                {
                    "path": str(path),
                    "task": spec.task,
                    "spec_id": spec.spec_id,
                    "features": list(spec.feature_pool_used),
                    "expression": spec.expression,
                }
            )
        except Exception as exc:  # noqa: BLE001
            fail_rows.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "checked": len(files),
        "ok": len(ok_rows),
        "failed": len(fail_rows),
        "ok_rows": ok_rows,
        "fail_rows": fail_rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not fail_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
