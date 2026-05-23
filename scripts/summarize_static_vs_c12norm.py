#!/usr/bin/env python3
import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TASK_RE = re.compile(r"Training for task:\s*([a-z_]+)")
VAL_RE = re.compile(r"\[Validation\]\s+selected threshold=([0-9.]+)\s+f1=([0-9.]+)")
TEST_RE = re.compile(r"Test Metrics \(threshold=([0-9.]+)\):\s*(\{.*\})")


def _task_title(task: str) -> str:
    return {
        "entity_matching": "EM",
        "joinable_table_search": "JTS",
        "schema_matching": "SM",
        "union_table_search": "UTS",
    }.get(task, task)


def parse_log(log_path: Path) -> List[Dict[str, object]]:
    dataset = log_path.stem
    rows: List[Dict[str, object]] = []

    current_task: Optional[str] = None
    current: Dict[str, object] = {}

    def flush_current() -> None:
        nonlocal current_task, current
        if not current_task:
            return
        row = {
            "dataset": dataset,
            "task": current_task,
            "threshold_valbest": current.get("threshold_valbest"),
            "val_f1_best": current.get("val_f1_best"),
            "test_f1_valbest": current.get("test_f1_valbest"),
            "test_f1_at05": current.get("test_f1_at05"),
        }
        rows.append(row)
        current_task = None
        current = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m_task = TASK_RE.search(line)
            if m_task:
                flush_current()
                current_task = m_task.group(1)
                current = {}
                continue

            if current_task is None:
                continue

            m_val = VAL_RE.search(line)
            if m_val:
                current["threshold_valbest"] = float(m_val.group(1))
                current["val_f1_best"] = float(m_val.group(2))
                continue

            m_test = TEST_RE.search(line)
            if m_test:
                th = float(m_test.group(1))
                try:
                    metrics = ast.literal_eval(m_test.group(2))
                except Exception:
                    metrics = {}
                f1 = metrics.get("link_f1")
                if isinstance(f1, (int, float)):
                    if abs(th - 0.5) <= 1e-9:
                        current["test_f1_at05"] = float(f1)
                    else:
                        tv = current.get("threshold_valbest")
                        if isinstance(tv, (int, float)):
                            if abs(th - float(tv)) <= 1e-6:
                                current["test_f1_valbest"] = float(f1)
                            else:
                                # Fallback: keep first non-0.5 as valbest when threshold line order differs.
                                current.setdefault("test_f1_valbest", float(f1))
                        else:
                            current.setdefault("test_f1_valbest", float(f1))
                continue

    flush_current()
    return rows


def parse_run_dir(run_dir: Path, variant: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for log in sorted(run_dir.glob("*.log")):
        for row in parse_log(log):
            row["variant"] = variant
            rows.append(row)
    return rows


def _fmt_num(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def _to_float(v: object) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def build_comparison(
    static_rows: List[Dict[str, object]],
    c12_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    static_map: Dict[Tuple[str, str], Dict[str, object]] = {
        (str(r["dataset"]), str(r["task"])): r for r in static_rows
    }
    c12_map: Dict[Tuple[str, str], Dict[str, object]] = {
        (str(r["dataset"]), str(r["task"])): r for r in c12_rows
    }

    keys = sorted(set(static_map.keys()) | set(c12_map.keys()))
    out: List[Dict[str, object]] = []
    for ds, task in keys:
        s = static_map.get((ds, task), {})
        c = c12_map.get((ds, task), {})
        s_val = _to_float(s.get("test_f1_valbest"))
        c_val = _to_float(c.get("test_f1_valbest"))
        s_05 = _to_float(s.get("test_f1_at05"))
        c_05 = _to_float(c.get("test_f1_at05"))

        out.append(
            {
                "dataset": ds,
                "task": task,
                "task_short": _task_title(task),
                "static_threshold_valbest": _to_float(s.get("threshold_valbest")),
                "static_test_f1_valbest": s_val,
                "static_test_f1_at05": s_05,
                "c12norm_threshold_valbest": _to_float(c.get("threshold_valbest")),
                "c12norm_test_f1_valbest": c_val,
                "c12norm_test_f1_at05": c_05,
                "delta_valbest": (c_val - s_val) if (c_val is not None and s_val is not None) else None,
                "delta_at05": (c_05 - s_05) if (c_05 is not None and s_05 is not None) else None,
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Static vs Static+C12Norm")
    lines.append("")
    lines.append(
        "| dataset | task | static(valbest/@0.5) | c12norm(valbest/@0.5) | delta_valbest | delta@0.5 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for r in rows:
        s = f"{_fmt_num(_to_float(r.get('static_test_f1_valbest')))} / {_fmt_num(_to_float(r.get('static_test_f1_at05')))}"
        c = f"{_fmt_num(_to_float(r.get('c12norm_test_f1_valbest')))} / {_fmt_num(_to_float(r.get('c12norm_test_f1_at05')))}"
        lines.append(
            f"| {r['dataset']} | {r['task_short']} | {s} | {c} | {_fmt_num(_to_float(r.get('delta_valbest')))} | {_fmt_num(_to_float(r.get('delta_at05')))} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize static vs static+c12norm from online run logs.")
    ap.add_argument("--static-dir", required=True, help="Run directory for static-only logs")
    ap.add_argument("--c12-dir", required=True, help="Run directory for static+c12norm logs")
    ap.add_argument("--output-dir", required=True, help="Output directory for csv/md")
    args = ap.parse_args()

    static_dir = Path(args.static_dir).resolve()
    c12_dir = Path(args.c12_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    static_rows = parse_run_dir(static_dir, "static")
    c12_rows = parse_run_dir(c12_dir, "static_c12norm")
    comp_rows = build_comparison(static_rows, c12_rows)

    write_csv(out_dir / "raw_static.csv", static_rows)
    write_csv(out_dir / "raw_static_c12norm.csv", c12_rows)
    write_csv(out_dir / "comparison.csv", comp_rows)
    write_markdown(out_dir / "comparison.md", comp_rows)

    print(f"[OK] static rows: {len(static_rows)}")
    print(f"[OK] c12norm rows: {len(c12_rows)}")
    print(f"[OK] comparison rows: {len(comp_rows)}")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()

