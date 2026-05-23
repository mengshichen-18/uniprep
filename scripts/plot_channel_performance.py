#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANNELS = ["c1", "c2", "c4", "c8", "c12"]
TASKS = ["EM", "JTS", "SM", "UTS"]
DATASETS = ["Magellan", "Santos", "Wikidbs"]
PALETTE = {"Magellan": "#376795", "Santos": "#72BCD5", "Wikidbs": "#E76254"}
MARKERS = {"Magellan": "o", "Santos": "s", "Wikidbs": "^"}


def _read_note_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'md\s*=\s*"""(.*)"""', text, flags=re.DOTALL)
    return m.group(1) if m else text


def _parse_mean_std(token: str) -> Tuple[float, float]:
    token = token.strip()
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*±\s*([0-9]*\.?[0-9]+)$", token)
    if not m:
        raise ValueError(f"Bad mean±std token: {token!r}")
    return float(m.group(1)), float(m.group(2))


def _find_table_rows(lines: List[str], header_prefix: str) -> List[str]:
    start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(header_prefix):
            start = i
            break
    if start < 0:
        raise ValueError(f"Cannot find table header: {header_prefix}")
    rows: List[str] = []
    for line in lines[start + 2 :]:
        s = line.strip()
        if not s.startswith("|"):
            break
        rows.append(s)
    return rows


def parse_note(path: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    text = _read_note_text(path)
    lines = text.splitlines()

    raw_rows = _find_table_rows(lines, "| Dataset | Task | c1 | c2 | c4 | c8 | c12 |")
    recs: List[Dict[str, object]] = []
    for row in raw_rows:
        cols = [c.strip() for c in row.strip("|").split("|")]
        if len(cols) != 7:
            continue
        dataset, task = cols[0], cols[1]
        if dataset not in DATASETS or task not in TASKS:
            continue
        for ch, token in zip(CHANNELS, cols[2:]):
            mean, std = _parse_mean_std(token)
            recs.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "channel": ch,
                    "mean": mean,
                    "std": std,
                }
            )
    df = pd.DataFrame(recs)
    if df.empty:
        raise ValueError("Parsed table is empty.")

    avg_rows = _find_table_rows(lines, "| Metric | c1 | c2 | c4 | c8 | c12 |")
    avg_vals: Dict[str, float] = {}
    for row in avg_rows:
        cols = [c.strip() for c in row.strip("|").split("|")]
        if len(cols) == 6 and cols[0].lower().startswith("avg"):
            avg_vals = {ch: float(v) for ch, v in zip(CHANNELS, cols[1:])}
            break
    if not avg_vals:
        raise ValueError("Cannot parse Avg F1 row.")
    return df, avg_vals


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, outdir: Path, stem: str, export_pgf: bool) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=420, bbox_inches="tight")
    if export_pgf:
        try:
            fig.savefig(outdir / f"{stem}.pgf", bbox_inches="tight")
        except Exception as exc:
            print(f"[warn] failed to export PGF for {stem}: {exc}")


def plot_main(
    df: pd.DataFrame, outdir: Path, stem: str, export_pgf: bool, paper_ready: bool
) -> None:
    _style()
    x = np.arange(len(CHANNELS))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        block = df[df["task"] == task]
        for ds in DATASETS:
            d = block[block["dataset"] == ds].copy()
            d["channel"] = pd.Categorical(d["channel"], CHANNELS, ordered=True)
            d = d.sort_values("channel")
            y = d["mean"].to_numpy()
            e = d["std"].to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=e,
                color=PALETTE[ds],
                marker=MARKERS[ds],
                markersize=6.0,
                linewidth=2.0,
                capsize=3.0,
                capthick=1.1,
                elinewidth=1.1,
                label=ds,
                alpha=0.98,
            )

        y_lo = float((block["mean"] - block["std"]).min())
        y_hi = float((block["mean"] + block["std"]).max())
        pad = max(0.01, 0.14 * (y_hi - y_lo))
        ax.set_ylim(max(0.0, y_lo - pad), min(1.0, y_hi + pad))
        if not paper_ready:
            ax.set_title(task)
        ax.set_xticks(x, CHANNELS)
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
        panel_label = f"({chr(ord('a') + idx)}) {task}" if paper_ready else f"({chr(ord('a') + idx)})"
        ax.text(
            0.02,
            0.95,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
    )
    fig.supxlabel("Number of symbolic channels (c)", y=0.03)
    fig.supylabel("F1 score (mean ± std)", x=0.04)
    if not paper_ready:
        fig.text(
            0.5,
            0.01,
            "Note: each subplot uses an independent y-axis range for readability.",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
    plt.tight_layout(rect=[0.04, 0.05, 1.0, 0.92])
    _save_figure(fig, outdir, stem, export_pgf=export_pgf)
    plt.close(fig)


def plot_avg(
    avg_vals: Dict[str, float], outdir: Path, stem: str, export_pgf: bool, paper_ready: bool
) -> None:
    _style()
    x = np.arange(len(CHANNELS))
    y = np.array([avg_vals[ch] for ch in CHANNELS], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(
        x,
        y,
        color="#1E466E",
        marker="D",
        markersize=6.0,
        linewidth=2.2,
        label="Avg F1 (12 points)",
    )
    for i, yi in enumerate(y):
        ax.text(i, yi + 0.0015, f"{yi:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, CHANNELS)
    ax.set_ylim(max(0.0, y.min() - 0.015), min(1.0, y.max() + 0.015))
    ax.set_xlabel("Number of symbolic channels (c)")
    ax.set_ylabel("Average F1")
    if not paper_ready:
        ax.set_title("Average F1 Trend Across Channel Counts")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    _save_figure(fig, outdir, stem, export_pgf=export_pgf)
    plt.close(fig)


def write_latex_templates(outdir: Path, use_pgf: bool, paper_ready: bool) -> None:
    if paper_ready:
        rel_main_pdf = "channel_performance_small_multiples_paper.pdf"
        rel_avg_pdf = "channel_performance_avg_trend_paper.pdf"
        rel_main_pgf = "channel_performance_small_multiples_paper.pgf"
        rel_avg_pgf = "channel_performance_avg_trend_paper.pgf"
    else:
        rel_main_pdf = "channel_performance_small_multiples.pdf"
        rel_avg_pdf = "channel_performance_avg_trend.pdf"
        rel_main_pgf = "channel_performance_small_multiples.pgf"
        rel_avg_pgf = "channel_performance_avg_trend.pgf"

    tex_pdf = rf"""% Auto-generated LaTeX snippet (PDF includegraphics version)
% Required packages:
% \usepackage{{graphicx}}
% \usepackage{{subcaption}}  % only if you keep the second figure as subfigure

\begin{{figure*}}[t]
  \centering
  \includegraphics[width=0.98\textwidth]{{{rel_main_pdf}}}
  \caption{{Impact of the number of symbolic feature channels ($c\in\{{1,2,4,8,12\}}$) on F1.
  Each panel corresponds to one task (EM, JTS, SM, UTS), with lines for Magellan, Santos, and Wikidbs.
  Error bars denote mean$\pm$std. Note that each panel uses an independent y-axis range for readability.}}
  \label{{fig:channel-performance-main}}
\end{{figure*}}

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.90\linewidth]{{{rel_avg_pdf}}}
  \caption{{Average F1 trend over all 12 dataset-task combinations as channel count increases.}}
  \label{{fig:channel-performance-avg}}
\end{{figure}}
"""
    (outdir / "channel_performance_latex_pdf.tex").write_text(tex_pdf, encoding="utf-8")

    tex_pgf_header = (
        "% Auto-generated LaTeX snippet (PGF version)\n"
        "% Required packages:\n"
        "% \\usepackage{pgf}\n"
        "% \\usepackage{graphicx}\n\n"
    )
    if use_pgf:
        tex_pgf = tex_pgf_header + rf"""\begin{{figure*}}[t]
  \centering
  \input{{{rel_main_pgf}}}
  \caption{{Impact of the number of symbolic feature channels ($c\in\{{1,2,4,8,12\}}$) on F1.
  Each panel corresponds to one task (EM, JTS, SM, UTS), with lines for Magellan, Santos, and Wikidbs.
  Error bars denote mean$\pm$std. Note that each panel uses an independent y-axis range for readability.}}
  \label{{fig:channel-performance-main-pgf}}
\end{{figure*}}

\begin{{figure}}[t]
  \centering
  \input{{{rel_avg_pgf}}}
  \caption{{Average F1 trend over all 12 dataset-task combinations as channel count increases.}}
  \label{{fig:channel-performance-avg-pgf}}
\end{{figure}}
"""
    else:
        tex_pgf = tex_pgf_header + "% PGF export is disabled or unavailable.\n"
    (outdir / "channel_performance_latex_pgf.tex").write_text(tex_pgf, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot channel-performance figures from note markdown.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("0410_feature/docs/notes/themes/channel_performance.md"),
        help="Path to channel_performance markdown/note file.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("0410_feature/docs/figures/channel_performance"),
        help="Output directory for figures.",
    )
    ap.add_argument(
        "--export-pgf",
        type=int,
        default=1,
        help="Export .pgf files for direct LaTeX input (1/0).",
    )
    ap.add_argument(
        "--paper-ready",
        type=int,
        default=1,
        help="Use publication style: no in-figure title, rely on caption (1/0).",
    )
    args = ap.parse_args()

    df, avg_vals = parse_note(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)

    export_pgf = bool(int(args.export_pgf))
    paper_ready = bool(int(args.paper_ready))
    main_stem = "channel_performance_small_multiples_paper" if paper_ready else "channel_performance_small_multiples"
    avg_stem = "channel_performance_avg_trend_paper" if paper_ready else "channel_performance_avg_trend"
    plot_main(
        df,
        args.outdir,
        stem=main_stem,
        export_pgf=export_pgf,
        paper_ready=paper_ready,
    )
    plot_avg(
        avg_vals,
        args.outdir,
        stem=avg_stem,
        export_pgf=export_pgf,
        paper_ready=paper_ready,
    )
    write_latex_templates(args.outdir, use_pgf=export_pgf, paper_ready=paper_ready)
    print(f"Saved figures to: {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
