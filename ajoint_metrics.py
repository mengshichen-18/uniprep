from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def _as_1d_numpy(x, *, dtype) -> np.ndarray:
    return np.array(x, dtype=dtype).reshape(-1)


def _binary_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = _as_1d_numpy(y_true, dtype=np.int64)
    y_score = _as_1d_numpy(y_score, dtype=np.float64)

    pos_mask = y_true == 1
    n_pos = int(pos_mask.sum())
    n = int(y_true.shape[0])
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based ranks, averaged over ties
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_pos_ranks = float(ranks[pos_mask].sum())
    return (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def _binary_classification_metrics(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted scores or probabilities.
        threshold: Decision threshold. score >= threshold is positive.

    Returns:
        Dictionary with auc, accuracy, precision, recall, and f1.
    """
    y_true = _as_1d_numpy(y_true, dtype=np.int64)
    y_score = _as_1d_numpy(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)

    correct = int((y_pred == y_true).sum())
    accuracy = float(correct) / float(y_true.shape[0])

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    auc = _binary_roc_auc_score(y_true, y_score)

    return {
        "link_auc": float(auc),
        "link_accuracy": float(accuracy),
        "link_precision": float(precision),
        "link_recall": float(recall),
        "link_f1": float(f1),
    }


def find_best_threshold(y_true, y_score, thresholds=None) -> Tuple[float, float]:
    """
    Search the threshold that maximizes F1 over a candidate range.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted scores or probabilities.
        thresholds: Candidate thresholds. Defaults to distinct observed validation scores.

    Returns:
        (best_threshold, best_f1): Best threshold and corresponding F1.
    """
    y_true = _as_1d_numpy(y_true, dtype=np.int64)
    y_score = _as_1d_numpy(y_score, dtype=np.float64)

    if y_true.shape != y_score.shape or not np.isfinite(y_score).all():
        raise ValueError("Labels and finite scores must have matching shapes")
    if len(y_true) == 0:
        return 0.5, 0.0

    if thresholds is None:
        order = np.argsort(-y_score, kind="stable")
        ordered = y_score[order]
        tp = np.cumsum(y_true[order] == 1, dtype=np.int64)
        ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(y_true) - 1]
        positives = int(np.sum(y_true == 1))
        best = max(
            ends,
            key=lambda i: (
                2.0 * tp[i] / max(i + 1 + positives, 1),
                tp[i] / max(positives, 1),
                ordered[i],
            ),
        )
        threshold = float(ordered[best])
        return threshold, float(2.0 * tp[best] / max(best + 1 + positives, 1))

    candidates = []
    for threshold in thresholds:
        metrics = _binary_classification_metrics(y_true, y_score, threshold=float(threshold))
        candidates.append((float(threshold), metrics["link_f1"], metrics["link_recall"]))
    if not candidates:
        return 0.5, 0.0
    threshold, f1, _ = max(candidates, key=lambda item: (item[1], item[2], item[0]))
    return threshold, f1


def _elbow_threshold_from_scores(scores: torch.Tensor) -> float:
    scores = scores.detach().cpu().to(dtype=torch.float32).reshape(-1)
    if scores.numel() == 0:
        return float("nan")
    finite_mask = torch.isfinite(scores)
    if not bool(finite_mask.any()):
        return float("nan")
    scores = scores[finite_mask]
    if scores.numel() < 3:
        return float(scores.min().item())

    scores_sorted = torch.sort(scores, descending=True).values
    max_score = float(scores_sorted[0].item())
    min_score = float(scores_sorted[-1].item())
    if max_score == min_score:
        return float(max_score)

    y = (scores_sorted - min_score) / (max_score - min_score)
    n = int(scores_sorted.numel())
    x = torch.linspace(0.0, 1.0, steps=n, dtype=y.dtype)
    line = 1.0 - x
    diff = y - line
    idx = int(torch.argmax(diff).item())
    return float(scores_sorted[idx].item())
