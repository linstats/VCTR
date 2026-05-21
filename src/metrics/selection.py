"""Selection metrics for sparse paired-eye VCTR."""

from __future__ import annotations

import numpy as np


def confusion_counts(mask_true: np.ndarray, mask_pred: np.ndarray) -> dict[str, int]:
    """Return TP, TN, FP, FN counts for two boolean masks."""

    truth = np.asarray(mask_true, dtype=bool).reshape(-1)
    pred = np.asarray(mask_pred, dtype=bool).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("mask_true and mask_pred must have the same shape.")

    tp = int(np.sum(truth & pred))
    tn = int(np.sum((~truth) & (~pred)))
    fp = int(np.sum((~truth) & pred))
    fn = int(np.sum(truth & (~pred)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def sensitivity(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """True positive rate."""

    counts = confusion_counts(mask_true, mask_pred)
    return _safe_ratio(counts["tp"], counts["tp"] + counts["fn"])


def specificity(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """True negative rate."""

    counts = confusion_counts(mask_true, mask_pred)
    return _safe_ratio(counts["tn"], counts["tn"] + counts["fp"])


def ppv(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """Positive predictive value."""

    counts = confusion_counts(mask_true, mask_pred)
    return _safe_ratio(counts["tp"], counts["tp"] + counts["fp"])


def npv(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """Negative predictive value."""

    counts = confusion_counts(mask_true, mask_pred)
    return _safe_ratio(counts["tn"], counts["tn"] + counts["fn"])


def accuracy(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """Overall mask recovery accuracy."""

    counts = confusion_counts(mask_true, mask_pred)
    total = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    return _safe_ratio(counts["tp"] + counts["tn"], total)


def f1_score(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """Binary F1 score for support recovery."""

    counts = confusion_counts(mask_true, mask_pred)
    precision = _safe_ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = _safe_ratio(counts["tp"], counts["tp"] + counts["fn"])
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a guarded ratio with zero-denominator fallback."""

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
