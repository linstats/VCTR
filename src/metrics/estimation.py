"""Basic estimation metrics for paired-eye VCTR."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def miae(A_true: np.ndarray, A_hat: np.ndarray) -> float:
    """Mean integrated absolute error for estimated coefficient functions."""

    return mae(A_true, A_hat)


def rmise(A_true: np.ndarray, A_hat: np.ndarray) -> float:
    """Root mean integrated squared error for estimated coefficient functions."""

    return rmse(A_true, A_hat)


def beta_mae(beta_true: np.ndarray, beta_hat: np.ndarray) -> float:
    """MAE for estimated scalar coefficients."""

    return mae(beta_true, beta_hat)


def beta_rmse(beta_true: np.ndarray, beta_hat: np.ndarray) -> float:
    """RMSE for estimated scalar coefficients."""

    return rmse(beta_true, beta_hat)
