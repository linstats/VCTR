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


def sigma2_abs_error(sigma2_true: float, sigma2_hat: float) -> float:
    """Absolute error for the shared marginal variance estimate."""

    return float(abs(float(sigma2_true) - float(sigma2_hat)))


def rho_abs_error(rho_true: float, rho_hat: float) -> float:
    """Absolute error for the shared correlation estimate."""

    return float(abs(float(rho_true) - float(rho_hat)))


def sigma_frobenius_error(Sigma_true: np.ndarray, Sigma_hat: np.ndarray) -> float:
    """Frobenius-norm error for the shared covariance matrix estimate."""

    Sigma_true = np.asarray(Sigma_true, dtype=float)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    return float(np.linalg.norm(Sigma_true - Sigma_hat, ord="fro"))
