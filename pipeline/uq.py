"""Predictive-uncertainty estimators and split-conformal calibration.

Three ways to attach an interval to a point prediction, in increasing order of how much they promise:

  rf_std            Spread of the forest's own trees, read as a Gaussian sigma.
                    Cheap and interpretable, but it only measures disagreement
                    between trees — it carries no noise or bias term, so it has
                    no reason to be calibrated and is included as the naive
                    reference, not as a recommendation.
  conformal         Split conformal on absolute residuals. Constant width, and
                    the only one of the three with a finite-sample coverage
                    guarantee (in-distribution, exchangeable data).
  conformal_norm    Split conformal on residuals divided by rf_std. Keeps the
                    guarantee while letting the width follow the model's own
                    uncertainty, so it is the adaptive baseline rf_std should
                    be judged against.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

from utils.core import DEFAULT_ALPHA

# rf_std is exactly 0 where every tree agrees, and the normalized variant divides by it. Flooring at a small fraction of
# the calibration set's mean keeps that score bounded while leaving ordinary sigmas untouched; the absolute fallback
# only applies if every calibration sigma is 0.
_SIGMA_FLOOR_FRACTION = 1e-3
_MIN_SIGMA = 1e-12


def _sigma_floor(sigma: np.ndarray) -> float:
    """A positive lower bound for sigma, scaled to the data it came from."""
    mean = float(np.mean(sigma))
    return _SIGMA_FLOOR_FRACTION * mean if mean > 0 else _MIN_SIGMA


RF_STD = "rf_std"
CONFORMAL = "conformal"
CONFORMAL_NORM = "conformal_norm"


def rf_tree_std(model, X) -> np.ndarray:
    """Standard deviation of the per-tree predictions of a fitted forest.

    Args:
        model: A fitted RandomForestRegressor. Tree ensembles are trained unscaled (see pipeline/model.py:_scaled), so
            `estimators_` accept the same X as the forest itself.
        X: Feature matrix to predict.

    Raises:
        TypeError: If `model` exposes no `estimators_`.
    """
    if not hasattr(model, "estimators_"):
        raise TypeError(f"{type(model).__name__} has no estimators_; rf_std needs a forest.")

    # The forest fits its trees on a bare array, so hand them one too — a
    # DataFrame here only earns a feature-names warning per tree per call.
    values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    return np.stack([tree.predict(values) for tree in model.estimators_]).std(axis=0)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected (1 - alpha) quantile of calibration scores.

    The ceil((n+1)(1-alpha))/n level, rather than a plain quantile, is what
    makes split conformal's coverage guarantee hold at finite n.
    """
    n = scores.size
    if n == 0:
        raise ValueError("Conformal calibration needs at least one residual.")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


@dataclass(frozen=True)
class ConformalCalibrator:
    """A fitted split-conformal interval width.

    Attributes:
        quantile: Calibrated score quantile — an absolute half-width when
            `normalized` is False, a multiplier on sigma when it is True.
        normalized: Whether widths scale with the model's own sigma.
        sigma_floor: Lower bound applied to sigma, fixed at calibration time so
            fit and predict divide by the same thing. Unused when not normalized.
    """

    quantile: float
    normalized: bool
    sigma_floor: float = 0.0

    @classmethod
    def fit(
        cls,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
        alpha: float = DEFAULT_ALPHA,
        sigma_cal: Optional[np.ndarray] = None,
    ) -> "ConformalCalibrator":
        """Calibrate on residuals the model never trained on.

        Args:
            y_cal: Targets of the calibration set.
            y_pred_cal: Predictions for the calibration set.
            alpha: Nominal miscoverage.
            sigma_cal: Per-sample sigma on the calibration set. Supplying it
                switches to the normalized (locally adaptive) variant.
        """
        residuals = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(y_pred_cal, dtype=float))
        normalized = sigma_cal is not None
        floor = 0.0
        if normalized:
            sigma_cal = np.asarray(sigma_cal, dtype=float)
            floor = _sigma_floor(sigma_cal)
            residuals = residuals / np.maximum(sigma_cal, floor)

        return cls(quantile=_conformal_quantile(residuals, alpha), normalized=normalized, sigma_floor=floor)

    def half_width(self, sigma: np.ndarray) -> np.ndarray:
        """Per-sample interval half-width for the test points behind `sigma`."""
        sigma = np.asarray(sigma, dtype=float)
        if self.normalized:
            return self.quantile * np.maximum(sigma, self.sigma_floor)
        return np.full(sigma.shape, self.quantile, dtype=float)


def gaussian_half_width(sigma: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Read a sigma estimate as a two-sided Gaussian interval half-width at `alpha`."""
    z = norm.ppf(1.0 - alpha / 2.0)
    return z * np.asarray(sigma, dtype=float)
