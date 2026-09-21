"""Paired significance testing between two models scored on the same splits."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from evaluate.metrics import FoldScores

MIN_PAIRS_FOR_TEST = 6


@dataclass(frozen=True)
class SignificanceResult:
    """Outcome of a paired comparison between two models.

    Attributes:
        metric: Which metric was compared.
        model_a, model_b: The compared model names.
        n_pairs: Number of paired observations behind the test.
        t_stat, t_pvalue: Paired t-test on the differences, on the corrected variance.
        mean_difference: mean(a) - mean(b); negative favors `model_a` for lower-is-better metrics.
        ci_low, ci_high: Confidence interval on `mean_difference`, in the metric's own units and on the same corrected
            variance as `t_pvalue`. An interval clear of zero is what a p-value below the same level reports, but it
            also says by how much and how precisely, which is what a result table needs.
        note: Why a p-value is absent, or a caveat on one that was computed. The p-values themselves say which
            it is: a NaN means the test did not run.
    """

    metric: str
    model_a: str
    model_b: str
    n_pairs: int
    t_stat: float
    t_pvalue: float
    mean_difference: float
    ci_low: float = np.nan
    ci_high: float = np.nan
    note: Optional[str] = None


def _paired_scores(results: FoldScores, model_a: str, model_b: str, metric: str):
    """Return the two models' score arrays for `metric`, checking they pair up.

    Raises:
        ValueError: If a model or metric is missing, or the lengths differ.
    """
    for name in (model_a, model_b):
        if name not in results:
            raise ValueError(f"Model {name!r} not in results: {sorted(results)}")
        if metric not in results[name]:
            raise ValueError(f"Metric {metric!r} not recorded for {name!r}.")

    a = np.asarray(results[model_a][metric], dtype=float)
    b = np.asarray(results[model_b][metric], dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Paired count mismatch: {model_a} has {len(a)}, {model_b} has {len(b)}")
    return a, b


def _corrected_ttest(a: np.ndarray, b: np.ndarray, test_train_ratio: float, confidence: float = 0.95):
    """Paired t-test whose variance accounts for observations that share training data.

    Resampled validation reuses most of the data in every round — two of ten K-fold training sets overlap by 89% — so
    the K differences are correlated and the textbook paired t-test, which assumes them independent, reports p-values
    that are too small. Nadeau and Bengio (2003) correct this by scaling the variance of the mean difference from
    1/K to 1/K + n_test/n_train. At `test_train_ratio` 0 the formula reduces exactly to `scipy.stats.ttest_rel`.

    Args:
        a, b: Paired scores, one per observation.
        test_train_ratio: n_test / n_train for one observation, or 0 when no training data is shared.
        confidence: Two-sided level for the returned interval.

    Returns:
        (t_stat, p_value, ci_low, ci_high), two-sided, on len(a) - 1 degrees of freedom. The interval is on the mean
        difference and uses the same corrected standard error as the p-value, so the two always agree.
    """
    differences = a - b
    n = len(differences)
    variance = float(np.var(differences, ddof=1))
    mean_difference = float(np.mean(differences))
    if variance == 0.0:
        return np.nan, np.nan, np.nan, np.nan

    standard_error = np.sqrt((1.0 / n + test_train_ratio) * variance)
    t_stat = mean_difference / standard_error
    p_value = 2.0 * stats.t.sf(abs(t_stat), df=n - 1)
    half_width = stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * standard_error
    return t_stat, float(p_value), mean_difference - half_width, mean_difference + half_width


def compare_models_significance(
    results: FoldScores,
    model_a: str,
    model_b: str,
    metric: str = "mse",
    min_pairs: int = MIN_PAIRS_FOR_TEST,
    test_train_ratio: float = 0.0,
) -> SignificanceResult:
    """Paired comparison of two models: the mean difference, its confidence interval, and a t-test.

    Two things have to hold for the p-values to mean anything, and only one of them is checked here. Each observation
    must come from a different evaluation set — the caller is responsible for not passing in scores that share a test
    set. The observations must also not share training data, which K-fold and repeated random splits both violate; pass
    `test_train_ratio` so the t-test can correct for it.

    Args:
        results: Paired scores, keyed by model name then metric.
        model_a, model_b: Model names to compare.
        metric: Which metric to compare; lower-is-better metrics only.
        min_pairs: Below this, p-values are withheld and `note` explains why; see MIN_PAIRS_FOR_TEST.
        test_train_ratio: n_test / n_train for one observation, or 0 when the observations share no training data.
            1 / (K - 1) for K-fold, (1 - train_size) / train_size for repeated random splits.

    Returns:
        A SignificanceResult. `mean_difference` is always populated; the
        p-values are NaN when the test could not or should not be run.

    Raises:
        ValueError: If a model or metric is missing, or the counts differ.
    """
    a, b = _paired_scores(results, model_a, model_b, metric)
    mean_difference = float(np.mean(a) - np.mean(b))

    def _result(t_stat, t_p, ci=(np.nan, np.nan), note=None):
        return SignificanceResult(
            metric=metric, model_a=model_a, model_b=model_b, n_pairs=len(a),
            t_stat=float(t_stat), t_pvalue=float(t_p),
            mean_difference=mean_difference, ci_low=float(ci[0]), ci_high=float(ci[1]), note=note,
        )

    if len(a) < min_pairs:
        return _result(
            np.nan, np.nan,
            note=f"only {len(a)} paired observation(s); an interval on fewer than {min_pairs} constrains nothing",
        )

    t_stat, t_p, ci_low, ci_high = _corrected_ttest(a, b, test_train_ratio)
    ci = (ci_low, ci_high)

    if not np.any(a != b):
        return _result(t_stat, t_p, ci=ci, note="all differences are zero")

    note = "variance corrected for observations that share training data" if test_train_ratio > 0 else None
    return _result(t_stat, t_p, ci=ci, note=note)
