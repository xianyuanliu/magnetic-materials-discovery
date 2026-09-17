"""Per-split scoring and paired significance testing.

Every function here returns data. Formatting and printing live in reporting.py, so these can be called from a notebook
or another library without a run's console output appearing as a side effect.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


from utils.registry import ModelSpec
from evaluate.metrics import METRICS, compute_metrics
from loaddata.splits import Split

# Below this many paired observations a signed-rank test cannot reach any conventional significance level at all: with n
# pairs the smallest attainable two-sided Wilcoxon p is 2 / 2**n, so n = 3 bottoms out at 0.25 and n = 5 at 0.0625.
# Reporting "p = 0.25, not significant" from three pairs reads as evidence of no difference when it is really the floor
# of the test.
MIN_PAIRS_FOR_TEST = 6

# Per-model, per-metric fold scores: {model name: {metric: [score per fold]}}.
FoldScores = Dict[str, Dict[str, List[float]]]


@dataclass(frozen=True)
class SignificanceResult:
    """Outcome of a paired comparison between two models.

    Attributes:
        metric: Which metric was compared.
        model_a, model_b: The compared model names.
        n_pairs: Number of paired observations behind the test.
        t_stat, t_pvalue: Paired t-test on the differences.
        w_stat, w_pvalue: Wilcoxon signed-rank on the same differences.
        mean_difference: mean(a) - mean(b); negative favors `model_a` for lower-is-better metrics.
        note: Why the p-values are absent or should not be read, when that applies; None when the test ran normally.
    """

    metric: str
    model_a: str
    model_b: str
    n_pairs: int
    t_stat: float
    t_pvalue: float
    w_stat: float
    w_pvalue: float
    mean_difference: float
    note: Optional[str] = None


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    specs: Sequence[ModelSpec],
    splits: Sequence[Split],
    hyperparameter_tuning: bool = False,
    best_hyperparams: Optional[Mapping[str, Dict]] = None,
    model_random_state: int = 0,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
) -> FoldScores:
    """Score the given models on splits the caller has already built.

    Which rows form each fold is not decided here: the caller passes splits from loaddata/splits.py, so a plain K-fold
    and a grouped or shifted split go through this same scoring path.

    When hyperparameter_tuning is set, the search re-runs inside every split (proper nested CV — a split's validation
    rows never inform its own search). That is why the search budget is a separate, smaller pair of knobs: tune_cv_folds
    inner folds and tune_n_iter sampled candidates, whose cost is multiplied by len(splits) x len(specs). Pass
    best_hyperparams to skip the search entirely and reuse one fixed set of parameters.

    Args:
        X: Feature matrix.
        y: Target.
        specs: Resolved model specifications to score.
        splits: (split_id, train_idx, valid_idx) tuples; the identifier is unused, scores are returned per split in
            the order given.
        hyperparameter_tuning: Search inside every split.
        best_hyperparams: Fixed parameters per model key, bypassing the search.
        model_random_state: Seed for model construction and the search.
        tune_cv_folds: Inner folds for the search.
        tune_n_iter: Candidates sampled by a randomized search.

    Returns:
        {model name: {metric: [one score per split]}}.
    """
    results: FoldScores = {
        spec.name: {metric: [] for metric in METRICS} for spec in specs
    }

    for _, train_idx, valid_idx in splits:
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        for spec in specs:
            hyperparams = None
            if best_hyperparams is not None and spec.key in best_hyperparams:
                hyperparams = best_hyperparams[spec.key]
            elif hyperparameter_tuning and spec.tune is not None:
                hyperparams = spec.tune(
                    X_train,
                    y_train,
                    cv_folds=tune_cv_folds,
                    random_state=model_random_state,
                    n_iter=tune_n_iter,
                )

            model = spec.train(X_train, y_train, hyperparams=hyperparams, random_state=model_random_state)
            metrics = compute_metrics(y_valid, model.predict(X_valid))

            for metric in METRICS:
                results[spec.name][metric].append(metrics[metric])

    return results


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


def _corrected_ttest(a: np.ndarray, b: np.ndarray, test_train_ratio: float):
    """Paired t-test whose variance accounts for observations that share training data.

    Resampled validation reuses most of the data in every round — two of ten K-fold training sets overlap by 89% — so
    the K differences are correlated and the textbook paired t-test, which assumes them independent, reports p-values
    that are too small. Nadeau and Bengio (2003) correct this by scaling the variance of the mean difference from
    1/K to 1/K + n_test/n_train. At `test_train_ratio` 0 the formula reduces exactly to `scipy.stats.ttest_rel`.

    Returns:
        (t_stat, p_value), two-sided, on len(a) - 1 degrees of freedom.
    """
    differences = a - b
    n = len(differences)
    variance = float(np.var(differences, ddof=1))
    if variance == 0.0:
        return np.nan, np.nan

    t_stat = float(np.mean(differences)) / np.sqrt((1.0 / n + test_train_ratio) * variance)
    p_value = 2.0 * stats.t.sf(abs(t_stat), df=n - 1)
    return t_stat, float(p_value)


def compare_models_significance(
    results: FoldScores,
    model_a: str,
    model_b: str,
    metric: str = "mse",
    min_pairs: int = MIN_PAIRS_FOR_TEST,
    test_train_ratio: float = 0.0,
) -> SignificanceResult:
    """Paired t-test and Wilcoxon signed-rank on two models' scores.

    Two things have to hold for the p-values to mean anything, and only one of them is checked here. Each observation
    must come from a different evaluation set — the caller is responsible for not passing in scores that share a test
    set. The observations must also not share training data, which K-fold and repeated holdout both violate; pass
    `test_train_ratio` so the t-test can correct for it.

    Args:
        results: Paired scores, keyed by model name then metric.
        model_a, model_b: Model names to compare.
        metric: Which metric to compare; lower-is-better metrics only.
        min_pairs: Below this, p-values are withheld and `note` explains why; see MIN_PAIRS_FOR_TEST.
        test_train_ratio: n_test / n_train for one observation, or 0 when the observations share no training data.
            1 / (K - 1) for K-fold, (1 - train_size) / train_size for repeated holdout.

    Returns:
        A SignificanceResult. `mean_difference` is always populated; the
        p-values are NaN when the test could not or should not be run.

    Raises:
        ValueError: If a model or metric is missing, or the counts differ.
    """
    a, b = _paired_scores(results, model_a, model_b, metric)
    mean_difference = float(np.mean(a) - np.mean(b))

    def _result(t_stat, t_p, w_stat, w_p, note=None):
        return SignificanceResult(
            metric=metric, model_a=model_a, model_b=model_b, n_pairs=len(a),
            t_stat=float(t_stat), t_pvalue=float(t_p),
            w_stat=float(w_stat), w_pvalue=float(w_p),
            mean_difference=mean_difference, note=note,
        )

    if len(a) < min_pairs:
        return _result(
            np.nan, np.nan, np.nan, np.nan,
            note=(
                f"only {len(a)} paired observation(s); a signed-rank test needs at "
                f"least {min_pairs} before any p-value below 0.05 is attainable"
            ),
        )

    t_stat, t_p = _corrected_ttest(a, b, test_train_ratio)

    if not np.any(a != b):
        return _result(t_stat, t_p, np.nan, np.nan, note="all differences are zero")

    w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox")
    # No signed-rank equivalent of the Nadeau-Bengio correction exists, so the Wilcoxon p-value stays optimistic
    # whenever the observations share training data.
    note = (
        "t-test variance corrected for shared training data; the Wilcoxon p-value is uncorrected and reads optimistic"
        if test_train_ratio > 0 else None
    )
    return _result(t_stat, t_p, w_stat, w_p, note=note)
