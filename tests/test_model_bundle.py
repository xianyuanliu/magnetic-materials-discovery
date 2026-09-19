"""Covers the one failure a saved model cannot announce: being applied to data it was not trained on.

A stale bundle loads cleanly and predicts happily; only the numbers are wrong, and they look like every other number
the model produces. So what is pinned here is that the mismatch is detected — including the shapes of it that leave
the row count unchanged, which a row count alone would miss.
"""

import unittest

import numpy as np
import pandas as pd

from pipeline.prediction import check_bundle_matches
from utils.persistence import TRAINING_FINGERPRINT, ModelBundle, fingerprint_training_data


def training_table(n_rows: int = 5, shift: float = 0.0):
    """A small feature table and target, shifted so two calls can differ in values but not in shape."""
    X = pd.DataFrame({"a": np.arange(n_rows, dtype=float) + shift, "b": np.ones(n_rows)})
    y = pd.Series(np.arange(n_rows, dtype=float))
    return X, y


def bundle_fitted_on(X: pd.DataFrame, y: pd.Series) -> ModelBundle:
    """A bundle carrying the fingerprint of `X`/`y`; the estimator itself is never called here."""
    return ModelBundle(
        model=None,
        model_key="rf",
        model_name="Random Forest",
        feature_columns=tuple(X.columns),
        target_column="saturation magnetization",
        dataset="Novamag",
        n_train=len(X),
        metadata={TRAINING_FINGERPRINT: fingerprint_training_data(X, y)},
    )


class BundleFreshnessTests(unittest.TestCase):
    """`check_bundle_matches` is the only thing standing between a corrected dataset and a stale model."""

    def test_the_same_data_passes(self):
        X, y = training_table()
        check_bundle_matches(bundle_fitted_on(X, y), X, y)

    def test_fewer_rows_are_rejected(self):
        """The case that occurred: the dataset was deduplicated from 460 rows to 327 and the model was not refitted."""
        X, y = training_table()
        with self.assertRaises(ValueError):
            check_bundle_matches(bundle_fitted_on(X, y), *training_table(n_rows=3))

    def test_changed_values_are_rejected(self):
        """Same rows, same columns, different numbers — a row count would not notice."""
        X, y = training_table()
        with self.assertRaises(ValueError):
            check_bundle_matches(bundle_fitted_on(X, y), *training_table(shift=1.0))

    def test_a_renamed_feature_is_rejected(self):
        X, y = training_table()
        with self.assertRaises(ValueError):
            check_bundle_matches(bundle_fitted_on(X, y), X.rename(columns={"a": "z"}), y)

    def test_reordered_features_are_rejected(self):
        """The model takes its features positionally, so the order is part of what it was trained on."""
        X, y = training_table()
        with self.assertRaises(ValueError):
            check_bundle_matches(bundle_fitted_on(X, y), X[["b", "a"]], y)

    def test_a_bundle_saved_before_fingerprints_warns_instead_of_raising(self):
        """Refusing every older bundle would be worse than saying it cannot be checked."""
        X, y = training_table()
        bundle = bundle_fitted_on(X, y)
        bundle.metadata.pop(TRAINING_FINGERPRINT)
        with self.assertWarns(UserWarning):
            check_bundle_matches(bundle, X, y)


if __name__ == "__main__":
    unittest.main()
