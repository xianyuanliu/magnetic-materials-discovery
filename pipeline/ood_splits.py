"""Builders for out-of-distribution (OOD) train/test splits.

Each build_*_splits function returns a list of (split_id, train_idx, test_idx)
tuples. Splits are deterministic where seeded.

Two of the builders describe in-distribution references rather than shifts —
build_kfold_splits and build_size_matched_split — so that an OOD score can be
compared against a baseline produced by the identical evaluation code path.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from evaluate.ood_evaluation import Split


def _standardized(X: pd.DataFrame) -> np.ndarray:
    """Return X as a z-scored array, so Euclidean distances weigh every feature equally."""
    # Fitted on all of X on purpose: these are unsupervised split definitions, not model inputs, so nothing leaks.
    return StandardScaler().fit_transform(X.to_numpy())


def _finalize_split(
    split_id: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    min_train: int = 1,
    min_test: int = 1,
) -> Optional[Split]:
    """Return (split_id, train_idx, test_idx), or None if below the minimum size."""
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    if train_idx.size < min_train or test_idx.size < min_test:
        return None

    return split_id, train_idx, test_idx


# ============================================================ In-distribution references
# ============================================================


def build_kfold_splits(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 0,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Plain K-fold, expressed as splits so the ID baseline reuses the OOD code path."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)

    splits: List[Split] = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples))):
        split = _finalize_split(f"ID_fold{fold}", train_idx, test_idx, min_train, min_test)
        if split is not None:
            splits.append(split)

    return splits


def build_size_matched_split(
    n_samples: int,
    n_train: int,
    n_test: int,
    seed: int,
    split_id: str = "RandomControl",
) -> Optional[Split]:
    """A random disjoint train/test split with prescribed sizes.

    The in-distribution control for one OOD split: same amount of training data and same test-set size, but drawn at
    random. The OOD-minus-control difference is therefore attributable to the shift, not to the smaller training set an
    OOD split leaves behind (holding out Fe on Novamag costs more than half the training data).

    Returns:
        The split, or None if the requested sizes do not fit in `n_samples`.
    """
    if n_train + n_test > n_samples:
        return None

    order = np.random.default_rng(seed).permutation(n_samples)
    return _finalize_split(split_id, order[:n_train], order[n_train:n_train + n_test])


# ============================================================ Leave-One-Element-Out (LOEO)
# ============================================================


def build_loeo_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_list: Sequence[str],
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Leave-One-Element-Out: test = samples containing element E, train = the rest."""
    splits: List[Split] = []

    for element in element_list:

        test_mask = np.array([element in set(els) for els in elements_per_sample], dtype=bool)

        split = _finalize_split(
            f"E={element}",
            np.where(~test_mask)[0],
            np.where(test_mask)[0],
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits


# ============================================================ Leave-One-Period-Out (LOPO) / Leave-One-Group-Out (LOGO)
# ============================================================


def _build_membership_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_to_attr: Dict[str, int],
    attr_values: Sequence[int],
    label: str,
    strict: bool,
    min_train: int,
    min_test: int,
) -> List[Split]:
    """Shared LOPO/LOGO logic, keyed by an element->period or element->group map.

    Default (strict=False): test = samples containing ANY element with this
    attribute value. strict=True: test = samples where ALL elements do.
    """
    splits: List[Split] = []

    for value in attr_values:

        if strict:
            test_mask = np.array(
                [
                    bool(els) and all(element_to_attr.get(e) == value for e in set(els))
                    for els in elements_per_sample
                ],
                dtype=bool,
            )
        else:
            heldout_elements = {e for e, v in element_to_attr.items() if v == value}
            test_mask = np.array(
                [len(set(els).intersection(heldout_elements)) > 0 for els in elements_per_sample],
                dtype=bool,
            )

        split = _finalize_split(
            f"{label}={value}",
            np.where(~test_mask)[0],
            np.where(test_mask)[0],
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits


def build_period_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_to_period: Dict[str, int],
    periods: Sequence[int],
    strict: bool = False,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Leave-One-Period-Out (see _build_membership_splits for strict/default semantics)."""
    return _build_membership_splits(elements_per_sample, element_to_period, periods, "P", strict, min_train, min_test)


def build_group_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_to_group: Dict[str, int],
    groups: Sequence[int],
    strict: bool = False,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Leave-One-Group-Out (see _build_membership_splits for strict/default semantics)."""
    return _build_membership_splits(elements_per_sample, element_to_group, groups, "G", strict, min_train, min_test)


# ============================================================ Representation Space OOD — KMeans (LOCO)
# ============================================================


def build_kmeans_cluster_splits(
    X: pd.DataFrame,
    k: int = 10,
    seed: int = 0,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Representation-space OOD (LOCO): KMeans on X, test = one cluster, train = the rest.

    Cluster assignments are computed unsupervised on the full feature matrix before the train/test split, as a pragmatic
    stress test rather than a strict train-only clustering protocol.
    """

    if k < 2:
        raise ValueError("k must be >= 2")

    X_mat = _standardized(X)

    km = KMeans(n_clusters=k, random_state=int(seed), n_init=10)

    labels = km.fit_predict(X_mat)

    splits: List[Split] = []

    for c in range(k):

        split = _finalize_split(
            f"C={c}",
            np.where(labels != c)[0],
            np.where(labels == c)[0],
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits


# ============================================================ SparseX OOD — Feature-space sparsity
# ============================================================


def build_sparsex_splits(
    X: pd.DataFrame,
    fractions: Sequence[float] = (0.1, 0.2),
    n_neighbors: int = 5,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """OOD splits based on feature-space sparsity: hold out the most isolated samples.

    "Isolated" = largest mean distance to its `n_neighbors` nearest neighbors in X.
    One split is built per fraction in `fractions`.
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")

    X_mat = _standardized(X)
    n = X_mat.shape[0]

    if n < 2:
        raise ValueError("SparseX requires at least 2 samples")

    # +1 because nearest neighbor includes the point itself at distance 0
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n))
    nn.fit(X_mat)
    distances, _ = nn.kneighbors(X_mat)

    # Exclude self-distance (first column = 0)
    if distances.shape[1] > 1:
        mean_dist = distances[:, 1:].mean(axis=1)
    else:
        mean_dist = distances[:, 0]

    order = np.argsort(mean_dist)[::-1]  # descending: sparsest first
    splits: List[Split] = []

    for frac in fractions:
        if not (0 < float(frac) < 1):
            raise ValueError(f"Each fraction must be in (0, 1), got {frac}")

        n_test = max(1, int(round(n * float(frac))))

        split = _finalize_split(
            f"SparseX_top{int(round(frac * 100))}pct",
            np.sort(order[n_test:]),
            np.sort(order[:n_test]),
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits


# ============================================================ SparseY OOD — Target-space sparsity
# ============================================================


def build_sparsey_splits(
    y: pd.Series,
    fractions: Sequence[float] = (0.1, 0.2),
    center: str = "median",
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """OOD splits based on target-space sparsity: hold out the most extreme y values.

    "Extreme" = farthest from `center` ("median" or "mean"). One split is built per fraction in `fractions`.
    """
    y_arr = np.asarray(y, dtype=float)
    n = y_arr.shape[0]

    if center == "median":
        ref = np.median(y_arr)
    elif center == "mean":
        ref = np.mean(y_arr)
    else:
        raise ValueError("center must be 'median' or 'mean'")

    extremeness = np.abs(y_arr - ref)
    order = np.argsort(extremeness)[::-1]  # descending: most extreme first
    splits: List[Split] = []

    for frac in fractions:
        if not (0 < float(frac) < 1):
            raise ValueError(f"Each fraction must be in (0, 1), got {frac}")

        n_test = max(1, int(round(n * float(frac))))

        split = _finalize_split(
            f"SparseY_top{int(round(frac * 100))}pct",
            np.sort(order[n_test:]),
            np.sort(order[:n_test]),
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits
