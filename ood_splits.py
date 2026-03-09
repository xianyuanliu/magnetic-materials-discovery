"""
OOD split

Design goals:
- deterministic where seeded
- simple and auditable
- explicit split construction rules
"""

from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


Split = Tuple[str, np.ndarray, np.ndarray]


def _validate_split(
    split_id: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_samples: int,
    min_train: int = 1,
    min_test: int = 1,
) -> Optional[Split]:
    """
    Validate split integrity:
    - size constraints
    - no leakage
    - index bounds
    """
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    if train_idx.size < min_train or test_idx.size < min_test:
        return None

    # No overlap between train and test
    if np.intersect1d(train_idx, test_idx).size > 0:
        raise ValueError(f"Leakage detected in split {split_id}")

    # Bounds safety
    if train_idx.max(initial=-1) >= n_samples or test_idx.max(initial=-1) >= n_samples:
        raise ValueError(f"Index out of bounds in split {split_id}")

    return split_id, train_idx, test_idx


# ============================================================
# Leave-One-Element-Out (LOEO)
# ============================================================

def build_loeo_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_list: Sequence[str],
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """
    Leave-One-Element-Out (LOEO)

    Test = samples containing element E
    Train = samples NOT containing E
    """
    n = len(elements_per_sample)
    splits: List[Split] = []

    for element in element_list:

        test_mask = np.array(
            [element in set(els) for els in elements_per_sample],
            dtype=bool,
        )

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]

        split = _validate_split(
            f"E={element}",
            train_idx,
            test_idx,
            n,
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits


# ============================================================
# Leave-One-Period-Out (LOPO)
# ============================================================

def build_period_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_to_period: Dict[str, int],
    periods: Sequence[int],
    strict: bool = False,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """
    Leave-One-Period-Out (LOPO)

    Default (strict=False):
        Test = samples containing ANY element from period P
        Train = samples containing NONE of those elements

    strict=True:
        Test = samples where ALL elements belong to period P
    """
    n = len(elements_per_sample)
    splits: List[Split] = []

    for p in periods:

        if strict:
            test_mask = []
            for els in elements_per_sample:
                els = list(set(els))
                if len(els) == 0:
                    test_mask.append(False)
                    continue

                ok = all(element_to_period.get(e) == p for e in els)
                test_mask.append(ok)

            test_mask = np.array(test_mask, dtype=bool)

        else:
            heldout_elements = {
                e for e, pe in element_to_period.items() if pe == p
            }

            test_mask = np.array(
                [
                    len(set(els).intersection(heldout_elements)) > 0
                    for els in elements_per_sample
                ],
                dtype=bool,
            )

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]

        split = _validate_split(
            f"P={p}",
            train_idx,
            test_idx,
            n,
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits


# ============================================================
# Leave-One-Group-Out (LOGO)
# ============================================================

def build_group_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_to_group: Dict[str, int],
    groups: Sequence[int],
    strict: bool = False,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """
    Leave-One-Group-Out (LOGO)

    Default (strict=False):
        Test = samples containing ANY element from group G
        Train = samples containing NONE of those elements

    strict=True:
        Test = samples where ALL elements belong to group G
    """
    n = len(elements_per_sample)
    splits: List[Split] = []

    for g in groups:

        if strict:
            test_mask = []
            for els in elements_per_sample:
                els = list(set(els))
                if len(els) == 0:
                    test_mask.append(False)
                    continue

                ok = all(element_to_group.get(e) == g for e in els)
                test_mask.append(ok)

            test_mask = np.array(test_mask, dtype=bool)

        else:
            heldout_elements = {
                e for e, ge in element_to_group.items() if ge == g
            }

            test_mask = np.array(
                [
                    len(set(els).intersection(heldout_elements)) > 0
                    for els in elements_per_sample
                ],
                dtype=bool,
            )

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]

        split = _validate_split(
            f"G={g}",
            train_idx,
            test_idx,
            n,
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits


# ============================================================
# Representation Space OOD — KMeans (LOCO)
# ============================================================

def build_kmeans_cluster_splits(
    X: pd.DataFrame,
    k: int = 10,
    seed: int = 0,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    
    """
    Representation-space OOD (LOCO)

    KMeans clustering on the full feature space.
    Test = one cluster
    Train = remaining clusters

    Note:
        Cluster assignments are derived unsupervised from the full feature matrix
        before the train/test partition is formed. This is intended as a pragmatic
        representation-space stress test, not a strict train-only clustering protocol.
    """

    if k < 2:
        raise ValueError("k must be >= 2")

    X_mat = X.to_numpy()

    km = KMeans(
        n_clusters=k,
        random_state=int(seed),
        n_init=10,
    )

    labels = km.fit_predict(X_mat)

    n = X.shape[0]
    splits: List[Split] = []

    for c in range(k):

        test_idx = np.where(labels == c)[0]
        train_idx = np.where(labels != c)[0]

        split = _validate_split(
            f"C={c}",
            train_idx,
            test_idx,
            n,
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits

# ============================================================
# SparseX OOD — Feature-space sparsity
# ============================================================

def build_sparsex_splits(
    X: pd.DataFrame,
    fractions: Sequence[float] = (0.1, 0.2),
    n_neighbors: int = 5,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """
    SparseX OOD splits based on feature-space sparsity.

    Idea:
        - compute average distance to k nearest neighbors in X
        - samples with largest distances are the sparsest / most isolated
        - hold out the top fraction as test set

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    fractions : sequence of float
        Fractions of sparsest samples to hold out (e.g. 0.1, 0.2).
    n_neighbors : int
        Number of neighbors used to estimate local density/sparsity.
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")

    X_mat = X.to_numpy()
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
        test_idx = np.sort(order[:n_test])
        train_idx = np.sort(order[n_test:])

        split = _validate_split(
            f"SparseX_top{int(round(frac * 100))}pct",
            train_idx,
            test_idx,
            n,
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits


# ============================================================
# SparseY OOD — Target-space sparsity
# ============================================================

def build_sparsey_splits(
    y: pd.Series,
    fractions: Sequence[float] = (0.1, 0.2),
    center: str = "median",
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """
    SparseY OOD splits based on sparsity in target/property space.

    Idea:
        - find samples with target values farthest from the central tendency
        - hold out the top fraction as test set

    Parameters
    ----------
    y : pd.Series
        Target values.
    fractions : sequence of float
        Fractions of most extreme samples to hold out.
    center : {"median", "mean"}
        Reference point used to define extremeness.
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
        test_idx = np.sort(order[:n_test])
        train_idx = np.sort(order[n_test:])

        split = _validate_split(
            f"SparseY_top{int(round(frac * 100))}pct",
            train_idx,
            test_idx,
            n,
            min_train=min_train,
            min_test=min_test,
        )

        if split is not None:
            splits.append(split)

    return splits