"""
OOD split

Design goals:
- deterministic given seed
- strict leakage prevention
- simple and auditable
"""

from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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

    KMeans clustering on feature space
    Test = one cluster
    Train = remaining clusters
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