"""Select magnetic records, aggregate equivalent compositions, and calculate model features.

The experiment predicts magnetization among selected magnetic records. Target thresholds, magnetic labels and element
exclusions therefore apply BEFORE the median is calculated. Source format cleaning belongs to loaddata/; all task
selection is explicit here. Target-free inference uses alloy_descriptors.add_engineered_features directly.
"""

import warnings
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from prepdata.alloy_descriptors import ENGINEERED_FEATURE_COLUMNS, add_engineered_features
from prepdata.composition import formula_contains_elements, get_composition_key


# Optional experiment exclusions, retained from the original MP preparation policy.
RARE_EARTH_ELEMENTS = (
    "La", "Sc", "Dy", "Sm", "Lu", "Er", "Y", "Pr", "Nd", "Gd", "Tm", "Pm", "Ce", "Tb", "Eu", "Ho", "Yb",
)
NON_COMMERCIAL_ELEMENTS = (
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
)


# Carried through for traceability and for a later modality to key off. Never model inputs: they are excluded from the
# feature fallback in loaddata.tabular_access.resolve_feature_columns.
PROVENANCE_COLUMNS = ("n_records", "sample_id")


def _describe_sources(data: pd.DataFrame, targets: pd.Series, target_column: str) -> pd.DataFrame:
    """Record how many source rows each composition aggregates, and which one the median came from.

    The median of an even number of differing values is the mean of the middle two, so no single source produced it and
    `sample_id` is left empty. Writing a nearby id instead would pair a later modality with a structure whose
    measurement is not the target being predicted.
    """
    grouped = data.groupby("composition_key", sort=True)
    provenance = pd.DataFrame({"n_records": grouped.size()})
    provenance.index.name = targets.index.name

    if "sample_id" in data.columns:
        median_of = data["composition_key"].map(targets)
        exact = data.loc[data[target_column] == median_of, ["composition_key", "sample_id"]]
        first = exact.groupby("composition_key")["sample_id"].first()
        provenance["sample_id"] = first.reindex(provenance.index)

    return provenance


def filter_samples(
    data: pd.DataFrame,
    min_target: Optional[float] = 0.18,
    excluded_elements: Optional[Iterable[str]] = None,
    magnetic_only: bool = False,
    target_column: str = "saturation magnetization",
    formula_column: str = "chemical formula",
) -> pd.DataFrame:
    """Select records before aggregation, preserving metadata and leaving the input unchanged.

    min_target is inclusive; None disables the threshold. magnetic_only additionally requires an explicit True in the
    standardized boolean is_magnetic column. Missing magnetic labels do not qualify; an absent column raises.
    Non-finite/missing targets and missing formulas cannot form training samples and are always removed.
    """
    required = [formula_column, target_column] + (["is_magnetic"] if magnetic_only else [])
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) for sample selection: {missing}")
    selected = data.copy()
    selected[target_column] = pd.to_numeric(selected[target_column], errors="coerce")
    selected = selected.loc[
        np.isfinite(selected[target_column]) & selected[formula_column].notna()
    ].copy()
    if min_target is not None:
        selected = selected.loc[selected[target_column] >= min_target]
    if magnetic_only:
        if not pd.api.types.is_bool_dtype(selected["is_magnetic"]):
            raise ValueError("is_magnetic must contain standardized boolean values")
        selected = selected.loc[selected["is_magnetic"].fillna(False)]
    excluded = tuple(excluded_elements) if excluded_elements is not None else ()
    if excluded:
        selected = selected.loc[~formula_contains_elements(selected, excluded, formula_column=formula_column)]
    return selected.reset_index(drop=True)


def build_feature_table(
    raw_data: pd.DataFrame,
    pt: pd.DataFrame,
    mm: pd.DataFrame,
    min_target: Optional[float] = 0.18,
    target_column: str = "saturation magnetization",
    formula_column: str = "chemical formula",
    *,
    excluded_elements: Optional[Iterable[str]] = None,
    magnetic_only: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build one training sample per normalized composition from selected magnetic records.

    Filtering precedes aggregation: the target is the median of qualifying records, including distinct structures of
    the same composition. Equivalent formulas (FeNi, Fe2Ni2, NiFe) share a composition key. Only the target is
    aggregated. Features are calculated once per composition and metadata never becomes a model input or a median
    operand.

    Returns a table indexed by composition key (index name formula_column), with the target and nine feature columns,
    plus the ordered feature list. Original formulas and metadata remain unchanged in raw_data.
    """
    data = filter_samples(
        raw_data, min_target=min_target, excluded_elements=excluded_elements, magnetic_only=magnetic_only,
        target_column=target_column, formula_column=formula_column,
    )
    # Grouping on the literal formula is not enough: Co5Ta1 and Co10Ta2 are the same composition written at different
    # supercell multiples, so they yield identical features. Left as separate rows they straddle a random split and the
    # model is scored on inputs it memorized.
    keys = {formula: get_composition_key(formula) for formula in data[formula_column].unique()}
    data["composition_key"] = data[formula_column].map(keys)
    data = data.dropna(subset=["composition_key"])
    # Keep one median target per normalized composition, without discarding distinct source measurements beforehand.
    grouped = data.groupby("composition_key", sort=True)
    targets = grouped[target_column].median()
    targets.index.name = formula_column
    provenance = _describe_sources(data, targets, target_column)
    data = add_engineered_features(targets.reset_index(), pt, mm, formula_column=formula_column)
    data = data.join(provenance, on=formula_column)

    feature_columns = list(ENGINEERED_FEATURE_COLUMNS)
    data = data.replace([np.inf, -np.inf], np.nan)
    incomplete = data[feature_columns].isna().any(axis=1)
    if incomplete.any():
        unusable = sorted(data.loc[incomplete, feature_columns].isna().any().loc[lambda s: s].index)
        warnings.warn(
            f"Dropped {int(incomplete.sum())} of {len(data)} compositions with features that could not be "
            f"calculated from the reference tables: {unusable}."
        )
    kept = data.loc[~incomplete].set_index(formula_column)
    carried = [column for column in PROVENANCE_COLUMNS if column in kept.columns]
    return kept[[target_column] + feature_columns + carried], feature_columns
