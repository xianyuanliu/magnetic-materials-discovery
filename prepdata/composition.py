"""Composition handling that any materials task can reuse.

Turns a chemical formula into structured, queryable quantities — parsed elements, a stoichiometry array, atomic
fractions — and the primitives that descriptor sets are built from (`get_weighted_property`, `get_mixing_entropy`).
Which descriptors a task actually wants is not decided here; see prepdata/alloy_descriptors.py for this project's nine.

A formula that cannot be parsed is warned about and carried through as an empty composition — an all-zero stoichiometry
row, which every descriptor below then reports as NaN rather than 0.0, so the dataset builder's dropna removes the row
instead of training on a fabricated all-zero feature vector.
"""

import re
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition


# --- Formula parsing ---


def get_composition_key(formula: str) -> Optional[str]:
    """Return the grouping key that equivalent formulas share: a parseable, scale-independent formula.

    Uses pymatgen's get_integer_formula_and_factor with its default max_denominator=10000 approximation for fractional
    amounts, then hill_formula for consistent formatting. Hill ordering writes sodium/nitrogen as NNa instead of NaN,
    which CSV readers treat as a missing value. FeNi, NiFe, Fe2Ni2 and Fe0.5Ni0.5 all yield FeNi. Invalid or empty
    formulas return None. This identifies composition, not structure; original formulas and source IDs remain in the
    loaded records.
    """
    if pd.isna(formula):
        return None
    try:
        composition = Composition(str(formula), strict=True)
        amounts = composition.get_el_amt_dict()
        if not amounts or any(not np.isfinite(amount) or amount <= 0 for amount in amounts.values()):
            raise ValueError("Element amounts must be finite and positive")
        total = sum(amounts.values())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Total element amount must be finite and positive")
        integer_formula, _ = composition.get_integer_formula_and_factor()
        return Composition(integer_formula).hill_formula.replace(" ", "")
    except (ValueError, TypeError) as exc:
        warnings.warn(f"Could not normalize chemical formula {formula!r}; dropping the row ({exc}).")
        return None


def get_elements(formula: str) -> List[str]:
    """Return the unique element symbols in one chemical formula.

    A missing formula (None/NaN) is not a failure and returns [] silently. A genuine parse failure warns, because an
    empty composition is invisible to every element/period/group-based OOD split.

    Example:
        Nd2Fe14B -> ["Nd", "Fe", "B"]
    """
    if pd.isna(formula):
        return []
    try:
        return [str(element) for element in Composition(str(formula)).elements]
    except Exception as exc:
        warnings.warn(f"Could not parse chemical formula {formula!r}; treating it as containing no elements ({exc}).")
        return []


def get_elements_per_row(df: pd.DataFrame, formula_column: str = "chemical formula") -> List[List[str]]:
    """Return one element list per row of `df`, in row order."""
    if formula_column not in df.columns:
        raise ValueError(f"Missing column: {formula_column}")

    formulas = df[formula_column].tolist()
    elements_per_row = [get_elements(formula) for formula in formulas]

    n_unparsed = sum(
        1 for formula, elements in zip(formulas, elements_per_row) if not elements and not pd.isna(formula)
    )
    if n_unparsed:
        warnings.warn(
            f"{n_unparsed} of {len(formulas)} formulas in '{formula_column}' could not be parsed and will be "
            f"excluded from any element/period/group-based filtering or splitting."
        )

    return elements_per_row


def formula_contains_elements(
    df: pd.DataFrame,
    elements: Iterable[str],
    formula_column: str = "chemical formula",
) -> np.ndarray:
    """Boolean mask over `df` rows whose formula contains any of `elements`."""
    wanted = set(elements)
    return np.array(
        [bool(wanted.intersection(row)) for row in get_elements_per_row(df, formula_column=formula_column)],
        dtype=bool,
    )


# --- Periodic table ---


def get_group_period_maps(
    pt: pd.DataFrame,
    element_col: str = "symbol",
    period_col: str = "period",
    group_block_col: str = "group_block",
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Derive element->group and element->period maps, as the OOD split families need them.

    Args:
        pt: Periodic table DataFrame (see loaddata.tabular_access.load_periodic_table).
        element_col: Column holding element symbols.
        period_col: Column holding period numbers.
        group_block_col: Column spelled like "group 1, s-block".

    Returns:
        (element_to_group, element_to_period). Elements with no group number, such as the f-block, are absent from the
        first map rather than present with a placeholder.

    Raises:
        ValueError: If any of the three columns is missing.
    """
    for column in (element_col, period_col, group_block_col):
        if column not in pt.columns:
            raise ValueError(f"Missing column '{column}' in periodic table file. Found: {list(pt.columns)}")

    def group_number(group_block):
        match = re.search(r"group\s*(\d+)", str(group_block), flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    symbols = pt[element_col].astype(str)
    element_to_period = dict(zip(symbols, pt[period_col].astype(int)))
    groups = (group_number(group_block) for group_block in pt[group_block_col])
    element_to_group = {symbol: group for symbol, group in zip(symbols, groups) if group is not None}
    return element_to_group, element_to_period


# --- Stoichiometry ---


def get_stoich_array(
    x: Union[pd.DataFrame, str],
    pt: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> pd.DataFrame:
    """Build the per-compound element-amount array from chemical formulas.

    Args:
        x: DataFrame with a formula column, or a single formula string.
        pt: Periodic table DataFrame (see loaddata.tabular_access.load_periodic_table), used for element symbols.
        formula_column: Name of the formula column when `x` is a DataFrame.

    Returns:
        Element amounts, one row per compound and one column per element symbol. Amounts stay float because pymatgen
        reports fractional stoichiometry for formulas like "Fe0.5Ni0.5"; casting to int would floor those to zero.
    """
    if isinstance(x, pd.DataFrame):
        formulas, index = x[formula_column], x.index
    else:
        formulas = pd.Series(x)
        index = formulas.index

    # Column order is longest symbol first. Nothing depends on it now that pymatgen does the parsing, but it fixes the
    # summation order of every downstream weighted mean, so changing it moves results in the last float digit.
    symbols = pt["symbol"].astype(str)
    symbols = symbols.reindex(symbols.str.len().sort_values(ascending=False).index)
    stoich_array = pd.DataFrame(np.zeros((len(formulas), len(symbols))), index=index, columns=symbols)

    for idx, formula in formulas.items():
        if pd.isna(formula):
            continue
        try:
            amounts = Composition(str(formula)).get_el_amt_dict()
        except Exception as exc:
            warnings.warn(f"Could not parse chemical formula {formula!r}; dropping the row ({exc}).")
            continue
        unknown = [element for element in amounts if element not in stoich_array.columns]
        if unknown:
            warnings.warn(
                f"Formula {formula!r} contains element(s) {unknown} missing from the periodic table file; they are "
                f"dropped from its stoichiometry."
            )
        for element, amount in amounts.items():
            if element in stoich_array.columns:
                stoich_array.at[idx, element] = amount

    return stoich_array


# --- Atomic fractions ---


def get_atomic_fraction(compound: pd.Series) -> pd.Series:
    """Atomic fractions of the elements present in one stoichiometry row, indexed by element symbol.

    An empty result means the formula produced no usable stoichiometry — unparseable, or every one of its elements
    missing from the periodic table file.
    """
    present = compound[compound != 0]
    if present.empty or present.sum() == 0:
        return pd.Series(dtype=float)
    return present / present.sum()


def get_atomic_fraction_array(stoich_array: pd.DataFrame) -> pd.DataFrame:
    """The atomic-fraction counterpart of `stoich_array`: same shape, index and columns, each row summing to 1.

    Elements absent from a compound come back NaN rather than 0, which is what separates this from a plain row-wise
    divide by the row sum.
    """
    rows = [get_atomic_fraction(compound) for _, compound in stoich_array.iterrows()]
    if not rows:
        return pd.DataFrame(columns=stoich_array.columns)
    # index= is explicit because a row with no usable stoichiometry yields an unnamed Series, which would otherwise be
    # relabelled and silently misalign the result against `stoich_array`.
    return pd.DataFrame(rows, index=stoich_array.index).reindex(columns=stoich_array.columns)


# --- Descriptor primitives ---


def _per_composition(stoich_array: pd.DataFrame, reduce: Callable[[pd.Series], float]) -> pd.Series:
    """Apply `reduce` to each composition's atomic fractions, mapping an empty composition to NaN."""
    values = pd.Series(index=stoich_array.index, dtype=float)
    for idx, compound in stoich_array.iterrows():
        fraction = get_atomic_fraction(compound)
        values.loc[idx] = np.nan if fraction.empty else reduce(fraction)
    return values


def get_weighted_property(values: pd.Series, stoich_array: pd.DataFrame) -> pd.Series:
    """Atomic-fraction-weighted mean of one element property, per composition.

    This is the primitive behind most element-weighted descriptors: pick a column of the periodic table and this
    averages it over each composition.

    Args:
        values: One element property, indexed by element symbol.
        stoich_array: Per-compound element amounts, as returned by get_stoich_array.

    Returns:
        One value per row of `stoich_array`, sharing its index.
    """
    return _per_composition(stoich_array, lambda fraction: float(np.dot(fraction, values.loc[fraction.index])))


def get_mixing_entropy(stoich_array: pd.DataFrame) -> pd.Series:
    """Ideal mixing entropy, -sum(f ln f) over each composition's atomic fractions."""
    return _per_composition(stoich_array, lambda fraction: -np.dot(fraction, np.log(fraction)))


def get_compound_radix(x: Union[pd.DataFrame, str], formula_column: str = "chemical formula") -> pd.Series:
    """Number of distinct elements per formula, NaN where the formula could not be parsed."""
    formulas = x[formula_column] if isinstance(x, pd.DataFrame) else pd.Series(x)
    return formulas.apply(lambda formula: len(get_elements(formula)) or np.nan)
