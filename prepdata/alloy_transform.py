"""Chemical-formula parsing and element-weighted alloy feature transforms.

Element-weighted feature calculations (get_*w below) are adapted from
https://github.com/rich970/ML-alloy-design/blob/master/alloys.py with modifications.
"""

import re
import warnings
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

# ====== Formula parsing and periodic table mapping ======


def parse_elements_from_formula(formula: str) -> List[str]:
    """Extract unique element symbols from a chemical formula using pymatgen.

    Uses the same parser as get_stoich_array so element identification is consistent across the pipeline.

    A row whose elements come back empty is invisible to any element/period/ group-based OOD split (it can never be
    selected as train or test for a given target), so a genuine parse failure is surfaced via warnings.warn rather than
    swallowed. A missing formula (None/NaN) is not a failure and stays silent.

    Example:
        Nd2Fe14B -> ["Nd", "Fe", "B"]
    """
    if pd.isna(formula):
        return []
    try:
        comp = Composition(str(formula))
        return [str(el) for el in comp.elements]
    except Exception as exc:
        warnings.warn(
            f"Could not parse chemical formula {formula!r}; treating it as "
            f"containing no elements, so it will be excluded from any "
            f"element/period/group-based OOD split ({exc})."
        )
        return []


def extract_elements_series(df_raw: pd.DataFrame, formula_column: str = "chemical formula") -> List[List[str]]:
    """Return the parsed element list per row, aligned with `df_raw`'s row order."""
    if formula_column not in df_raw.columns:
        raise ValueError(f"Missing column: {formula_column}")

    formulas = df_raw[formula_column].tolist()
    elements_per_row = [parse_elements_from_formula(x) for x in formulas]

    n_unparsed = sum(
        1
        for formula, elements in zip(formulas, elements_per_row)
        if not elements and not pd.isna(formula)
    )
    if n_unparsed:
        print(
            f"[WARN] {n_unparsed} of {len(formulas)} formulas in "
            f"'{formula_column}' could not be parsed and will be excluded "
            f"from any element/period/group-based filtering or splitting."
        )

    return elements_per_row


def elements_mask(elements_per_row: Sequence[Sequence[str]], elements: Iterable[str]) -> np.ndarray:
    """Boolean mask: True where a row's parsed elements intersect `elements`.

    Built on the same pymatgen-based parsing as parse_elements_from_formula, so element-based filtering/grouping stays
    consistent with the OOD element splits instead of relying on ad hoc formula substring matching.
    """
    target = set(elements)
    return np.array([bool(target.intersection(els)) for els in elements_per_row], dtype=bool)


def formula_contains_elements(
    df: pd.DataFrame,
    elements: Iterable[str],
    formula_column: str = "chemical formula",
) -> np.ndarray:
    """Boolean mask over `df` rows whose formula contains any of `elements`."""
    elements_per_row = extract_elements_series(df, formula_column=formula_column)
    return elements_mask(elements_per_row, elements)


def load_periodic_table_map(
    pt_path: str,
    element_col: str = "symbol",
    period_col: str = "period",
    group_block_col: str = "group_block",
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Load the periodic table spreadsheet into element->group and element->period maps.

    Expects `symbol`, `period`, and `group_block` columns (e.g. "group 1, s-block").

    Returns:
        (element_to_group, element_to_period).
    """
    pt = pd.read_excel(pt_path)

    for c in [element_col, period_col, group_block_col]:
        if c not in pt.columns:
            raise ValueError(f"Missing column '{c}' in periodic table file. Found: {list(pt.columns)}")

    pt = pt[[element_col, period_col, group_block_col]].copy()

    pt[element_col] = pt[element_col].astype(str)
    pt[period_col] = pt[period_col].astype(int)

    def _extract_group(gb):
        if pd.isna(gb):
            return None

        m = re.search(r"group\s*(\d+)", str(gb), flags=re.IGNORECASE)

        return int(m.group(1)) if m else None

    pt["group"] = pt[group_block_col].map(_extract_group)

    element_to_period = dict(zip(pt[element_col], pt[period_col]))

    element_to_group = {
        sym: int(g)
        for sym, g in zip(pt[element_col], pt["group"])
        if pd.notnull(g)
    }

    return element_to_group, element_to_period


# ====== Element-weighted alloy feature calculations ======


def _sorted_elements(periodic_table):
    """Return element symbols sorted by descending length."""
    symbols = periodic_table["symbol"].astype(str)
    order = symbols.str.len().sort_values(ascending=False).index
    return symbols.reindex(order)


def _atomic_fraction(compound: pd.Series):
    """Return atomic fractions and element labels for a stoichiometry row.

    An empty result means the row's formula produced no usable stoichiometry (unparseable, or all its elements are
    absent from the periodic table file). Callers must map that to NaN rather than 0, so build_features' dropna removes
    the row instead of training on a fabricated all-zero feature vector — see _weighted_mean.
    """
    # mask of elements that appear in the compound
    mask = compound != 0
    subset = compound[mask]

    if subset.empty:
        # no elements → return empty atomic fraction
        empty = pd.Series(dtype=float)
        return empty, empty.index

    total = subset.sum()
    if total == 0:
        empty = pd.Series(dtype=float)
        return empty, empty.index

    af = subset / total
    return af, af.index


def _weighted_mean(at_fraction: pd.Series, values: pd.Series) -> float:
    """Atomic-fraction-weighted mean of `values`, or NaN for an empty compound."""
    if at_fraction.empty:
        # NaN, not 0.0: build_features' dropna then removes the row instead of training on an all-zero feature vector.
        return np.nan
    return float(np.dot(at_fraction, values.loc[at_fraction.index]))


def get_stoich_array(x, pt, formula_column: str = "chemical formula"):
    """Create stoichiometry array (element counts) from chemical formulas.

    Args:
        x: DataFrame with a formula column, or a single formula string.
        pt: Periodic table DataFrame (see loaddata.alloy_access.import_periodic_table), used for element symbols.
        formula_column: Name of the formula column when `x` is a DataFrame.

    Returns:
        DataFrame of per-compound element amounts (float), columns = element
        symbols. Amounts stay float because pymatgen reports fractional
        stoichiometry for formulas like "Fe0.5Ni0.5"; casting to int would
        floor those to 0, leaving an all-zero row whose features then come out
        as a plausible-looking 0.0 rather than NaN.
    """
    if isinstance(x, pd.DataFrame):
        formulas = x[formula_column].copy()  # if user passes a whole dataset
        index = x.index
    else:
        formulas = pd.Series(x)  # if user passes a single chemical formula string
        index = formulas.index

    # Get a list of element symbols and sort in order of descending length
    # Need longest first as elements like S will be found within Si, As etc.
    symbols = _sorted_elements(pt)

    # Will encode chemical formula data in a large array
    stoich_array = pd.DataFrame(np.zeros([len(formulas), len(symbols)]), index=index, columns=symbols.copy())

    for idx, f in formulas.items():
        if pd.isna(f):
            continue
        try:
            el_dict = Composition(str(f)).get_el_amt_dict()
        except Exception as exc:
            # Same contract as parse_elements_from_formula: warn and leave the row empty. Its features then come out NaN
            # (see _weighted_mean), so build_features drops it instead of crashing the whole run.
            warnings.warn(f"Could not parse chemical formula {f!r}; dropping the row ({exc}).")
            continue
        unknown = [el for el in el_dict if el not in stoich_array.columns]
        if unknown:
            warnings.warn(
                f"Formula {f!r} contains element(s) {unknown} missing from the "
                f"periodic table file; they are dropped from its stoichiometry."
            )
        for el, amt in el_dict.items():
            if el in stoich_array.columns:
                stoich_array.at[idx, el] = amt

    return stoich_array


def get_electronegw(pt, stoich_array):
    """Calculate element-weighted electronegativity."""
    electronegw = pd.Series(index=stoich_array.index, dtype=float)
    en_list = pt["electronegativity"].str.extract(pat=r"(?P<digit>\d*\.\d+)").astype(float)["digit"]
    for i, compound in stoich_array.iterrows():
        at_fraction, _ = _atomic_fraction(compound)
        electronegw.loc[i] = _weighted_mean(at_fraction, en_list)
    return electronegw


def get_zw(pt, stoich_array):
    """Calculate element-weighted atomic weight."""
    zw = pd.Series(index=stoich_array.index, dtype=float)
    for i, compound in stoich_array.iterrows():
        at_fraction, _ = _atomic_fraction(compound)
        zw.loc[i] = _weighted_mean(at_fraction, pt["atomic_weight"])
    return zw


def get_groupw(pt, stoich_array):
    """Calculate element-weighted group number."""
    group_block = pt["group_block"].str.extract(r"(\d+)")[0]
    group_block = group_block.astype(float)
    groupw = pd.Series(index=stoich_array.index, dtype=float)
    for i, compound in stoich_array.iterrows():
        at_fraction, labels = _atomic_fraction(compound)

        # Handle case where no elements are found
        if at_fraction.empty:
            groupw.loc[i] = np.nan
            continue

        # Filter to only valid elements with group numbers
        valid_labels = [
            el for el in labels
            if el in group_block.index and not pd.isna(group_block.loc[el])
        ]
        if not valid_labels:
            groupw.loc[i] = np.nan
            continue

        # Re-normalise atomic fractions to only valid elements
        af_sub = at_fraction.loc[valid_labels]
        af_sub = af_sub / af_sub.sum()

        groupw.loc[i] = np.dot(af_sub, group_block.loc[valid_labels])
    return groupw


def get_periodw(pt, stoich_array):
    """Calculate element-weighted period number."""
    periodw = pd.Series(index=stoich_array.index, dtype=float)

    for idx, compound in stoich_array.iterrows():
        af, _ = _atomic_fraction(compound)
        periodw.loc[idx] = _weighted_mean(af, pt["period"])
    return periodw


def get_melting_tw(pt, stoich_array):
    """Calculate element-weighted melting temperature."""
    meltingTw = pd.Series(index=stoich_array.index, dtype=float)
    for i, compound in stoich_array.iterrows():
        at_fraction, _ = _atomic_fraction(compound)
        meltingTw.loc[i] = _weighted_mean(at_fraction, pt["melting_point"])
    return meltingTw


def get_valencew(pt, stoich_array):
    """Calculate element-weighted valence electron number."""
    valencew = pd.Series(index=stoich_array.index, dtype=float)
    for i, compound in stoich_array.iterrows():
        at_fraction, _ = _atomic_fraction(compound)
        valencew.loc[i] = _weighted_mean(at_fraction, pt["valence"])
    return valencew


def get_miedemaw(mm, stoich_array):
    """Calculate weighted Miedema enthalpy of formation (pairwise sum over elements)."""
    miedemaw = pd.Series(index=stoich_array.index, dtype=float)
    for i, compound in stoich_array.iterrows():
        at_fraction, labels = _atomic_fraction(compound)

        if at_fraction.empty:
            miedemaw.loc[i] = np.nan
            continue

        # Calculate pairwise contributions
        H = 0
        valid = True
        for a, b in combinations(labels, 2):
            try:
                H += 4 * at_fraction[a] * at_fraction[b] * mm.loc[a, b]
            except KeyError:
                valid = False
                break
        miedemaw.loc[i] = H if valid else np.nan
    return miedemaw


def get_stoic_entw(stoich_array):
    """Calculate stoichiometric (mixing) entropy."""
    stoicentw = pd.Series(index=stoich_array.index, dtype=float)
    for i, compound in stoich_array.iterrows():
        at_fraction, _ = _atomic_fraction(compound)
        if at_fraction.empty:
            stoicentw.loc[i] = np.nan
            continue
        stoicentw.loc[i] = -np.dot(at_fraction, np.log(at_fraction))
    return stoicentw


def get_atomic_frac(stoich_array):
    """Calculate atomic fractions of each element for every compound."""
    rows = []
    for _, compound in stoich_array.iterrows():
        at_fraction, _ = _atomic_fraction(compound)
        rows.append(at_fraction)
    if not rows:
        return pd.DataFrame(columns=stoich_array.columns)
    return pd.DataFrame(rows).reindex(columns=stoich_array.columns)


def get_compound_radix(X, formula_column: str = "chemical formula"):
    """Calculate compound radix (number of distinct elements) for each formula."""
    if isinstance(X, pd.DataFrame):
        formulas = X[formula_column].copy()
    else:
        formulas = pd.Series(X)

    # compound radix = number of distinct elements in the formula. Routed through the guarded parser so an unparseable
    # formula yields NaN (and is dropped by build_features) rather than raising mid-run.
    def _radix(formula):
        n_elements = len(parse_elements_from_formula(formula))
        return n_elements if n_elements else np.nan

    return formulas.apply(_radix)
