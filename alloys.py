# Adapted from https://github.com/rich970/ML-alloy-design/blob/master/alloys.py with modifications.

import os
from itertools import combinations

import numpy as np
import pandas as pd

def _flatten(x):
    """Flatten nested dicts that only contain a single 'value' field."""
    if isinstance(x, dict) and "value" in x and len(x) == 1:
        return x["value"]
    return x

def _sorted_elements(periodic_table):
    """Return element symbols sorted by descending length."""
    symbols = periodic_table["symbol"].astype(str)
    order = symbols.str.len().sort_values(ascending=False).index
    return symbols.reindex(order)

def _atomic_fraction(compound: pd.Series):
    """Return atomic fractions and element labels for a stoichiometry row."""
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

def _element_occurrence(df, periodic_table, formula_col, verbose=False):
    """Count how many compounds each element appears in for the given formula column."""
    formulas = df[formula_col].copy()
    symbols = _sorted_elements(periodic_table)

    # Calculate the occurrence of each element
    n_el_rows = []
    for el in symbols:
        regex_list = formulas.str.extractall(pat=r"(?P<element>{0})(?P<digit>\d*)".format(el))
        # drop the multi-index that extractall creates
        regex_list = regex_list.droplevel(level=1).copy()
        count = len(regex_list)
        n_el_rows.append({"element": el, "count": count})
        if verbose is True:
            print("Number of compounds containing {0} is {1}".format(el, count))
        # Remove the elements we have just found from the formulas list
        formulas[regex_list.index] = formulas[regex_list.index].replace(
            to_replace=regex_list.element + regex_list.digit, value=None, regex=True
        )
    return pd.DataFrame(n_el_rows, columns=["element", "count"])

def importNovamag(root_dir):
    """
    Load all Novamag JSON files into a flat DataFrame of chemistry, crystal, and magnetic properties.
    """

    rows = []
    failed_files = []

    for dirName, subdirList, fileList in os.walk(root_dir):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname.endswith(".json"):
                filepath = os.path.join(dirName, fname)
                try:
                    df = pd.read_json(filepath, encoding="Latin")
                    row = {**df.properties.chemistry, **df.properties.crystal, **df.properties.magnetics}
                    rows.append(row)
                except ValueError:
                    print("Import failed", filepath)
                    failed_files.append(filepath)
    X = pd.DataFrame(rows)
    X = X.apply(lambda col: col.map(_flatten))
    return X


def import_periodic_table(root_dir):
    """Import the periodic table spreadsheet and index by symbol."""
    pt = pd.read_excel(root_dir)
    pt.index = pt["symbol"]
    return pt


def import_miedema_weight(root_dir):
    """Import Miedema model enthalpies spreadsheet and symmetrise it."""
    mm = pd.read_excel(root_dir, header=1, index_col=73, usecols=range(0, 74), nrows=73).fillna(0)
    mm_T = mm.transpose().fillna(0)
    mm = mm + mm_T
    return mm

def get_K_mag(X):
    # Extract the magnetocrystalline anisotropy constant K1
    K = X["magnetocrystalline anisotropy constants"].copy()
    for i in range(len(K)):
        try:
            # Turns out we have no non-zero K2 values, so the magnitude is just the K1 value.
            X["magnetocrystalline anisotropy constants"].iloc[i] = K.iloc[i][0]
        except:
            TypeError  # to deal with 'nan' values which are vectors
    return X


def get_element_occurrence_novamag(x, pt, verbose=False):
    """Novamag: use 'chemical formula' column."""
    return _element_occurrence(x, pt, "chemical formula", verbose=verbose)


def get_element_occurrence_mp(x, pt, verbose=False):
    """Materials Project: use 'composition' column."""
    return _element_occurrence(x, pt, "composition", verbose=verbose)


def get_stoich_array(x, pt):
    """Create stoichiometry array (element counts) from chemical formulas."""
    if isinstance(x, pd.DataFrame):
        formulas = x["chemical formula"].copy()  # if user passes whole of Novamag
    else:
        formulas = pd.Series(x)  # if user passes a single chemical formula string
        print(formulas)

    # Get a list of element symbols and sort in order of descending length
    # Need longest first as elements like S will be found within Si, As etc.
    symbols = _sorted_elements(pt)

    # Will encode chemical formula data in a large array
    stoich_array = pd.DataFrame(
        np.zeros([len(formulas), len(symbols)]),
        index=formulas.index,
        columns=symbols.copy(),
    )

    for el in symbols:
        # Ensure each element in the chemical formula has an explicit digit (e.g., Fe1Co1 instead of FeCo)
        regex_list = formulas.str.extractall(pat=r"(?P<element>{0})(?P<digit>\d*)".format(el))
        # drop the multi-indexing that 'extractall' creates
        regex_list = regex_list.droplevel(level=1).copy()
        count = len(regex_list)
        if count > 0:
            # add the number of atoms to the correct el col in the stoich array
            digits = regex_list["digit"].replace("", "1").astype(float)
            stoich_array.loc[regex_list.index, el] = digits

        # Remove the elements we have just found from the formulas list
        formulas[regex_list.index] = formulas[regex_list.index].replace(
            # to_replace=regex_list.element + regex_list.digit, value=None, regex=True
            to_replace=regex_list.element + regex_list.digit,
            regex=True,
        )

    # Need to rewrite the string numbers as integers in our stoichiometry array
    stoich_array = stoich_array.fillna(0).astype(int)
    # Restore index to formula strings for readability/access
    stoich_array.index = formulas.values

    return stoich_array


def get_Electronegw(pt, stoich_array):
    """Calculate element-weighted electronegativity."""
    electronegw = pd.Series(np.zeros(len(stoich_array)))
    en_list = pt["electronegativity"].str.extract(pat=r"(?P<digit>\d*\.\d+)").astype(float)
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        at_fraction, labels = _atomic_fraction(compound)
        electronegw.iloc[i] = np.dot(at_fraction, en_list.loc[labels])
    return electronegw


def get_Zw(pt, stoich_array):
    """Calculate element-weighted atomic weight."""
    zw = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        at_fraction, labels = _atomic_fraction(compound)
        zw.iloc[i] = np.dot(at_fraction, pt.loc[labels]["atomic_weight"])
    return zw


def get_Groupw(pt, stoich_array):
    """Calculate element-weighted group number."""
    groupw = pd.Series(np.zeros(len(stoich_array)))
    group_block = pt["group_block"].str.extract(pat=r"(\d+)").dropna()
    group_block = group_block.astype(int)
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        at_fraction, labels = _atomic_fraction(compound)
        groupw.iloc[i] = np.dot(at_fraction, group_block.loc[labels])
    return groupw


def get_Periodw(pt, stoich_array):
    """Calculate element-weighted period number."""
    periodw = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        at_fraction, labels = _atomic_fraction(compound)
        periodw.iloc[i] = np.dot(at_fraction, pt.loc[labels]["period"])
    return periodw


def get_MeltingTw(pt, stoich_array):
    """Calculate element-weighted melting temperature."""
    meltingTw = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        at_fraction, labels = _atomic_fraction(compound)
        meltingTw.iloc[i] = np.dot(at_fraction, pt.loc[labels]["melting_point"])
    return meltingTw


def get_Valencew(pt, stoich_array):
    """Calculate element-weighted valence electron number."""
    valencew = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        at_fraction, labels = _atomic_fraction(compound)
        valencew.iloc[i] = np.dot(at_fraction, pt.loc[labels]["valence"])
    return valencew


def get_Miedemaw(mm, stoich_array):
    """Calculate weighted Miedema enthalpy of formation (pairwise sum over elements)."""
    miedemaw = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        at_fraction, _ = _atomic_fraction(compound)
        # Calculate Miedema enthalpy by summing all binary combinations
        comb = combinations(at_fraction.index, 2)
        for el in list(comb):
            try:
                miedemaw.iloc[i] += 4 * at_fraction[el[0]] * at_fraction[el[1]] * mm.loc[el[0]][el[1]]
            except KeyError:
                miedemaw.iloc[i] = np.nan
    return miedemaw


def get_StoicEntw(stoich_array):
    """Calculate stoichiometric (mixing) entropy."""
    stoicentw = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        at_fraction, _ = _atomic_fraction(compound)
        stoicentw.iloc[i] = -np.dot(at_fraction, np.log(at_fraction))
    return stoicentw


def get_AtomicFrac(stoich_array):
    """Calculate atomic fractions of each element for every compound."""
    rows = []
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        at_fraction, _ = _atomic_fraction(compound)
        rows.append(at_fraction)
    if not rows:
        return pd.DataFrame(columns=stoich_array.columns)
    return pd.DataFrame(rows).reindex(columns=stoich_array.columns)


def get_CompoundRadix(pt, X):
    """Calculate compound radix (binary, ternary, quaternary etc.)"""
    if type(X) == pd.DataFrame:
        formulas = X["chemical formula"].copy()  # if user passes whole of Novamag
    else:
        formulas = pd.Series(X)  # if user passes a single chemical formula string
        print(formulas)
    # Make a new column for the compound index i.e. 2 = binary
    compoundradix = pd.Series(np.zeros(len(formulas)))
    # Get a list of symbols and sort in order of descending string length
    symbols = _sorted_elements(pt)

    for el in symbols:
        regex_list = formulas.str.extractall(pat=r"(?P<element>{0})(?P<digit>\d*)".format(el))
        # drop the multi-indexing that 'extractall' creates
        regex_list = regex_list.droplevel(level=1).copy()
        # Remove the elements we have just found from the formulas list
        formulas[regex_list.index] = formulas[regex_list.index].replace(
            # to_replace=regex_list.element + regex_list.digit, value=None, regex=True
            to_replace=regex_list.element + regex_list.digit,
            regex=True,
        )

        # Use the regex indices to update the compound radix column
        compoundradix[regex_list.index] += 1
    return compoundradix
