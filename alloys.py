# Adapted from https://github.com/rich970/ML-alloy-design/blob/master/alloys.py with modifications.

import os
from itertools import combinations

import numpy as np
import pandas as pd


def importNovamag(root_dir):
    """
    Auto import all json files in the Novamag database, creating a pandas dataframe object.
    Parameters
    ----------
    root_dir : string
        location of root directory where all the Novamag data is stored.

    Returns
    -------
    X : dataframe
        Output dataframe of the Novamag database.
    """

    X = pd.DataFrame()
    ls = dict()
    failedfiles = []

    for dirName, subdirList, fileList in os.walk(root_dir):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname.endswith(".json"):
                filepath = dirName + "/" + fname
                try:
                    df = pd.read_json(filepath, encoding="Latin")
                    ls = {**df.properties.chemistry, **df.properties.crystal, **df.properties.magnetics}
                    X = X._append(pd.DataFrame.from_dict(ls))
                except ValueError:
                    print("Import failed")
                    failedfiles.append(filepath)
    X.index = range(len(X))  
    return X


def import_periodic_table(root_dir):
    # Import Periodic Table
    pt = pd.read_excel(root_dir)
    pt.index = pt["symbol"]
    return pt


def import_miedema_weight(root_dir):
    # Import Miedema model enthalpies
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


def get_element_occurance(x, pt, verbose=False):
    """
    Count the number of distinct compounds each element appears in across a dataset.

    Args:
        x (pd.DataFrame): DataFrame containing a "chemical formula" column with formula strings.
        pt (pd.DataFrame): DataFrame containing an element "symbol" column used to build regex matches.
        verbose (bool, optional): If True, print the count per element during processing. Defaults to False.

    Returns:
        pd.DataFrame: Two-column DataFrame with element symbols and the number of compounds they occur in.
    """
    # Get the number of compounds each element appears in
    formulas = x["chemical formula"].copy()
    # Get a list of element symbols, sort in order of descending string length
    symbols = pt["symbol"].copy()
    s = symbols.str.len().sort_values(ascending=False).index
    symbols = symbols.reindex(s)

    # Calculate the occurance of each element
    n_el = pd.DataFrame(columns=["element", "count"])
    for el in symbols:
        regex_list = formulas.str.extractall(pat=r"(?P<element>{0})(?P<digit>\d*)".format(el))
        # drop the multi-index that extractall creates
        regex_list = regex_list.droplevel(level=1).copy()
        count = len(regex_list)
        n_el = n_el.append({"element": el, "count": count}, ignore_index=True)
        if verbose is True:
            print("Number of compounds containing {0} is {1}".format(el, count))
        # Remove the elements we have just found from the formulas list
        formulas[regex_list.index] = formulas[regex_list.index].replace(
            to_replace=regex_list.element + regex_list.digit, value=None, regex=True
        )
    return n_el


def get_element_occurance_mp(x, pt, verbose=False):
    # Get the number of compounds each element appears in MP dataset
    formulas = x["composition"].copy()
    # Get a list of element symbols, sort in order of descending string length
    symbols = pt["symbol"].copy()
    s = symbols.str.len().sort_values(ascending=False).index
    symbols = symbols.reindex(s)

    # Calculate the occurance of each element
    n_el = pd.DataFrame(columns=["element", "count"])
    for el in symbols:
        regex_list = formulas.str.extractall(pat=r"(?P<element>{0})(?P<digit>\d*)".format(el))
        # drop the multi-index that extractall creates
        regex_list = regex_list.droplevel(level=1).copy()
        count = len(regex_list)
        n_el = n_el.append({"element": el, "count": count}, ignore_index=True)
        if verbose is True:
            print("Number of compounds containing {0} is {1}".format(el, count))
        # Remove the elements we have just found from the formulas list
        formulas[regex_list.index] = formulas[regex_list.index].replace(
            to_replace=regex_list.element + regex_list.digit, value=None, regex=True
        )
    return n_el


def get_stoich_array(x, pt):
    # Create stoichiometry array from chemical formulas
    if type(x) == pd.DataFrame:
        formulas = x["chemical formula"].copy()  # if user passes whole of Novamag
    else:
        formulas = pd.Series(x)  # if user passes a single chemical formula string
        print(formulas)
    # Get a list of element symbols and sort in order of descending length
    # Need longest first as elements like S will be found within Si, As etc.
    symbols = pt["symbol"].copy()
    s = symbols.str.len().sort_values(ascending=False).index
    symbols = symbols.reindex(s)
    # Will encode chemical formula data in a large array
    stoich_array = pd.DataFrame(np.zeros([len(formulas), len(symbols)]))
    stoich_array.columns = symbols.copy()
    stoich_array.index = formulas.copy()

    for el in symbols:
        # Ensure each element in the chemical formula has an explicit digit (e.g., Fe1Co1 instead of FeCo)
        regex_list = formulas.str.extractall(pat=r"(?P<element>{0})(?P<digit>\d*)".format(el))
        # drop the multi-indexing that 'extractall' creates
        regex_list = regex_list.droplevel(level=1).copy()
        count = len(regex_list)
        if count > 0:
            # add the number of atoms to the correct el col in the stoich array
            stoich_array[el][regex_list.index] = regex_list.digit

        # Remove the elements we have just found from the formulas list
        formulas[regex_list.index] = formulas[regex_list.index].replace(
            # to_replace=regex_list.element + regex_list.digit, value=None, regex=True
            to_replace=regex_list.element + regex_list.digit,
            regex=True,
        )

    # Need to rewrite the string numbers as integers in our stoichiometry array
    stoich_array = stoich_array.fillna(0)
    for col in stoich_array.columns:
        stoich_array[col] = stoich_array[col].astype(int)

    return stoich_array


def get_Electronegw(pt, stoich_array):
    # Calculate weighted electronegativity
    electronegw = pd.Series(np.zeros(len(stoich_array)))
    en_list = pt["electronegativity"].str.extract(pat=r"(?P<digit>\d*\.\d+)").astype(float)
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()  # nonzero elements columns
        at_fraction = compound.iloc[cols] / sum(compound.iloc[cols])
        electronegw.iloc[i] = np.dot(at_fraction, en_list.loc[compound.index[cols]])
    return electronegw


def get_Zw(pt, stoich_array):
    # Calculate weighted atomic number
    zw = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()  # nonzero elements columns
        at_fraction = compound.iloc[cols] / sum(compound.iloc[cols])
        zw.iloc[i] = np.dot(at_fraction, pt.loc[compound.index[cols]]["atomic_weight"])
    return zw


def get_Groupw(pt, stoich_array):
    # Calculate weighted group number
    groupw = pd.Series(np.zeros(len(stoich_array)))
    group_block = pt["group_block"].str.extract(pat=r"(\d+)").dropna()
    group_block = group_block.astype(int)
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()  # nonzero elements columns
        at_fraction = compound.iloc[cols] / sum(compound.iloc[cols])
        groupw.iloc[i] = np.dot(at_fraction, group_block.loc[compound.index[cols]])
    return groupw


def get_Periodw(pt, stoich_array):
    # Calculate weighted period number
    periodw = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()  # nonzero elements columns
        at_fraction = compound.iloc[cols] / sum(compound.iloc[cols])
        periodw.iloc[i] = np.dot(at_fraction, pt.loc[compound.index[cols]]["period"])
    return periodw


def get_MeltingTw(pt, stoich_array):
    # Calculate weighted melting temperature
    meltingTw = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()  # nonzero elements columns
        at_fraction = compound.iloc[cols] / sum(compound.iloc[cols])
        meltingTw.iloc[i] = np.dot(at_fraction, pt.loc[compound.index[cols]]["melting_point"])
    return meltingTw


def get_Valencew(pt, stoich_array):
    # Calculate weighted valence electron number
    valencew = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()  # nonzero elements columns
        at_fraction = compound.iloc[cols] / sum(compound.iloc[cols])
        valencew.iloc[i] = np.dot(at_fraction, pt.loc[compound.index[cols]]["valence"])
    return valencew


def get_Miedemaw(mm, stoich_array):
    # Calculate weighted Miedema enthalpy of formation
    miedemaw = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()
        at_fraction = compound.iloc[cols] / sum(compound.iloc[cols])
        # Calculate Miedema enthalpy by summing all binary combinations
        comb = combinations(at_fraction.index, 2)
        for el in list(comb):
            try:
                miedemaw.iloc[i] += 4 * at_fraction[el[0]] * at_fraction[el[1]] * mm.loc[el[0]][el[1]]
            except KeyError:
                miedemaw.iloc[i] = np.nan
    return miedemaw


def get_StoicEntw(stoich_array):
    # Calculate stoichiometric entropy
    stoicentw = pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()  # nonzero elements columns
        at_fraction = compound.iloc[cols] / sum(compound.iloc[cols])
        stoicentw.iloc[i] = -np.dot(at_fraction, np.log(at_fraction))
    return stoicentw


def get_AtomicFrac(stoich_array):
    # Calculate atomic fraction of each element in compound
    at_fraction = pd.DataFrame()
    for i in range(len(stoich_array)):
        compound = stoich_array.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()
        at_fraction = at_fraction._append(compound.iloc[cols] / sum(compound.iloc[cols]))
    return at_fraction


def get_CompoundRadix(pt, X):
    # Calculate compound radix (binary, ternary, quaternary etc.)
    if type(X) == pd.DataFrame:
        formulas = X["chemical formula"].copy()  # if user passes whole of Novamag
    else:
        formulas = pd.Series(X)  # if user passes a single chemical formula string
        print(formulas)
    # Make a new column for the compound index i.e. 2 = binary
    compoundradix = pd.Series(np.zeros(len(formulas)))
    # Get a list of symbols and sort in order of descending string length
    symbols = pt["symbol"].copy()
    s = symbols.str.len().sort_values(ascending=False).index
    symbols = symbols.reindex(s)

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
