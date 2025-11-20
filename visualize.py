import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import alloys

def plot_ms_distribution_by_tm(X):
    """
    Plot histograms of saturation magnetization grouped by TM elements, such as Fe/Co/Cr/Mn.

    X : DataFrame
        Must contain 'chemical formula' and 'saturation magnetization'.
    """
    X_Fe = X[X["chemical formula"].str.contains(pat="Fe")]
    X_Co = X[X["chemical formula"].str.contains(pat="Co")]
    X_Cr = X[X["chemical formula"].str.contains(pat="Cr")]
    X_Mn = X[X["chemical formula"].str.contains(pat="Mn")]

    plt.figure(figsize=(8, 6))
    bins = np.arange(0.0, 2.6, 0.2)

    sns.histplot(x=X["saturation magnetization"], kde=True, label="all")
    sns.histplot(x=X_Fe["saturation magnetization"], kde=True, bins=bins, label="Fe")
    sns.histplot(x=X_Co["saturation magnetization"], kde=True, bins=bins, label="Co")
    sns.histplot(x=X_Cr["saturation magnetization"], kde=True, bins=bins, label="Cr")
    sns.histplot(x=X_Mn["saturation magnetization"], kde=True, bins=bins, label="Mn")

    plt.xlabel("saturation magnetization")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_violin_ms_by_tm(X):
    """
    Plot a violin plot of saturation magnetization grouped by TM elements.
    """

    # Subsets grouped by element
    X_Fe = X[X['chemical formula'].str.contains(pat='Fe')]
    X_Co = X[X['chemical formula'].str.contains(pat='Co')]
    X_Cr = X[X['chemical formula'].str.contains(pat='Cr')]
    X_Mn = X[X['chemical formula'].str.contains(pat='Mn')]

    # Combine into one DataFrame for seaborn violinplot
    data = pd.concat([
        X['saturation magnetization'].rename('all'),
        X_Fe['saturation magnetization'].rename('Fe'),
        X_Co['saturation magnetization'].rename('Co'),
        X_Cr['saturation magnetization'].rename('Cr'),
        X_Mn['saturation magnetization'].rename('Mn')
    ], axis=1)

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, inner="quartile")
    plt.xlabel("Element", fontsize=16)
    plt.ylabel("Saturation Magnetization (T)", fontsize=16)
    plt.title("Novamag Violin Plot", fontsize=16)
    plt.tight_layout()
    plt.show()

def summarize_compound_radix(X, PT):
    """
    Print the number of compounds in the dataset.
    """

    # Compute compound radix (number of unique elements in formula)
    X['compoundradix'] = alloys.get_CompoundRadix(PT, X)

    total_compound_radix = X['compoundradix'].value_counts().sort_index()

    # Print results cleanly
    for radix, count in total_compound_radix.items():
        if radix == 2:
            name = "binary"
        elif radix == 3:
            name = "ternary"
        elif radix == 4:
            name = "quaternary"
        elif radix == 5:
            name = "quinary"
        elif radix == 6:
            name = "senary"
        elif radix == 7:
            name = "septenary"
        else:
            name = f"{radix}-component"

        print(f"We have {count} {name} compounds")
