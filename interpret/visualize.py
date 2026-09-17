"""Dataset distribution plots: Ms histograms/violin plots, compound-radix summaries."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from prepdata.composition import formula_contains_elements, get_compound_radix


def plot_ms_distribution_by_tm(data, save_path=None):
    """Plot Ms histograms for the full dataset and Fe/Co/Cr/Mn-containing subsets.

    Args:
        data: DataFrame with 'chemical formula' and 'saturation magnetization' columns.
        save_path: If set, save the figure here instead of showing it.
    """
    data_Fe = data[formula_contains_elements(data, ["Fe"])]
    data_Co = data[formula_contains_elements(data, ["Co"])]
    data_Cr = data[formula_contains_elements(data, ["Cr"])]
    data_Mn = data[formula_contains_elements(data, ["Mn"])]

    plt.figure(figsize=(8, 6))
    bins = np.arange(0.0, 2.6, 0.2)

    sns.histplot(x=data["saturation magnetization"], kde=True, label="all")
    sns.histplot(x=data_Fe["saturation magnetization"], kde=True, bins=bins, label="Fe")
    sns.histplot(x=data_Co["saturation magnetization"], kde=True, bins=bins, label="Co")
    sns.histplot(x=data_Cr["saturation magnetization"], kde=True, bins=bins, label="Cr")
    sns.histplot(x=data_Mn["saturation magnetization"], kde=True, bins=bins, label="Mn")
    plt.xlabel("saturation magnetization")
    plt.ylabel("Count")
    plt.legend(title="Element", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_violin_ms_by_tm(data, title: str = "Violin Plot", save_path=None):
    """Plot a violin plot of saturation magnetization grouped by TM elements.

    Args:
        data: DataFrame with 'chemical formula' and 'saturation magnetization' columns.
        save_path: If set, save the figure here instead of showing it.
    """
    # Subsets grouped by element
    data_Fe = data[formula_contains_elements(data, ["Fe"])]
    data_Co = data[formula_contains_elements(data, ["Co"])]
    data_Cr = data[formula_contains_elements(data, ["Cr"])]
    data_Mn = data[formula_contains_elements(data, ["Mn"])]

    # Combine into one DataFrame for seaborn violinplot
    plot_data = pd.concat([
        data['saturation magnetization'].rename('all'),
        data_Fe['saturation magnetization'].rename('Fe'),
        data_Co['saturation magnetization'].rename('Co'),
        data_Cr['saturation magnetization'].rename('Cr'),
        data_Mn['saturation magnetization'].rename('Mn')
    ], axis=1)

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=plot_data, inner="quartile")
    plt.xlabel("Element", fontsize=16)
    plt.ylabel("Saturation Magnetization (T)", fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def count_compounds_by_radix(data) -> pd.Series:
    """Count how many compounds have each number of distinct elements.

    Returns:
        Counts indexed by radix, ascending.
    """
    return get_compound_radix(data).value_counts().sort_index()
