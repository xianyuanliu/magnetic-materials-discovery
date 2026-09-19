"""Dataset distribution plots: how a target is distributed overall and within element-containing subsets.

Nothing here is about magnetism, or about any one property: the target column, the formula column and the elements to
split on are all arguments. Any composition dataset — an oxide formation energy grouped by rare earth, a band gap
grouped by anion — plots the same way. What the caller supplies is which column holds the number and which elements
are worth separating out.
"""

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from prepdata.composition import formula_contains_elements, get_compound_radix


def _subsets_by_element(
    data: pd.DataFrame,
    elements: Sequence[str],
    formula_column: str,
) -> Dict[str, pd.DataFrame]:
    """Split `data` into the rows whose formula contains each element.

    The subsets overlap: a formula containing both Fe and Co appears under both, which is the point — the question is
    what the presence of an element does to the target, not how the dataset partitions.

    Returns:
        {element: rows containing it}, in the order the elements were given.
    """
    return {
        element: data[formula_contains_elements(data, [element], formula_column=formula_column)]
        for element in elements
    }


def plot_target_distribution_by_element(
    data: pd.DataFrame,
    *,
    target_column: str,
    elements: Sequence[str],
    formula_column: str = "chemical formula",
    bins="auto",
    save_path=None,
) -> None:
    """Overlay the target's histogram for the whole dataset and for each element-containing subset.

    Args:
        data: Rows with a formula column and the target column.
        target_column: Column holding the value being plotted.
        elements: Elements to draw a separate subset for.
        formula_column: Column holding the chemical formula.
        bins: Bin edges for the subsets, as seaborn takes them; "auto" lets it choose.
        save_path: If set, save the figure here instead of showing it.
    """
    subsets = _subsets_by_element(data, elements, formula_column)

    plt.figure(figsize=(8, 6))
    sns.histplot(x=data[target_column], kde=True, label="all")
    for element, subset in subsets.items():
        sns.histplot(x=subset[target_column], kde=True, bins=bins, label=element)

    plt.xlabel(target_column)
    plt.ylabel("Count")
    plt.legend(title="Element", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_target_violin_by_element(
    data: pd.DataFrame,
    *,
    target_column: str,
    elements: Sequence[str],
    formula_column: str = "chemical formula",
    target_label: Optional[str] = None,
    title: str = "Violin Plot",
    save_path=None,
) -> None:
    """Plot the target's distribution as one violin per element-containing subset, next to the whole dataset.

    Args:
        data: Rows with a formula column and the target column.
        target_column: Column holding the value being plotted.
        elements: Elements to draw a separate violin for.
        formula_column: Column holding the chemical formula.
        target_label: Y-axis label; the column name is used when this is None.
        title: Figure title.
        save_path: If set, save the figure here instead of showing it.
    """
    subsets = _subsets_by_element(data, elements, formula_column)

    plot_data = pd.concat(
        [data[target_column].rename("all")]
        + [subset[target_column].rename(element) for element, subset in subsets.items()],
        axis=1,
    )

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=plot_data, inner="quartile")
    plt.xlabel("Element", fontsize=16)
    plt.ylabel(target_label or target_column, fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def count_compounds_by_radix(data, formula_column: str = "chemical formula") -> pd.Series:
    """Count how many compounds have each number of distinct elements.

    Returns:
        Counts indexed by radix, ascending.
    """
    return get_compound_radix(data, formula_column=formula_column).value_counts().sort_index()
