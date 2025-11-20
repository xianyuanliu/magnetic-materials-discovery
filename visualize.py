import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import alloys

def plot_ms_distribution_by_tm(data, save_path=None):
    """
    Plot histograms of saturation magnetization grouped by TM elements, such as Fe/Co/Cr/Mn.

    X : DataFrame
        Must contain 'chemical formula' and 'saturation magnetization'.
    """
    data_Fe = data[data["chemical formula"].str.contains(pat="Fe")]
    data_Co = data[data["chemical formula"].str.contains(pat="Co")]
    data_Cr = data[data["chemical formula"].str.contains(pat="Cr")]
    data_Mn = data[data["chemical formula"].str.contains(pat="Mn")]

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


def plot_violin_ms_by_tm(data, save_path=None):
    """
    Plot a violin plot of saturation magnetization grouped by TM elements.
    """
    # Subsets grouped by element
    data_Fe = data[data['chemical formula'].str.contains(pat='Fe')]
    data_Co = data[data['chemical formula'].str.contains(pat='Co')]
    data_Cr = data[data['chemical formula'].str.contains(pat='Cr')]
    data_Mn = data[data['chemical formula'].str.contains(pat='Mn')]

    # Combine into one DataFrame for seaborn violinplot
    data = pd.concat([
        data['saturation magnetization'].rename('all'),
        data_Fe['saturation magnetization'].rename('Fe'),
        data_Co['saturation magnetization'].rename('Co'),
        data_Cr['saturation magnetization'].rename('Cr'),
        data_Mn['saturation magnetization'].rename('Mn')
    ], axis=1)

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, inner="quartile")
    plt.xlabel("Element", fontsize=16)
    plt.ylabel("Saturation Magnetization (T)", fontsize=16)
    plt.title("Novamag Violin Plot", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
    
def summarize_compound_radix(data, PT):
    """
    Print the number of compounds in the dataset.
    """

    # Compute compound radix (number of unique elements in formula)
    data['compoundradix'] = alloys.get_CompoundRadix(PT, data)

    total_compound_radix = data['compoundradix'].value_counts().sort_index()

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
