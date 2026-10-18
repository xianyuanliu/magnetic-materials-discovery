"""Figures drawn from saved result tables: per-split scores by model and OOD generalization gaps."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kalematerials.loaddata.splits import ID_KFOLD, ID_RANDOM, ID_REFERENCE, OOD

# Panel order for split types; others follow alphabetically.
SPLIT_TYPE_ORDER = (OOD, ID_REFERENCE, ID_RANDOM, ID_KFOLD)


def plot_metric_by_model(results: pd.DataFrame, *, metric: str = "mse", save_path=None) -> None:
    """Box plot of per-split `metric` by scenario and model, one panel per split type in SPLIT_TYPE_ORDER.

    Models are colored in name order, as in `plot_generalization_gap`.
    `results` uses evaluate.metrics.RESULT_COLUMNS. Save to `save_path` when set; otherwise show the figure.
    """
    rows = results[results["metric"] == metric]
    if rows.empty:
        raise ValueError(f"No {metric!r} rows to plot; metrics present: {sorted(results['metric'].unique())}")

    present = set(rows["split_type"])
    split_types = [t for t in SPLIT_TYPE_ORDER if t in present] + sorted(present - set(SPLIT_TYPE_ORDER))
    models = sorted(rows["model"].unique())
    n_scenarios = rows["scenario"].nunique()
    fig, axes = plt.subplots(
        1,
        len(split_types),
        figsize=(max(6.0, 1.6 * n_scenarios + 3.0) * len(split_types), 5.0),
        squeeze=False,
        sharey=True,
    )

    for ax, split_type in zip(axes[0], split_types):
        part = rows[rows["split_type"] == split_type]
        sns.boxplot(data=part, x="scenario", y="value", hue="model", hue_order=models, ax=ax, fliersize=0)
        # A swarm places points deterministically, so redrawing a saved run gives the same figure.
        sns.swarmplot(
            data=part, x="scenario", y="value", hue="model", hue_order=models, ax=ax, dodge=True, size=3, legend=False
        )
        ax.set_title(split_type, fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper(), fontsize=14)
        ax.legend(title="Model", fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_generalization_gap(generalization_gap: pd.DataFrame, *, save_path=None) -> None:
    """Grouped bars of shift_gap and train_pool_gap by scenario and model.

    `generalization_gap` follows interpret.result_analysis.summarize_generalization_gap.
    Save to `save_path` when set; otherwise show the figure.
    """
    if generalization_gap.empty:
        raise ValueError("plot_generalization_gap needs a nonempty generalization-gap table.")

    n_scenarios = generalization_gap["scenario"].nunique()
    models = sorted(generalization_gap["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(2 * max(6.0, 1.4 * n_scenarios + 3.0), 5.0), squeeze=False)

    titles = {
        "shift_gap": "Shift gap (OOD − ID-reference)",
        "train_pool_gap": "Training-pool gap (ID-reference − ID-random)",
    }
    for ax, (column, title) in zip(axes[0], titles.items()):
        sns.barplot(data=generalization_gap, x="scenario", y=column, hue="model", hue_order=models, ax=ax)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("MSE difference", fontsize=14)
        ax.legend(title="Model", fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
