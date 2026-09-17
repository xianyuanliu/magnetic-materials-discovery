"""Dataset distribution plots, drawn from the feature table rather than from any model's predictions.

Optional and independent of the evaluation modes: nothing downstream reads what this writes, so a run that skips it
produces the same metrics.
"""

from pathlib import Path

from config import RunConfig
from interpret.visualize import count_compounds_by_radix, plot_ms_distribution_by_tm, plot_violin_ms_by_tm
from utils.reporting import print_compound_counts
from loaddata.tabular_access import load_feature_table


def run_data_visualization(cfg: RunConfig) -> None:
    """Plot the target's distribution across the transition-metal subsets.

    Args:
        cfg: The resolved run configuration; `dataset_path` must be set, and `plots_output_dir` names where the
            figures go.

    Raises:
        ValueError: If the config names no `dataset_path`, which OOD configs do not because they name a train and a
            test file instead.
    """
    if not cfg.dataset_path:
        raise ValueError(
            "enable_data_visualization needs dataset_path, but this config sets none. OOD configs name "
            "train_dataset_path and test_dataset_path instead; point dataset_path at the table you want plotted."
        )

    plots_dir = Path(cfg.plots_output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _, y, metadata = load_feature_table(
        cfg.dataset_path,
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )
    data = metadata.assign(**{cfg.target_column: y})

    plot_ms_distribution_by_tm(data, save_path=plots_dir / f"{cfg.prefix}_ms_distribution_by_tm.png")
    plot_violin_ms_by_tm(
        data,
        title=f"{cfg.prefix.upper()} Violin Plot",
        save_path=plots_dir / f"{cfg.prefix}_violin_ms_by_tm.png",
    )
    print_compound_counts(count_compounds_by_radix(data))
