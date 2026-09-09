"""CLI entry point: run one evaluation mode against a YAML run config."""

import argparse
from pathlib import Path

from config import RunConfig, load_run_config
from interpret.visualize import plot_ms_distribution_by_tm, plot_violin_ms_by_tm, summarize_compound_radix
from loaddata.feature_csv_access import load_raw_data
from pipeline.cross_validation_pipeline import run_cross_validation
from pipeline.holdout_pipeline import run_holdout
from pipeline.model import MODEL_REGISTRY
from pipeline.ood_pipeline import run_ood_evaluation
from pipeline.predict_pipeline import run_predict
from pipeline.uq_pipeline import run_uq_evaluation


def parse_args() -> argparse.Namespace:
    """Parse the --config CLI flag."""
    parser = argparse.ArgumentParser(description="Train and evaluate ML models for material property prediction")
    parser.add_argument("--config", type=str, default="./configs/novamag.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


def run_data_visualization(cfg: RunConfig, plots_dir: Path) -> None:
    """Plot the target's distribution across the transition-metal subsets."""
    raw = load_raw_data(cfg.dataset_path)
    plot_ms_distribution_by_tm(raw, save_path=plots_dir / f"{cfg.prefix}_ms_distribution_by_tm.png")
    plot_violin_ms_by_tm(
        raw,
        title=f"{cfg.prefix.upper()} Violin Plot",
        save_path=plots_dir / f"{cfg.prefix}_violin_ms_by_tm.png",
    )
    summarize_compound_radix(raw)


def main() -> None:
    """Run one predict/holdout/cross_validation/ood/uq job, per the --config file."""
    args = parse_args()
    cfg = load_run_config(args.config)

    plots_dir = Path(cfg.plots_output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if cfg.evaluation_mode == "predict":
        run_predict(cfg=cfg, model_registry=MODEL_REGISTRY)
        return

    if cfg.evaluation_mode == "cross_validation":
        run_cross_validation(cfg, MODEL_REGISTRY)
    elif cfg.evaluation_mode == "ood":
        run_ood_evaluation(cfg=cfg, model_registry=MODEL_REGISTRY)
    elif cfg.evaluation_mode == "uq":
        run_uq_evaluation(cfg=cfg, model_registry=MODEL_REGISTRY)
    else:
        run_holdout(cfg, MODEL_REGISTRY, plots_dir)

    # Data visualization reads the raw dataset, which OOD mode does not require.
    if cfg.enable_data_visualization and cfg.evaluation_mode != "ood":
        run_data_visualization(cfg, plots_dir)


if __name__ == "__main__":
    main()
