"""CLI entry point: run one evaluation mode against a YAML run config."""

import argparse
from pathlib import Path

from config import load_run_config
from pipeline.cross_validation_pipeline import run_cross_validation
from pipeline.data_visualization import run_data_visualization
from pipeline.holdout_pipeline import run_holdout
from pipeline.inference_pipeline import run_predict
from pipeline.ood_pipeline import run_ood_evaluation
from pipeline.uq_pipeline import run_uq_evaluation
from predict.sklearn_models import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    """Parse the --config CLI flag."""
    parser = argparse.ArgumentParser(description="Train and evaluate ML models for material property prediction")
    parser.add_argument("--config", type=str, default="./configs/novamag.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


def main() -> None:
    """Run one predict/holdout/cross_validation/ood/uq job, per the --config file."""
    args = parse_args()
    cfg = load_run_config(args.config)

    plots_dir = Path(cfg.plots_output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if cfg.evaluation_mode == "predict":
        run_predict(cfg=cfg, registry=MODEL_REGISTRY)
        return

    if cfg.evaluation_mode == "cross_validation":
        run_cross_validation(cfg=cfg, registry=MODEL_REGISTRY)
    elif cfg.evaluation_mode == "ood":
        run_ood_evaluation(cfg=cfg, registry=MODEL_REGISTRY)
    elif cfg.evaluation_mode == "uq":
        run_uq_evaluation(cfg=cfg, registry=MODEL_REGISTRY)
    else:
        run_holdout(cfg=cfg, registry=MODEL_REGISTRY, plots_dir=plots_dir)

    if cfg.enable_data_visualization:
        run_data_visualization(cfg, plots_dir)


if __name__ == "__main__":
    main()
