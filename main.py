"""CLI entry point: run one evaluation mode against a YAML run config."""

import argparse

from config import load_run_config
from examples import cross_validation, random_split
from pipeline.data_visualization import run_data_visualization
from pipeline.inference_pipeline import run_predict
from pipeline.ood_pipeline import run_ood_evaluation
from pipeline.uncertainty_pipeline import run_uncertainty_evaluation
from predict.sklearn_models import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    """Parse the --config CLI flag."""
    parser = argparse.ArgumentParser(description="Train and evaluate ML models for material property prediction")
    parser.add_argument("--config", type=str, default="./configs/novamag.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


def main() -> None:
    """Run one predict, random_split, cross_validation, ood or uncertainty job, per the --config file."""
    args = parse_args()
    cfg = load_run_config(args.config)

    if cfg.evaluation_mode == "predict":
        run_predict(cfg=cfg, registry=MODEL_REGISTRY)
        return

    if cfg.evaluation_mode == "cross_validation":
        cross_validation.run(cfg=cfg, registry=MODEL_REGISTRY)
    elif cfg.evaluation_mode == "ood":
        run_ood_evaluation(cfg=cfg, registry=MODEL_REGISTRY)
    elif cfg.evaluation_mode == "uncertainty":
        run_uncertainty_evaluation(cfg=cfg, registry=MODEL_REGISTRY)
    else:
        random_split.run(cfg=cfg, registry=MODEL_REGISTRY)

    if cfg.enable_data_visualization:
        run_data_visualization(cfg)


if __name__ == "__main__":
    main()
