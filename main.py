"""CLI entry point: compose a run config with Hydra and dispatch on its evaluation mode.

    python main.py dataset=novamag mode=ood
    python main.py dataset=mp mode=cross_validation tuning.enabled=false kfold.seeds=[0]

The first two arguments pick a file from configs/dataset/ and configs/mode/; anything else overrides one key of the
composed config. Hydra checks the result against the schema in config.py before this file sees it.
"""

from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

import runner
from config import RunConfig, validate
from predict.sklearn_models import MODEL_REGISTRY


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(composed: DictConfig) -> None:
    """Run one predict, random_split, cross_validation, ood or uncertainty job."""
    cfg = cast(RunConfig, OmegaConf.to_object(composed))
    validate(cfg)

    if cfg.evaluation_mode == "predict":
        runner.run_predict(cfg=cfg, registry=MODEL_REGISTRY)
        return

    if cfg.evaluation_mode == "cross_validation":
        runner.run_cross_validation(cfg=cfg, registry=MODEL_REGISTRY)
    elif cfg.evaluation_mode == "ood":
        runner.run_ood(cfg=cfg, registry=MODEL_REGISTRY)
    elif cfg.evaluation_mode == "uncertainty":
        runner.run_uncertainty(cfg=cfg, registry=MODEL_REGISTRY)
    else:
        runner.run_random_split(cfg=cfg, registry=MODEL_REGISTRY)

    if cfg.enable_data_visualization:
        runner.run_data_visualization(cfg=cfg)


if __name__ == "__main__":
    main()
