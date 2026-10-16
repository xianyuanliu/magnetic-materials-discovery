"""Run an experiment with Hydra configuration overrides.

    python main.py dataset=novamag workflow=ood
    python main.py dataset=materials_project workflow=cross_validation tuning.enabled=false kfold.seeds=[0]

`dataset` and `workflow` select config files; other arguments override individual settings.
"""

from typing import cast

import matplotlib

matplotlib.use("Agg")

import hydra
import runner
from config import RunConfig, validate
from omegaconf import DictConfig, OmegaConf

from kalematerials.predict.sklearn_models import MODEL_REGISTRY


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(composed: DictConfig) -> None:
    """Run the configured evaluation or prediction job."""
    cfg = cast(RunConfig, OmegaConf.to_object(composed))
    validate(cfg)

    if cfg.workflow == "predict":
        runner.run_predict(cfg=cfg, registry=MODEL_REGISTRY)
        return

    if cfg.workflow == "cross_validation":
        runner.run_cross_validation(cfg=cfg, registry=MODEL_REGISTRY)
    elif cfg.workflow == "ood":
        runner.run_ood(cfg=cfg, registry=MODEL_REGISTRY)
    elif cfg.workflow == "uncertainty":
        runner.run_uncertainty(cfg=cfg, registry=MODEL_REGISTRY)
    else:
        runner.run_random_split(cfg=cfg, registry=MODEL_REGISTRY)


if __name__ == "__main__":
    main()
