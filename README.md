# Magnetic Materials Discovery

Predicts the saturation magnetization of an alloy from its chemical formula, using nine
descriptors built from periodic-table and Miedema-model reference data. Loads Novamag and
Materials Project exports, fits eight regressors, and scores them in distribution, under
composition shift, and on their uncertainty.

## Quick Start

```bash
pip install -r requirements.txt

python build_dataset.py
python main.py dataset=novamag mode=cross_validation
```

`build_dataset.py` writes `data/novamag-magnetism.csv` (327 compositions) and
`data/mp-magnetism.csv` (2051). Each run writes `results/<dataset>_<mode>/results.csv`, one row per
(scenario, split_type, split_id, seed, model, metric), and `summary.csv`, its mean and standard
deviation. Figures go to `plots/`.

## Running

A run is one dataset and one mode. Hydra composes `configs/config.yaml` with `configs/dataset/<name>.yaml`
and `configs/mode/<name>.yaml`; other arguments override single keys:

```bash
python main.py dataset=materials_project mode=ood
python main.py dataset=novamag mode=cross_validation tuning.enabled=false 'kfold.seeds=[0]'
python main.py dataset=novamag mode=ood 'ood.scenarios=[element]' 'ood.elements=[Fe,Co]'
```

| `mode=` | Purpose |
| --- | --- |
| `cross_validation` | Repeated K-fold evaluation |
| `random_split` | Repeated 80/20 splits, with interpretation figures |
| `ood` | Evaluation under composition shift |
| `uncertainty` | Calibration of prediction intervals |
| `predict` | Predictions for new formulas |

## Layout

```
main.py              CLI entry point
config.py            config schema
configs/             config.yaml + dataset/ + mode/
runner.py            turns a config into kalematerials calls, saves and prints results
build_dataset.py     builds the feature tables from the raw sources
case_study_references.py   literature measurements for the case-study figure
data/                raw sources and feature tables
kalematerials/
  loaddata/          dataset readers, reference tables, train/test splits
  prepdata/          composition parsing, stoichiometry, atomic fractions
  embed/             composition descriptors
  predict/           models, fitting, conformal uncertainty
  evaluate/          metrics, significance, calibration, OOD summaries
  interpret/         figures
  pipeline/          one workflow per mode, plus the feature-table build
  utils/             model registry, persistence, reporting
  resources/         periodic-table and Miedema spreadsheets
```

## OOD Output

Each OOD split is scored three ways, under `split_type`:

| `split_type` | Trained on | Tested on |
| --- | --- | --- |
| `OOD` | the split's train portion | the held-out region |
| `ID-reference` | the same rows and models | an inner validation fold |
| `ID-random` | a random subset of the same size | a random test set of the same size |

`generalization_gap.csv` reports `OOD − ID-reference` (shift gap) and `ID-reference − ID-random`
(training-pool gap).

## Saved Models

`cross_validation` and `random_split` refit the model with the best mean MSE on all rows and save
it to `models/<dataset>_<mode>.joblib`. `predict` loads `models/<dataset>_cross_validation.joblib`;
`predict.model=<key>` fits a named model instead and `predict.retrain=true` refits. A bundle whose
training-data fingerprint does not match the configured dataset is refused.
