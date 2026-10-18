# Magnetic Materials Discovery

Predicts the saturation magnetization of an alloy from its chemical formula, using nine
descriptors built from periodic-table and Miedema-model reference data. Loads Novamag and
Materials Project exports, fits eight regressors, and scores them in distribution, under
composition shift, and on their uncertainty.

## Quick Start

```bash
pip install -r requirements.txt

python build_dataset.py
python main.py dataset=novamag workflow=cross_validation
```

`build_dataset.py` writes `data/novamag-magnetism.csv` (327 compositions) and
`data/mp-magnetism.csv` (2051). Each run writes `results/<dataset>_<workflow>/results.csv`, one row per
(scenario, split_type, split_id, seed, model, metric), and `summary.csv`, its mean and standard
deviation. Figures go to `plots/`.

## Running

A run is one dataset and one workflow. Hydra composes `configs/config.yaml` with
`configs/dataset/<name>.yaml` and `configs/workflow/<name>.yaml`; other arguments
override single keys:

```bash
python main.py dataset=materials_project workflow=ood
python main.py dataset=novamag workflow=cross_validation tuning.enabled=false 'kfold.seeds=[0]'
python main.py dataset=novamag workflow=ood 'ood.scenarios=[element]' 'ood.elements=[Fe,Co]'
```

| `workflow=` | Purpose |
| --- | --- |
| `cross_validation` | Repeated K-fold evaluation |
| `random_split` | Repeated 80/20 splits, with interpretation figures |
| `ood` | Evaluation under composition shift |
| `uncertainty` | Calibration of prediction intervals |
| `predict` | Predictions for new formulas |
| `interpret` | Re-derive the analysis tables from a saved run (`interpret.results_dir`) |

## Results

`cross_validation`, `random_split` and `ood` write the same files to `results/<dataset>_<workflow>/`:

| File | Contents |
| --- | --- |
| `results.csv` | one row per (scenario, split_type, split_id, seed, model, metric) |
| `summary.csv` | mean, standard deviation and count per (scenario, split_type, model, metric) |
| `splits.csv` | `n_train` and `n_test` per split and seed |

The interpret step then derives, from those files alone, whatever the results support:

| File | Written when |
| --- | --- |
| `significance.csv` | `compare_models` names two models: a paired test per scenario, split type and metric |
| `generalization_gap.csv` | the results contain OOD and ID-reference rows (the `ood` workflow) |

K-fold and random-split comparisons pair every (seed, split) and correct the variance by
`1/n + n_test/n_train` (Nadeau and Bengio), so repeated K-fold pools all seeds into one test.
OOD comparisons average seeds per held-out set and pair the held-out sets.
`python main.py dataset=novamag workflow=interpret 'compare_models=[ridge,rf]' interpret.results_dir=results/novamag_ood`
recomputes them for a saved run.

## Layout

```
main.py              CLI entry point
config.py            config schema
configs/             config.yaml + dataset/ + workflow/
runner.py            turns a config into kalematerials calls, saves and prints results
build_dataset.py     builds the feature tables from the raw sources
case_study_references.py   literature measurements for the case-study figure
data/                raw sources and feature tables
kalematerials/
  loaddata/          dataset readers, reference tables, train/test splits
  prepdata/          composition parsing, stoichiometry, atomic fractions
  embed/             composition descriptors
  predict/           models, fitting, conformal uncertainty
  evaluate/          metrics, calibration, result summaries
  interpret/         significance tests, generalization gaps, explanations, figures
  pipeline/          one module per workflow, plus the feature-table build and interpretation
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
(training-pool gap), both in MSE.

## Saved Models

`cross_validation` and `random_split` refit the model with the best mean MSE on all rows and save
it to `models/<dataset>_<workflow>.joblib`. `predict` loads `models/<dataset>_cross_validation.joblib`;
`predict.model=<key>` fits a named model instead and `predict.retrain=true` refits. A bundle whose
training-data fingerprint does not match the configured dataset is refused.
