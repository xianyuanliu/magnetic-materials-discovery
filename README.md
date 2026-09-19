# Magnetic Materials Discovery

Predicts the saturation magnetization of an alloy from its chemical formula, using nine
descriptors built from periodic-table and Miedema-model reference data. Loads Novamag and
Materials Project exports, fits eight regressors, and scores them in distribution, under
composition shift, and on how well their uncertainty is earned.

The library underneath reads no config: `pipeline/` and the stage packages take a feature
matrix, a target and model specifications, so another dataset — or another property — reuses
them by calling them with different arguments. `runner.py` is this study's use of them.

## Quick Start

```bash
pip install -U numpy pandas scikit-learn matplotlib seaborn shap xgboost pymatgen

python build_feature_tables.py            # once, and again whenever sources or features change
python main.py --config configs/novamag.yaml
```

`build_feature_tables.py` writes `data/novamag-magnetism.csv` (327 compositions) and
`data/mp-magnetism.csv` (2051). Results go to the console, `plots/` and `results/`.

## Evaluation Modes

`evaluation_mode` in the config picks one:

| Mode | What it answers | Config |
| --- | --- | --- |
| `cross_validation` | How good is this model? | `configs/novamag.yaml` |
| `random_split` | Same, from repeated 80/20 draws | `configs/novamag_case_study.yaml` |
| `ood` | How much worse under composition shift, and why? | `configs/novamag_ood.yaml` |
| `uncertainty` | Are the model's own error bars earned? | `configs/novamag_uncertainty.yaml` |
| `predict` | What is this material's magnetization? | `configs/novamag_predict.yaml` |

`enable_hyperparameter_tuning`, `enable_interpretation` and `enable_data_visualization` turn on
the optional stages. `save_results` and `print_results` control saving and printing separately.

## Layout

```
main.py              CLI: parse --config, dispatch on evaluation_mode
config.py  configs/  the run config, as frozen dataclasses; unknown keys are rejected
runner.py            the only module that reads a config: it turns config values into the
                     arguments pipeline/ takes, then saves and prints what the config asks for
case_study_references.py   this study's literature measurements, passed to the case-study figure
pipeline/            one module per mode, config-free: cross_validation, random_split,
                     prediction, ood, uncertainty. They return numbers; progress goes to a
                     Logger the caller supplies, silent by default
loaddata/            dataset readers, reference tables, and every train/test split builder
prepdata/            composition parsing and the nine descriptors; no file I/O, no paths
predict/             each model's constructor, tuner and trainer as a ModelSpec, plus the
                     conformal uncertainty estimators
evaluate/            scores splits it is handed; metrics, paired significance, calibration
interpret/           figures. model_explanations and reference_comparison work on any
                     tabular model; dataset_distributions and alloy_series need the chemistry
utils/               ModelSpec registry, saving/loading, console formatting
```

Directory names follow [PyKale](https://github.com/pykale/pykale)'s stage convention.
`tests/test_layering.py` enforces the config boundary: a `from config import ...` in any file
other than the four at the top of that list fails the suite. `embed/` is empty — the descriptors are engineered by hand,
so there is no learned representation yet.

## Reading the Output

**Naming the model inputs.** `feature_columns` lists them explicitly. Leaving it out falls back
to "every column that is not the target or the formula", which promotes any stray column into a
model input. The shipped configs name them.

**OOD tables.** A shifted score alone says the model got worse, not why: holding out Fe on
Novamag also takes 57% of the training data with it. So every split is scored three ways, under
a `split_type` column:

| `split_type` | Trained on | Tested on |
| --- | --- | --- |
| `OOD` | the split's train portion | the held-out region |
| `ID-paired` | the same rows, same fitted models | the inner validation fold |
| `ID-random` | a random subset of the same size | a random test set of the same size |

`OOD − ID-paired` is the test-side shift alone; `ID-paired − ID-random` is what an unmatched
comparison would charge to the shift instead of to the smaller training pool. Table 5 reports
both. `ood_size_matched_control: false` skips the second pass.

**Seeds.** Three independent sources of randomness:

| Key | Seeds |
| --- | --- |
| `random_state` | model construction and hyperparameter sampling |
| `cv_seeds` / `cv_random_state` | the K-fold split in `cross_validation` and `ood` |
| `random_split_seeds` | the train/valid split in `random_split` |

**Search budget.** With tuning on, the search re-runs inside every outer fold, so the cost is
`cv_seeds × cv_folds × tunable_models × tune_n_iter × tune_cv_folds` fits. `tune_cv_folds` (3)
and `tune_n_iter` (20) are deliberately smaller than `cv_folds`; the run prints the fit count
before it starts.

## Saved Models

`predict` mode fits on the whole dataset — the model is going to be used, not scored — and saves
the estimator with the feature columns in training order. It also records a fingerprint of the
rows it was fitted on, and refuses to load a bundle whose fingerprint no longer matches the
configured dataset: a stale model returns entirely plausible numbers, so nothing else would say
anything was wrong. `predict_retrain: true` refits.

## Where the Reasoning Lives

Why a split family exists, why a comparison is paired the way it is, why a descriptor drops a
row rather than imputing it — those live in the docstring of the module that does it, next to
the code they constrain. `build_feature_tables.py --help` lists the dataset-preparation options.
