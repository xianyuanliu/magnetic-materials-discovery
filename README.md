# Magnetic Materials Discovery

End-to-end pipeline for predicting saturation magnetization of alloys using engineered features from the periodic table and Miedema model data. Includes loaders for Novamag and Materials Project exports, multiple regressors, and interpretability plots.

## What's Inside
- Data loaders/cleaners for Novamag CSV exports and Materials Project `mp-data.csv`, plus periodic table and Miedema weight helpers.
- Alloy feature builder: stoichiometric array, mixing entropy, weighted atomic properties, and filtering of non-magnetic entries.
- Model zoo with optional GridSearchCV tuning: linear/ridge/lasso/elasticnet, random forest, XGBoost, SVR, and MLP.
  Scale-sensitive models (linear family, SVR, MLP) are fitted inside a `StandardScaler` pipeline, so the
  scaler is fitted on training data only and never leaks across a split; the tree ensembles are left
  unscaled on purpose (scale-invariant, and it keeps SHAP on the fast exact `TreeExplainer` path).
- Evaluation utilities: MSE/MAE/R^2 reporting, permutation importance, SHAP summaries, and FeAl/FeCo/FeCr case studies.
- Visualization helpers for magnetization histograms, violin plots, and compound radix summaries.

## Project Layout
Module directories loosely follow [PyKale](https://github.com/pykale/pykale)'s pipeline
convention (`loaddata → prepdata → predict → evaluate → interpret`, with `pipeline` for
domain-specific orchestration), simplified for this repo's scale: no `embed/` stage
(feature engineering already produces the final feature vector consumed directly by the
regressors), and no directory for a single file — `models.py` stays a flat top-level
module since it's the only "predict"-stage file and is meant to be reusable on its own
(e.g. `from models import build_rf_model`) without pulling in the training/tuning
machinery in `pipeline/train.py`.

- `main.py`: thin CLI entry point; parses `--config`, then dispatches to a holdout,
  cross-validation, or OOD run.
- `models.py`: bare model builders for all regressors (Ridge, RF, XGBoost, SVR, MLP, ...).
- `preprocess_data.py`: standalone script that builds `data/novamag-magnetism.csv` and
  `data/mp-magnetism.csv` from the raw source data.
- `loaddata/`: raw file readers — Novamag JSON, periodic table/Miedema spreadsheets
  (`alloy_access.py`), Novamag/MP raw-data cleaning (`raw_loaders.py`), and
  already-featurized CSV loaders + the holdout train/valid split (`feature_csv_access.py`).
- `prepdata/`: chemical-formula parsing and element-weighted feature engineering
  (`alloy_transform.py`), plus the higher-level feature-table builder
  (`build_features.py`).
- `pipeline/`: combines `models.py`'s builders with hyperparameter search into fittable
  units (`train.py`, exposes `MODEL_REGISTRY`); the OOD stress-test pipeline
  (`ood_pipeline.py` orchestration + `ood_scenarios.py` config-to-splits selection +
  `ood_splits.py` split-family builders: LOEO/LOPO/LOGO/LOCO/SparseX/SparseY, plus the
  two in-distribution reference builders); and the uncertainty pipeline (`uq_pipeline.py`
  orchestration + `uq.py` estimators and split-conformal calibration).
- `evaluate/`: metric primitives (`metrics.py`), K-fold CV / holdout reporting /
  significance testing (`cross_validation.py`), OOD-specific per-split scoring +
  tables (`ood_evaluation.py`), and calibration metrics (`calibration.py`).
- `interpret/`: dataset distribution plots (`visualize.py`), permutation importance +
  SHAP (`model_weights.py`), and FeAl/FeCo/FeCr literature case studies
  (`case_studies.py` + `case_study_references.py`).
- `configs/`: YAML run configs (one per dataset/mode).
- `data/`: expected inputs (`mp-data.csv`, `Novamag_Data_Files/`,
  `Periodic-table/periodic_table.xlsx`, `Miedema-model/Miedema-model-reduced.xlsx`).
- `plots/`: generated figures (permutation importance, SHAP, case studies, distributions).

## Quick Start
1) Install deps (Python 3.9+ recommended):
```bash
pip install -U numpy pandas scikit-learn matplotlib seaborn shap xgboost
```
2) Ensure data files match the paths in your chosen config (`configs/novamag.yaml`,
   `configs/mp.yaml`, or `configs/novamag_ood.yaml`).
3) Run the pipeline:
```bash
python main.py --config configs/novamag.yaml
```
   - `evaluation_mode` in the config selects `holdout`, `cross_validation`, `ood`, or
     `uq`; `enable_hyperparameter_tuning`, `enable_ablation_study`, and
     `enable_data_visualization` toggle the optional stages.
4) Check outputs in the console (metrics), `plots/` (figures prefixed by the dataset
   name), and `results/` (CSV tables from the `ood` and `uq` modes).

## Attributing an OOD Drop
An OOD score on its own says a model got worse, not why. Holding out Fe on Novamag also
takes 57% of the training data with it, and comparing against a cross-validation run
from a different config adds a tuning and fold-count difference on top. So `ood` mode
scores every split three ways and reports them side by side under a `split_type` column:

| `split_type` | Trained on | Tested on |
| --- | --- | --- |
| `OOD` | the split's train portion | the held-out region |
| `ID-paired` | the same rows, same fitted models | the inner validation fold |
| `ID-random` | a random subset of the same size | a random test set of the same size |

`ID-paired` is free — the inner K-fold already holds those rows out, they were simply
being discarded — and it is the only reference with identical training data, so
`OOD - ID-paired` is the test-side shift alone. `ID-random` costs a second pass and
holds the training-set *size* fixed, so `ID-paired - ID-random` is what an unmatched
comparison would silently charge to the shift. Table 5 reports both gaps; set
`ood_size_matched_control: false` to skip the second pass.

Leave `ood_max_splits: null` for a real run. Targets are ordered by frequency, so a
numeric cap keeps precisely the splits with the largest test sets and the least
remaining training data — a smoke-test setting, not a smaller experiment.
`ood_min_test` / `ood_min_train` drop splits too small to score meaningfully.

## Uncertainty (`evaluation_mode: uq`)
Fits a Random Forest per split and attaches three intervals: `rf_std` (tree spread read
as a Gaussian sigma), `conformal` (split conformal on absolute residuals, constant
width), and `conformal_norm` (split conformal on residuals divided by the tree spread,
so the width adapts). The first is the naive reference — tree disagreement carries no
noise or bias term, so it has no reason to be calibrated; the other two are what it
should be judged against.

Scoring is by empirical coverage of the interval and by `rms_z = rms(|error| / sigma)`,
both compared against their nominal targets, pooled per sample and repeated over
`uq_seeds`. Aggregation is per-sample throughout: averaging per-split errors while
pooling per-sample uncertainties makes the two sides of any error-vs-uncertainty ratio
incommensurable once splits differ in size, which LOCO clusters always do.

## Seeds and Search Budget
Three independent sources of randomness, each with its own config key:

| Key | Seeds |
| --- | --- |
| `random_state` | model construction and hyperparameter sampling |
| `cv_seeds` / `cv_random_state` | the K-fold split in `cross_validation` and `ood` modes |
| `holdout_seeds` | the train/valid split in `holdout` mode |

`holdout` mode repeats the split once per seed in `holdout_seeds` and reports mean ± std, because a
single 80/20 draw on a 460-sample dataset moves R² by more than the gaps between models. Ablation
plots come from the first seed only, so figure filenames stay stable.

When `enable_hyperparameter_tuning` is on, the search re-runs inside every outer fold (proper nested
CV — an outer fold's validation data never informs its own search). That makes the total cost
`cv_seeds x cv_folds x tunable_models x tune_n_iter x tune_cv_folds` model fits, which is why the
search budget has its own two keys, deliberately smaller than `cv_folds`:

- `tune_cv_folds` (default 3): inner CV folds for the search.
- `tune_n_iter` (default 20): candidates sampled by `RandomizedSearchCV` (RF/XGB/SVR/MLP).

`main.py` prints the resulting fit count before starting a tuned cross-validation run. In `ood` mode
the search additionally re-runs per split, so keep these low there.
