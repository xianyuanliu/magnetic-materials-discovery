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

Dependencies run one way. `core.py` and `config.py` are leaves that import nothing from
the stage packages; `evaluate/` scores the splits it is handed and never imports
`pipeline/`; `pipeline/` decides which splits exist and calls into `evaluate/`. Nothing
under `evaluate/` prints — all console output lives in `reporting.py`, so the scoring
functions can be called from a notebook or another project without a run's output
appearing as a side effect.

- `main.py`: thin CLI entry point; parses `--config`, then dispatches to a predict,
  holdout, cross-validation, OOD, or UQ run.
- `core.py`: shared vocabulary with no stage dependencies — the `Split` type, the
  metric list, tuning defaults, and `ModelSpec` (the typed registry entry).
- `config.py`: the whole run config as frozen dataclasses (`RunConfig`, `OODConfig`,
  `UQConfig`, `PredictConfig`). Unknown keys are rejected rather than ignored, so a
  typo in a config file is an error instead of a silently disabled setting.
- `reporting.py`: every `print_*` and display formatter in the codebase.
- `persistence.py`: `ModelBundle` — a fitted model plus the feature columns, in
  training order, that it must be given — with `save_model_bundle` / `load_model_bundle`.
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
  units (`train.py`, exposes `MODEL_REGISTRY` as `ModelSpec`s); the property-prediction
  pipeline (`predict_pipeline.py`); the OOD stress-test pipeline
  (`ood_pipeline.py` orchestration + `ood_scenarios.py` config-to-splits selection +
  `ood_splits.py` split-family builders: LOEO/LOPO/LOGO/LOCO/SparseX/SparseY, plus the
  two in-distribution reference builders); and the uncertainty pipeline (`uq_pipeline.py`
  orchestration + `uq.py` estimators and split-conformal calibration).
- `evaluate/`: metric primitives (`metrics.py`), K-fold CV scoring and paired
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
   - `evaluation_mode` in the config selects `predict`, `holdout`, `cross_validation`,
     `ood`, or `uq`; `enable_hyperparameter_tuning`, `enable_ablation_study`, and
     `enable_data_visualization` toggle the optional stages.
4) Check outputs in the console (metrics), `plots/` (figures prefixed by the dataset
   name), and `results/` (CSV tables from the `ood` and `uq` modes).

## Predicting a Property (`evaluation_mode: predict`)
The other modes answer "how good is this model"; this one answers "what is the predicted
property of this material", and is where a fitted model leaves the process.

```bash
python main.py --config configs/novamag_predict.yaml
```

The first run fits `predict_model` on the **whole** dataset — the model is going to be
used rather than scored, so holding data back would only make it worse — and saves it to
`predict_model_path`. Later runs load that file and skip training; `predict_retrain: true`
refits and overwrites it. The saved bundle carries the feature columns *in training
order* alongside the estimator, because a fitted model cannot be applied without them.

Compositions come from `predict_formulas` (a list in the config) or `predict_input_path`
(a CSV with the formula column), are featurized by the same `add_engineered_features`
call the training data went through, and are written to `predict_output_path`. A formula
that cannot be parsed yields NaN features rather than a plausible-looking zero vector, so
it is reported as skipped instead of silently predicted.

## Naming the Model Inputs
`feature_columns` in the config lists the model inputs explicitly. Leaving it out falls
back to "every column that is not the target or the formula", which promotes any stray
column in the CSV into a model input — a `sample_id` leaks row order into the model, and
a text column reaches the scaler as a string. The fallback now rejects non-numeric
columns rather than passing them through, but naming the features is the reliable
version and is what the shipped configs do.

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

Hold-out targets are named per family — `ood_elements` (symbols), `ood_periods` and
`ood_groups` (integers) — because one shared list cannot serve all three.

### Comparing two models
`compare_models: [rf, xgb]` selects the pair reported in Table 3. The comparison is
paired **across OOD splits**, one observation per split: each split is a different test
set and both models saw identical training data on it. It is deliberately not paired
across the inner folds of a single split, which all score the *same* fixed test set —
those are repeated measurements of one quantity, not independent observations of a
difference, and pairing them inflates the apparent evidence. Below
`MIN_PAIRS_FOR_TEST` (6) paired observations no p-value is reported at all, because a
two-sided signed-rank test on n pairs cannot go below `2 / 2**n` — with three splits its
floor is 0.25, which reads as "not significant" when it is really "this test cannot
answer that".

## Uncertainty (`evaluation_mode: uq`)
Fits the model named by `uq_model` per split. That model must declare
`provides_ensemble_std` in the registry — the estimators read the spread of an ensemble's
members, so the pipeline checks the capability rather than assuming a Random Forest.
Fits it per split and attaches three intervals: `rf_std` (tree spread read
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
