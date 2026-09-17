# Magnetic Materials Discovery

End-to-end pipeline for predicting saturation magnetization of alloys using engineered features from the periodic table and Miedema model data. Includes loaders for Novamag and Materials Project exports, multiple regressors, and interpretability plots.

## What's Inside
- Dataset readers for Novamag JSON records and Materials Project `mp-data.csv`, plus shared reference-table and CSV access.
- Alloy feature builder: magnetic-record selection, normalized-composition aggregation, and composition descriptors.
- Model zoo with optional GridSearchCV tuning: linear/ridge/lasso/elasticnet, random forest, XGBoost, SVR, and MLP.
  Scale-sensitive models (linear family, SVR, MLP) are fitted inside a `StandardScaler` pipeline, so the
  scaler is fitted on training data only and never leaks across a split; the tree ensembles are left
  unscaled on purpose (scale-invariant, and it keeps SHAP on the fast exact `TreeExplainer` path).
- Evaluation utilities: MSE/MAE/R^2 reporting, permutation importance, SHAP summaries, and FeAl/FeCo/FeCr case studies.
- Visualization helpers for magnetization histograms, violin plots, and compound radix summaries.

## Project Layout
Module directories follow [PyKale](https://github.com/pykale/pykale)'s stage convention
(`loaddata → prepdata → embed → predict → evaluate → interpret`, with `pipeline` for the
workflows that string them together and `utils` for what more than one stage shares).
`embed/` is empty for now: the nine features are engineered by hand in `prepdata/`, so
there is no learned representation yet; it is where deep-learning encoders will go.

Dependencies run one way. `config.py` and `utils/registry.py` are leaves that import
nothing from the stage packages; `evaluate/` scores the splits it is handed and never
imports `pipeline/`; `pipeline/` decides which splits exist and calls into `evaluate/`.
Nothing under `evaluate/` prints — all console output lives in `utils/reporting.py`,
which formats what `evaluate/` returns, so the scoring functions can be called from a
notebook or another project without a run's output appearing as a side effect.

- `main.py`: thin CLI entry point; parses `--config`, then dispatches to a predict,
  holdout, cross-validation, OOD, or UQ run.
- `utils/`: infrastructure shared by more than one stage package, not itself a stage.
  - `registry.py`: `ModelSpec` (what a model declares about itself) and `resolve_models`
    (config keys to specs). A leaf, because `evaluate/` needs `ModelSpec` too and must
    not import `pipeline/` to get it.
  - `reporting.py`: every `print_*` and display formatter in the codebase.
  - `persistence.py`: what a run leaves behind — `ModelBundle` (a fitted model plus the feature
    columns, in training order, that it must be given) with `save_model_bundle` /
    `load_model_bundle`, and `save_results`, which every mode writes its numbers through.
- `config.py`: the whole run config as frozen dataclasses (`RunConfig`, `KFoldConfig`,
  `HoldoutConfig`, `TuningConfig`, `OODConfig`, `UQConfig`, `PredictConfig`). Unknown
  keys are rejected rather than ignored, so a typo in a config file is an error instead
  of a silently disabled setting.
- `build_feature_tables.py`: standalone entry point that builds the feature tables.
  Builds `data/novamag-magnetism.csv` and `data/mp-magnetism.csv`; `main.py` reuses these files.
- `loaddata/`: owns the data and how it is organized — which rows exist, and which of them are
  held out from which.
  - `novamag.py`: reads local JSON records, flattens nested fields and standardizes names and types.
  - `materials_project.py`: reads a local MP CSV export and converts magnetization to tesla.
  - `tabular_access.py`: shared record standardization, periodic-table/Miedema spreadsheet readers,
    and feature-table loading with explicit model-feature selection.
  - `splits.py`: every train/test split builder — plain K-fold, a single holdout, the size-matched
    control, and the OOD families. A split is a property of the data, not of the metric computed on
    it, so the same held-out region applies to every modality a sample carries; WILDS and PyKale
    organize theirs the same way.
  Both dataset readers return formula and numeric target columns plus source IDs and metadata.
  They preserve original formula stoichiometry and do not select magnetic records or exclude elements.
  No loader imports `prepdata`, and loading never triggers a download.
  To add a dataset, implement its source mapping and call `standardize_records`, then add
  its preparation step in `build_feature_tables.py`.
- `prepdata/`: composition handling, alloy descriptors and feature-table assembly.
  - `composition.py`: formula parsing, normalized composition keys, stoichiometry, atomic fractions
    and reusable descriptor primitives.
  - `alloy_descriptors.py`: this project's nine features, also used for target-free inference.
  - `feature_table.py`: configurable sample selection, then median targets per normalized composition,
    then feature calculation. Only the target is aggregated; source metadata stays in loaded records.
  Nothing in `prepdata/` reads a file or takes a path; `loaddata/` hands it loaded frames.
- `embed/`: learned representations. Empty until deep-learning encoders are added.
- `predict/`: every model's bare constructor, tuner and trainer, one block each, combined
  into fittable `ModelSpec`s (`sklearn_models.py`, exposes `MODEL_REGISTRY`); and the predictive-
  uncertainty estimators with split-conformal calibration (`uncertainty.py`).
- `pipeline/`: one orchestration module per evaluation mode — `holdout_pipeline.py`,
  `cross_validation_pipeline.py`, `inference_pipeline.py`, `uq_pipeline.py`, `ood_pipeline.py`
  (with `ood_scenarios.py` turning the config into a list of scenarios), plus the optional dataset
  distribution plots (`data_visualization.py`). Nothing else lives here: helpers a mode needs but
  does not orchestrate belong in `utils/` or `evaluate/`.
  Every evaluation mode has the same shape: resolve the models, load the feature table, build
  splits from `loaddata/splits.py`, score them, then write one `results.csv` of raw per-split
  numbers and one `summary.csv` of means and standard deviations. Only the split family and the tables
  differ, so a new mode is a new choice of those two rather than a new pipeline shape.
- `evaluate/`: scores the splits it is handed and decides nothing about them — metric
  primitives (`metrics.py`), per-split scoring and paired significance testing
  (`cross_validation.py`), OOD scoring and its summary tables (`ood_evaluation.py`), and
  calibration metrics (`calibration.py`). `KFold` and `train_test_split` appear nowhere here;
  the caller passes `(split_id, train_idx, test_idx)` tuples, so swapping in a grouped or
  shifted split changes a line in `pipeline/` and nothing in `evaluate/`.
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
3) Prepare the datasets once, or again after changing source data, selection rules or features:
```bash
python build_feature_tables.py
```
   This reads local sources and saves the feature CSVs used by the run configs. It also saves
   flattened Novamag records with original formulas and source IDs to `data/novamag/novamag-raw.csv`.
4) Run the experiment pipeline:
```bash
python main.py --config configs/novamag.yaml
```
   - `evaluation_mode` in the config selects `predict`, `holdout`, `cross_validation`,
     `ood`, or `uq`; `enable_hyperparameter_tuning`, `enable_ablation_study`, and
     `enable_data_visualization` toggle the optional stages.
5) Check outputs in the console (metrics), `plots/` (figures prefixed by the dataset
   name), and `results/` (CSV tables from the `ood` and `uq` modes).

## Dataset Preparation and Sample Definition

`build_feature_tables.py` is independent of model training. `main.py` reads the saved feature tables and does not
rebuild them automatically. The preparation command replaces its output files, so use `--output-dir`
when comparing preparation settings. Existing model bundles and experiment results are not regenerated;
retrain and re-evaluate them after changing the feature tables.

```bash
# Prepare only MP with the existing study exclusions and target >= 0.18 T.
python build_feature_tables.py --dataset mp

# Compare an alternative magnetic threshold in a separate output directory.
python build_feature_tables.py --min-target 0.3 --output-dir /tmp/magnetic-data-03

# Explicitly exclude these elements for every selected dataset.
python build_feature_tables.py --exclude-elements Nd Sm U

# Disable element exclusions (pass no element values).
python build_feature_tables.py --dataset mp --exclude-elements
```

Defaults retain the existing study scope: both datasets use the inclusive 0.18 T record threshold;
MP additionally requires `is_magnetic=True` and excludes the existing rare-earth/actinide list.
Novamag has no default element exclusions and is selected by the target threshold.
`--mp-include-nonmagnetic` disables only MP's magnetic-label filter, leaving the target threshold active.
A requested magnetic-label filter requires a standardized boolean column; missing labels do not qualify.
The reusable `filter_samples` function also accepts `min_target=None` to disable the threshold.

The target is the median magnetization **among selected magnetic records** for each composition:
source normalization → record selection → composition normalization and median target → features.
`FeNi`, `Fe2Ni2`, `NiFe` and `Fe0.5Ni0.5` share the key `FeNi`. Keys use pymatgen's
`Composition.get_integer_formula_and_factor()` with the default `max_denominator=10000` approximation
for fractional amounts, then `hill_formula` for consistent formatting. Hill ordering avoids the
CSV missing-value token `NaN` by writing sodium/nitrogen as `NNa`. Keys are written as the feature table's
`chemical formula` column, preserving compatibility with formula-based evaluation and inference.
Different structures of the same composition contribute to the median; the output is a composition
sample, not an individual structure. Original records remain available separately for traceability.
Filtering takes place before the median: records with targets 0.05, 0.10 and 1.50 T yield 1.50 T
under the default threshold. No low-target records contribute to that median.

Run the preparation regression checks with:
```bash
python -m unittest discover -s tests -v
```

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
Fits the model named by `uq_model` per split. That model must be in
`pipeline.uq_pipeline.ENSEMBLE_STD_MODELS` — the estimators read the spread of an
ensemble's members, so the pipeline checks this rather than assuming a Random Forest.
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
