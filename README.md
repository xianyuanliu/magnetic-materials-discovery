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
  units (`train.py`, exposes `MODEL_REGISTRY`), and the OOD stress-test pipeline
  (`ood_pipeline.py` orchestration + `ood_splits.py` split-family builders: LOEO/LOPO/
  LOGO/LOCO/SparseX/SparseY).
- `evaluate/`: metric primitives (`metrics.py`), K-fold CV / holdout reporting /
  significance testing (`cross_validation.py`), and OOD-specific per-split scoring +
  tables (`ood_evaluation.py`).
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
   - `evaluation_mode` in the config selects `holdout`, `cross_validation`, or `ood`;
     `enable_hyperparameter_tuning`, `enable_ablation_study`, and
     `enable_data_visualization` toggle the optional stages.
4) Check outputs in the console (metrics) and `plots/` (figures prefixed by the dataset name).
