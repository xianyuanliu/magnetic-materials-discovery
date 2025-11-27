# Magnetic Materials Discovery

End-to-end pipeline for predicting saturation magnetization of alloys using engineered features from the periodic table and Miedema model data. Includes loaders for Novamag and Materials Project exports, multiple regressors, and interpretability plots.

## What's Inside
- Data loaders/cleaners for Novamag CSV exports and Materials Project `mp-data.csv`, plus periodic table and Miedema weight helpers.
- Alloy feature builder: stoichiometric array, mixing entropy, weighted atomic properties, and filtering of non-magnetic entries.
- Model zoo with optional GridSearchCV tuning: linear/ridge/lasso/elasticnet, random forest, XGBoost, SVR, and MLP.
- Evaluation utilities: MSE/MAE/R^2 reporting, permutation importance, SHAP summaries, and FeAl/FeCo/FeCr case studies.
- Visualization helpers for magnetization histograms, violin plots, and compound radix summaries.

## Project Layout
- `main.py`: orchestrates the full run; toggle dataset (`MP` or `Novamag`), hyperparameter tuning, and plotting flags here.
- `data.py`: raw data ingestion, cleaning, feature engineering, and train/validation splits.
- `train.py` / `models.py`: model builders plus optional hyperparameter search grids.
- `evaluate.py`: metrics, permutation importance, SHAP, and case-study plotting.
- `visualize.py`: dataset distribution plots.
- `data/`: expected inputs (`mp-data.csv`, `Novamag_Data_Files/`, `Periodic-table/periodic_table.xlsx`, `Miedema-model/Miedema-model-reduced.xlsx`).
- `plots/`: generated figures (permutation importance, SHAP, case studies, distributions).

## Quick Start
1) Install deps (Python 3.9+ recommended):
```bash
pip install -U numpy pandas scikit-learn matplotlib seaborn shap xgboost
```
2) Ensure data files match the paths in `main.py` (`data/mp-data.csv` or `data/Novamag_Data_Files/` plus the periodic table and Miedema Excel files).
3) Run the pipeline:
```bash
python main.py
```
   - Adjust `dataset_name`, `hyperparameter_tuning`, and `data_visualization` flags near the top of `main.py`.
4) Check outputs in the console (metrics) and `plots/` (figures prefixed by the dataset name).

## Tests
Run `pytest` to execute `tests/test_alloys.py` (data files required).
