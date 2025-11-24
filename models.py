# models.py
"""
Define the base model constructors (without tuning logic):
- RandomForestRegressor
- XGBRegressor
- Ridge
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb


def build_rf_model(
    n_estimators: int = 200,
    max_depth: int | None = 15,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 0,
) -> RandomForestRegressor:
    """Construct a RandomForestRegressor with the project defaults."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


def build_xgb_model(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 7,
    min_child_weight: int = 1,
    subsample: float = 0.6,
    colsample_bytree: float = 0.8,
    random_state: int = 0,
) -> xgb.XGBRegressor:
    """Construct an XGBRegressor with the project defaults."""
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
    )


def build_ridge_model(alpha: float = 1.0, solver: str = "lsqr") -> Ridge:
    """Construct a Ridge model with the project defaults."""
    return Ridge(alpha=alpha, solver=solver)
