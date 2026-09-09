"""Permutation feature importance and SHAP summary plots for a trained model."""

import os
from typing import Optional

# Must be set before `import shap` pulls in cv2: cv2 bundles its own Qt platform plugin, which collides with the system
# one and aborts the process (SIGABRT) as soon as a subprocess is forked for hyperparameter tuning. This process never
# renders a GUI, so the offscreen platform is always correct.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

from sklearn.inspection import permutation_importance  # noqa: E402


def plot_permutation_importance(
    model,
    X_valid,
    y_valid,
    title: str = "",
    save_path: Optional[str] = None,
    random_state: int = 0,
):
    """Plot permutation importance for RFR / XGB / Ridge."""
    perm_import = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=random_state)

    sorted_idx = perm_import.importances_mean.argsort()

    plt.figure(figsize=(14, 7))
    plt.barh(range(len(sorted_idx)), perm_import.importances_mean[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), X_valid.columns[sorted_idx], fontsize=16)
    plt.xlabel("Permutation Feature Importance", fontsize=16)
    plt.ylabel("Features", fontsize=16)
    plt.xticks(fontsize=16)
    if title:
        plt.title(title, fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_shap_summary(model, X_train, X_valid, save_path: Optional[str] = None):
    """Generate SHAP summary plots."""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_valid, check_additivity=False)

    shap.summary_plot(shap_values, X_valid, feature_names=X_valid.columns, show=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
