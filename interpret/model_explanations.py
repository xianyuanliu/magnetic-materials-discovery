"""Why a fitted model predicts what it does: permutation importance and SHAP.

Neither method reads the model's parameters — permutation importance shuffles one feature and measures the score it
costs, SHAP attributes each prediction across the features that produced it — so both apply to any estimator with a
`predict`, not only to the linear ones whose coefficients can be read off.
"""

import os
from pathlib import Path
from typing import Optional

# Must be set before `import shap` pulls in cv2: cv2 bundles its own Qt platform plugin, which collides with the system
# one and aborts the process (SIGABRT) as soon as a subprocess is forked for hyperparameter tuning. This process never
# renders a GUI, so the offscreen platform is always correct.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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


def plot_shap_summary(model, X_train, X_valid, save_path: Optional[str] = None, random_state: int = 0):
    """Plot the SHAP summary: per-feature attributions for every validation row.

    Args:
        model: A fitted estimator.
        X_train: Rows the model was fitted on, used as the background distribution.
        X_valid: Rows the attributions are computed for.
        save_path: Where the figure goes; shown instead when None.
        random_state: Seeds the beeswarm jitter, so the same values give the same figure.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_valid, check_additivity=False)

    # The beeswarm scatters overlapping points at random, so the same attributions plotted twice give two different
    # figures unless it is seeded. `rng` takes the seed without touching numpy's global state.
    shap.summary_plot(
        shap_values, X_valid, feature_names=X_valid.columns, show=False,
        rng=np.random.default_rng(random_state),
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_model_explanations(
    model,
    X_train,
    X_valid,
    y_valid,
    *,
    name: str,
    save_dir: Path,
    prefix: str = "",
    random_state: int = 0,
) -> None:
    """Produce the standard explanation set for one fitted model.

    Permutation importance and SHAP answer different questions — which features the model relies on for its score, and
    how each feature moves an individual prediction — so a model is worth looking at through both.

    Args:
        model: A fitted estimator with `predict`.
        X_train: Rows the model was fitted on, used as the SHAP background distribution.
        X_valid, y_valid: Held-out rows the explanations are computed over.
        name: Display name, used in the title and, lowercased, in the filenames. Figures are output, so they carry
            the same name the result tables do rather than the short key a config is written with.
        save_dir: Directory the figures are written to; created if missing.
        prefix: Prepended to both filenames, so several runs can share a directory.
        random_state: Seed for the permutation shuffles.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    slug = name.lower().replace(" ", "_")

    plot_permutation_importance(
        model, X_valid, y_valid,
        title=f"{name} Permutation Importance",
        save_path=save_dir / f"{prefix}perm_importance_{slug}.png",
        random_state=random_state,
    )
    plot_shap_summary(
        model, X_train, X_valid,
        save_path=save_dir / f"{prefix}shap_summary_{slug}.png", random_state=random_state,
    )
