"""
uq_minimum_test.py
==================
Minimum test for Uncertainty Quantification (UQ) in magnetic materials
prediction — demonstrating the confidence-error paradox under OOD conditions.

Usage:
    python uq_minimum_test.py --data data/novamag-magnetism.csv

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

TARGET_COL   = "saturation magnetization"
FORMULA_COL  = "chemical formula"

DESCRIPTOR_COLS = [
    "stoicentw",
    "valencew",
    "periodw",
    "groupw",
    "meltingTw",
    "miedemaH",
    "electronegw",
    "Zw",
    "compoundradix",
]

RF_PARAMS = dict(n_estimators=200, random_state=42, n_jobs=-1)
N_LOCO_CLUSTERS = 10
CV_FOLDS = 5
OUTPUT_DIR = Path("figures/uq")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in DESCRIPTOR_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available: {df.columns.tolist()}"
        )
    df = df.dropna(subset=DESCRIPTOR_COLS + [TARGET_COL]).reset_index(drop=True)
    print(f"Loaded {len(df)} samples after dropping NaN rows.")
    return df


def get_rf_uncertainty(rf_model, X: np.ndarray) -> np.ndarray:
    """
    Estimate uncertainty as the standard deviation of predictions
    across all trees in the Random Forest.
    This is the simplest and most interpretable UQ approach for RF.
    """
    tree_preds = np.array([tree.predict(X) for tree in rf_model.estimators_])
    return tree_preds.std(axis=0)


def compute_ece(errors: np.ndarray,
                uncertainties: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Proxy ECE: bins samples by uncertainty level and checks whether
    higher uncertainty corresponds to higher actual error.
    A well-calibrated model should show a monotonic relationship.
    Returns the mean absolute deviation from perfect calibration.
    """
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return np.nan

    bin_mean_unc, bin_mean_err = [], []
    for i in range(len(bin_edges) - 1):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_mean_unc.append(uncertainties[mask].mean())
        bin_mean_err.append(np.abs(errors[mask]).mean())

    if len(bin_mean_unc) < 2:
        return np.nan

    # Normalise both to [0,1] for comparison
    unc_norm = np.array(bin_mean_unc) / (max(bin_mean_unc) + 1e-9)
    err_norm = np.array(bin_mean_err) / (max(bin_mean_err) + 1e-9)
    return float(np.mean(np.abs(unc_norm - err_norm)))


# ── Evaluation routines ───────────────────────────────────────────────────────

def evaluate_id(X: np.ndarray, y: np.ndarray) -> dict:
    """Standard K-Fold cross-validation (in-distribution baseline)."""
    print("\n── In-Distribution (5-Fold CV) ──")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    records = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_tr_s, y_tr)

        y_pred = rf.predict(X_te_s)
        unc    = get_rf_uncertainty(rf, X_te_s)
        errors = y_pred - y_te

        records.append({
            "split_type": "ID",
            "fold": fold,
            "y_true": y_te,
            "y_pred": y_pred,
            "uncertainty": unc,
            "error": errors,
            "mae":  mean_absolute_error(y_te, y_pred),
            "mse":  mean_squared_error(y_te, y_pred),
            "r2":   r2_score(y_te, y_pred),
            "ece":  compute_ece(errors, unc),
        })
        print(f"  Fold {fold+1}: MAE={records[-1]['mae']:.4f}  "
              f"R²={records[-1]['r2']:.4f}  "
              f"Mean-Unc={unc.mean():.4f}  ECE={records[-1]['ece']:.4f}")

    return records


def evaluate_loco(X: np.ndarray,
                  y: np.ndarray,
                  X_scaled_full: np.ndarray) -> list:
    """
    Leave-One-Cluster-Out (LOCO) OOD evaluation.
    Clusters are formed in descriptor space — held-out clusters
    represent genuine distributional shift (the hardest OOD scenario).
    """
    print(f"\n── OOD: LOCO (k={N_LOCO_CLUSTERS} clusters) ──")
    km = KMeans(n_clusters=N_LOCO_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled_full)

    records = []
    for c in range(N_LOCO_CLUSTERS):
        test_mask  = cluster_labels == c
        train_mask = ~test_mask

        if test_mask.sum() < 3:
            continue

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_tr_s, y_tr)

        y_pred = rf.predict(X_te_s)
        unc    = get_rf_uncertainty(rf, X_te_s)
        errors = y_pred - y_te

        r2 = r2_score(y_te, y_pred) if len(y_te) > 1 else np.nan

        records.append({
            "split_type": "OOD-LOCO",
            "fold": c,
            "y_true": y_te,
            "y_pred": y_pred,
            "uncertainty": unc,
            "error": errors,
            "mae":  mean_absolute_error(y_te, y_pred),
            "mse":  mean_squared_error(y_te, y_pred),
            "r2":   r2,
            "ece":  compute_ece(errors, unc),
        })
        print(f"  Cluster {c:2d}: MAE={records[-1]['mae']:.4f}  "
              f"R²={r2:.4f}  "
              f"Mean-Unc={unc.mean():.4f}  ECE={records[-1]['ece']:.4f}")

    return records


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_confidence_error_paradox(id_records: list,
                                  ood_records: list,
                                  save_dir: Path) -> None:
    """
    The key figure: scatter plot of uncertainty vs absolute error.
    The paradox appears when OOD points (red) cluster in the
    LOW-uncertainty / HIGH-error quadrant — the model is confident
    but wrong.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all samples
    def flatten(records, label):
        rows = []
        for r in records:
            for unc, err in zip(r["uncertainty"], r["error"]):
                rows.append({
                    "uncertainty": unc,
                    "abs_error":   abs(err),
                    "split":       label,
                })
        return rows

    df_plot = pd.DataFrame(
        flatten(id_records,  "In-Distribution (ID)") +
        flatten(ood_records, "OOD — LOCO")
    )

    # ── Figure 1: Scatter — Uncertainty vs Absolute Error ────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    palette = {"In-Distribution (ID)": "#4C72B0", "OOD — LOCO": "#DD4949"}
    for label, grp in df_plot.groupby("split"):
        ax.scatter(
            grp["uncertainty"], grp["abs_error"],
            alpha=0.45, s=18,
            color=palette[label], label=label, edgecolors="none"
        )

    # Paradox region annotation
    x_thresh = df_plot["uncertainty"].quantile(0.35)
    y_thresh = df_plot["abs_error"].quantile(0.65)
    ax.axvline(x_thresh, color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(y_thresh, color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.text(
        df_plot["uncertainty"].min() * 1.05,
        df_plot["abs_error"].max() * 0.92,
        "⚠ Paradox zone\n(low confidence, high error)",
        fontsize=8.5, color="grey"
    )

    ax.set_xlabel("Prediction Uncertainty (RF std across trees)", fontsize=12)
    ax.set_ylabel("Absolute Prediction Error  |ŷ − y|", fontsize=12)
    ax.set_title(
        "Confidence–Error Paradox under OOD Conditions\n"
        "Novamag Dataset — Random Forest",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11, framealpha=0.9)
    sns.despine(ax=ax)
    fig.tight_layout()
    out = save_dir / "uq_paradox_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close(fig)

    # ── Figure 2: Calibration curves — ID vs OOD ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, (label, grp) in zip(axes, df_plot.groupby("split")):
        n_bins = 8
        bin_edges = np.percentile(grp["uncertainty"], np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        bin_unc, bin_err, bin_counts = [], [], []

        for i in range(len(bin_edges) - 1):
            mask = (grp["uncertainty"] >= bin_edges[i]) & \
                   (grp["uncertainty"] < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_unc.append(grp.loc[mask, "uncertainty"].mean())
            bin_err.append(grp.loc[mask, "abs_error"].mean())
            bin_counts.append(mask.sum())

        color = palette[label]
        ax.bar(range(len(bin_unc)), bin_err, color=color, alpha=0.7,
               label="Mean |error|")
        ax2 = ax.twinx()
        ax2.plot(range(len(bin_unc)), bin_unc, "o--", color="black",
                 lw=1.5, ms=5, label="Mean uncertainty")

        ax.set_xticks(range(len(bin_unc)))
        ax.set_xticklabels(
            [f"Bin {i+1}" for i in range(len(bin_unc))],
            rotation=30, fontsize=8
        )
        ax.set_xlabel("Uncertainty bin (low → high)", fontsize=10)
        ax.set_ylabel("Mean absolute error", fontsize=10, color=color)
        ax2.set_ylabel("Mean uncertainty (std)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
        sns.despine(ax=ax)

    fig.suptitle(
        "Calibration: Does Higher Uncertainty → Higher Error?\n"
        "Well-calibrated = yes (bars and line move together)",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out = save_dir / "uq_calibration_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── Figure 3: Summary bar — Mean uncertainty ID vs OOD ───────────────
    id_unc  = np.concatenate([r["uncertainty"] for r in id_records])
    ood_unc = np.concatenate([r["uncertainty"] for r in ood_records])
    id_err  = np.concatenate([np.abs(r["error"]) for r in id_records])
    ood_err = np.concatenate([np.abs(r["error"]) for r in ood_records])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, vals_id, vals_ood, ylabel in zip(
        axes,
        [id_unc,  id_err],
        [ood_unc, ood_err],
        ["Mean Uncertainty (RF std)", "Mean Absolute Error"]
    ):
        ax.bar(["ID", "OOD-LOCO"],
               [vals_id.mean(), vals_ood.mean()],
               color=[palette["In-Distribution (ID)"], palette["OOD — LOCO"]],
               alpha=0.85,
               yerr=[vals_id.std() / np.sqrt(len(vals_id)),
                     vals_ood.std() / np.sqrt(len(vals_ood))],
               capsize=6)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        sns.despine(ax=ax)

    fig.suptitle(
        "Uncertainty vs Error: ID vs OOD-LOCO\n"
        "The paradox: OOD error rises more than uncertainty",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out = save_dir / "uq_summary_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def print_summary_table(id_records: list, ood_records: list) -> None:
    rows = []
    for label, records in [("ID (5-Fold CV)", id_records),
                            ("OOD — LOCO",    ood_records)]:
        maes  = [r["mae"]  for r in records]
        r2s   = [r["r2"]   for r in records if not np.isnan(r["r2"])]
        eces  = [r["ece"]  for r in records if not np.isnan(r["ece"])]
        uncs  = np.concatenate([r["uncertainty"] for r in records])
        rows.append({
            "Split":        label,
            "MAE (mean)":   f"{np.mean(maes):.4f} ± {np.std(maes):.4f}",
            "R² (mean)":    f"{np.mean(r2s):.4f}"  if r2s  else "—",
            "ECE (proxy)":  f"{np.mean(eces):.4f}" if eces else "—",
            "Mean Unc":     f"{uncs.mean():.4f} ± {uncs.std():.4f}",
        })

    df = pd.DataFrame(rows).set_index("Split")
    print("\n" + "═" * 70)
    print("  UQ SUMMARY TABLE")
    print("═" * 70)
    print(df.to_string())
    print("═" * 70)
    print("\nKey finding to report:")
    id_unc  = np.concatenate([r["uncertainty"] for r in id_records]).mean()
    ood_unc = np.concatenate([r["uncertainty"] for r in ood_records]).mean()
    id_mae  = np.mean([r["mae"] for r in id_records])
    ood_mae = np.mean([r["mae"] for r in ood_records])
    unc_ratio = ood_unc / id_unc
    err_ratio = ood_mae / id_mae
    print(f"  Error increases {err_ratio:.1f}× from ID → OOD")
    print(f"  Uncertainty increases only {unc_ratio:.1f}× from ID → OOD")
    if err_ratio > unc_ratio:
        print("  ⚠ PARADOX CONFIRMED: error grows faster than uncertainty")
        print("    → Model is overconfident under OOD conditions")
    else:
        print("  ✓ Model uncertainty tracks error increase (well-calibrated)")


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_comparative_paradox(datasets: dict, save_dir: Path) -> None:
    """
    Side-by-side comparison of paradox across multiple datasets.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    palette = {"In-Distribution (ID)": "#4C72B0", "OOD — LOCO": "#DD4949"}
    
    for ax, (dataset_name, records) in zip(axes, datasets.items()):
        id_records, ood_records = records
        
        # Flatten all samples
        def flatten(recs, label):
            rows = []
            for r in recs:
                for unc, err in zip(r["uncertainty"], r["error"]):
                    rows.append({
                        "uncertainty": unc,
                        "abs_error": abs(err),
                        "split": label,
                    })
            return rows
        
        df_plot = pd.DataFrame(
            flatten(id_records, "In-Distribution (ID)") +
            flatten(ood_records, "OOD — LOCO")
        )
        
        for label, grp in df_plot.groupby("split"):
            ax.scatter(
                grp["uncertainty"], grp["abs_error"],
                alpha=0.45, s=18,
                color=palette[label], label=label, edgecolors="none"
            )
        
        ax.set_xlabel("Prediction Uncertainty (RF std)", fontsize=11)
        ax.set_ylabel("Absolute Error |ŷ − y|", fontsize=11)
        ax.set_title(f"{dataset_name} Dataset", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        sns.despine(ax=ax)
    
    fig.suptitle(
        "Confidence–Error Paradox across Datasets\n"
        "ID vs OOD-LOCO — Random Forest",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    out = save_dir / "uq_paradox_comparative.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparative: {out}")
    plt.close(fig)


def plot_comparative_summary(datasets: dict, save_dir: Path) -> None:
    """
    Comparative summary bars across datasets.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors_id  = ["#4C72B0", "#5A8FD9"]
    colors_ood = ["#DD4949", "#E67070"]
    
    for i, metric in enumerate(["Mean Uncertainty", "Mean Absolute Error"]):
        ax = axes[i]
        x_pos = 0
        width = 0.35
        
        for j, (dataset_name, (id_records, ood_records)) in enumerate(datasets.items()):
            id_vals   = np.concatenate([r["uncertainty"] if metric == "Mean Uncertainty" 
                                        else np.abs(r["error"]) for r in id_records])
            ood_vals  = np.concatenate([r["uncertainty"] if metric == "Mean Uncertainty" 
                                        else np.abs(r["error"]) for r in ood_records])
            
            ax.bar(x_pos, id_vals.mean(), width, 
                   color=colors_id[j], alpha=0.85, label=f"{dataset_name} ID",
                   yerr=id_vals.std() / np.sqrt(len(id_vals)), capsize=4)
            ax.bar(x_pos + width, ood_vals.mean(), width,
                   color=colors_ood[j], alpha=0.85, label=f"{dataset_name} OOD",
                   yerr=ood_vals.std() / np.sqrt(len(ood_vals)), capsize=4)
            
            x_pos += 2 * width + 0.2
        
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.legend(fontsize=8, framealpha=0.9)
        sns.despine(ax=ax)
    
    fig.suptitle("Uncertainty and Error Comparison Across Datasets",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = save_dir / "uq_summary_comparative.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved comparative summary: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Minimum UQ test for magnetic materials prediction"
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=["data/novamag-magnetism.csv"],
        help="Paths to dataset CSVs (can specify multiple)"
    )
    args = parser.parse_args()

    datasets_results = {}
    
    for data_path in args.data:
        # Load
        df = load_data(data_path)
        dataset_name = Path(data_path).stem.replace("-magnetism", "").upper()
        
        X = df[DESCRIPTOR_COLS].values
        y = df[TARGET_COL].values

        # Pre-scale for clustering only (LOCO needs consistent space)
        scaler_full = StandardScaler()
        X_scaled_full = scaler_full.fit_transform(X)

        # Run evaluations
        print(f"\n{'='*70}")
        print(f"  {dataset_name} DATASET")
        print(f"{'='*70}")
        
        id_records  = evaluate_id(X, y)
        ood_records = evaluate_loco(X, y, X_scaled_full)

        # Print summary
        print_summary_table(id_records, ood_records)

        # Plot individual
        plot_confidence_error_paradox(id_records, ood_records, 
                                     OUTPUT_DIR / dataset_name.lower())
        
        datasets_results[dataset_name] = (id_records, ood_records)
    
    # Comparative plots if multiple datasets
    if len(datasets_results) > 1:
        print(f"\n{'='*70}")
        print("  COMPARATIVE ANALYSIS")
        print(f"{'='*70}")
        plot_comparative_paradox(datasets_results, OUTPUT_DIR)
        plot_comparative_summary(datasets_results, OUTPUT_DIR)

    print(f"\nAll figures saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
