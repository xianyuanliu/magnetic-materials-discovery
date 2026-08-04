from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
repo_root = Path(__file__).resolve().parents[2]

novamag_csv = repo_root / "results" / "ood" / "table4_combined_comparison.csv"
mp_csv = repo_root / "results" / "mp_ood" / "table4_combined_comparison.csv"

output_png = repo_root / "figures" / "paper" / "figure1_ood_summary_mse.png"
output_png.parent.mkdir(parents=True, exist_ok=True)
# -----------------------------
# Settings
# -----------------------------
scenario_order = ["LOEO", "LOPO", "LOGO", "LOCO (k=10)"]
raw_scenario_map = {
    "LOEO": "LOEO",
    "LOPO": "LOPO",
    "LOGO": "LOGO",
    "LOCO(k=10)": "LOCO (k=10)",
}

model_order = ["Random Forest", "XGBoost"]

rf_color = "#1f77b4"
xgb_color = "#d62728"
width = 0.35

# -----------------------------
# Helper functions
# -----------------------------
def load_plot_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    plot_df = df[["scenario", "model", "MSE"]].copy()
    plot_df["scenario"] = plot_df["scenario"].replace(raw_scenario_map)

    plot_df = plot_df[plot_df["scenario"].isin(scenario_order)]
    plot_df = plot_df[plot_df["model"].isin(model_order)]

    plot_df["MSE_mean"] = plot_df["MSE"].str.split("±").str[0].str.strip().astype(float)
    plot_df["MSE_std"] = plot_df["MSE"].str.split("±").str[1].str.strip().astype(float)

    plot_df["scenario"] = pd.Categorical(
        plot_df["scenario"],
        categories=scenario_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["scenario", "model"])

    return plot_df


def get_model_rows(plot_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return (
        plot_df[plot_df["model"] == model_name]
        .set_index("scenario")
        .reindex(scenario_order)
    )


def plot_panel(ax, plot_df: pd.DataFrame, panel_title: str) -> None:
    x = range(len(scenario_order))

    rf = get_model_rows(plot_df, "Random Forest")
    xgb = get_model_rows(plot_df, "XGBoost")

    ax.bar(
        [i - width / 2 for i in x],
        rf["MSE_mean"],
        width=width,
        yerr=rf["MSE_std"],
        capsize=3,
        error_kw={"lw": 1.2, "alpha": 0.8, "capthick": 1.2},
        label="Random Forest",
        color=rf_color,
        edgecolor="none",
    )

    ax.bar(
        [i + width / 2 for i in x],
        xgb["MSE_mean"],
        width=width,
        yerr=xgb["MSE_std"],
        capsize=3,
        error_kw={"lw": 1.2, "alpha": 0.8, "capthick": 1.2},
        label="XGBoost",
        color=xgb_color,
        edgecolor="none",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(scenario_order, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(panel_title, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# -----------------------------
# Load data
# -----------------------------
novamag_df = load_plot_df(novamag_csv)
mp_df = load_plot_df(mp_csv)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

plot_panel(axes[0], novamag_df, "(a) Novamag dataset")
plot_panel(axes[1], mp_df, "(b) Materials Project dataset")

# Y-axis only on left panel
axes[0].set_ylabel("MSE (lower is better)", fontsize=12)
axes[1].set_ylabel("")
axes[1].tick_params(labelleft=False)

# Global x-label
fig.supxlabel("Out-of-distribution scenario", fontsize=12)

# Single global legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=2,
    frameon=False,
    fontsize=11,
)

plt.subplots_adjust(wspace=0.08)
plt.tight_layout(rect=[0, 0.05, 1, 0.92])
plt.savefig(output_png, dpi=600, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved figure to: {output_png}")