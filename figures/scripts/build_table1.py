from pathlib import Path
import pandas as pd


def build_indistribution_table(input_csv: Path, output_csv: Path, dataset_name: str) -> None:
    df = pd.read_csv(input_csv)

    print(f"\nColumns in {input_csv.name}:")
    print(df.columns.tolist())

    # -------------------------------
    # Clean & map model names
    # -------------------------------
    model_map = {
        "rf": "Random Forest",
        "xgb": "XGBoost",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "lr": "Linear Regression",
        "linear": "Linear Regression",
        "Random Forest": "Random Forest",
        "XGBoost": "XGBoost",
        "Ridge": "Ridge",
    }

    df["model"] = df["model"].map(lambda x: model_map.get(x, x))

    # Remove rows with missing model names
    df = df.dropna(subset=["model"])

    # -------------------------------
    # Pivot (long → wide)
    # -------------------------------
    pivot_mean = df.pivot(index="model", columns="metric", values="avg_runs_mean")
    pivot_std = df.pivot(index="model", columns="metric", values="avg_runs_std")

    # -------------------------------
    # Format mean ± std
    # -------------------------------
    def fmt(m, s, decimals=4):
        return f"{m:.{decimals}f} ± {s:.{decimals}f}"

    table = pd.DataFrame({
        "Model": pivot_mean.index,
        "MSE ↓": [fmt(pivot_mean.loc[m, "mse"], pivot_std.loc[m, "mse"]) for m in pivot_mean.index],
        "MAE ↓": [fmt(pivot_mean.loc[m, "mae"], pivot_std.loc[m, "mae"]) for m in pivot_mean.index],
        "MRE ↓": [fmt(pivot_mean.loc[m, "mre"], pivot_std.loc[m, "mre"]) for m in pivot_mean.index],
        "R² ↑": [fmt(pivot_mean.loc[m, "r2"], pivot_std.loc[m, "r2"]) for m in pivot_mean.index],
    })

    # -------------------------------
    # Order models (RF & XGB first)
    # -------------------------------
    preferred_order = ["Random Forest", "XGBoost"]

    table["_order"] = table["Model"].apply(
        lambda x: preferred_order.index(x) if x in preferred_order else len(preferred_order)
    )

    table = table.sort_values(["_order", "Model"]) \
                 .drop(columns="_order") \
                 .reset_index(drop=True)

    # -------------------------------
    # Save
    # -------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)

    print(f"\n{dataset_name} Table 1 preview:\n")
    print(table.to_string(index=False))
    print(f"\nSaved to: {output_csv}")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]

    novamag_input = repo_root / "results" / "novamag_cv10_runs.csv"
    mp_input = repo_root / "results" / "mp_cv10_runs.csv"

    novamag_output = repo_root / "figures" / "paper" / "tables" / "table1a_novamag_indistribution.csv"
    mp_output = repo_root / "figures" / "paper" / "tables" / "table1b_mp_indistribution.csv"

    if novamag_input.exists():
        build_indistribution_table(novamag_input, novamag_output, "Novamag")
    else:
        print(f"Novamag file not found: {novamag_input}")

    if mp_input.exists():
        build_indistribution_table(mp_input, mp_output, "Materials Project")
    else:
        print(f"MP file not found: {mp_input}")
