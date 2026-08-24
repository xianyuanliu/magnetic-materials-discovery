from pathlib import Path
import pandas as pd


def format_p(x: float) -> str:
    if pd.isna(x):
        return "—"
    if x < 1e-4:
        return "< 1e-4"
    return f"{x:.4f}"


def build_table3_compact(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)

    print("\nColumns:")
    print(df.columns.tolist())

    # Normalize metric labels
    df["metric"] = df["metric"].astype(str).str.upper()

    # Aggregate by scenario + metric
    grouped = (
        df.groupby(["scenario", "metric"], dropna=False)
        .agg(
            significant_count=("significant", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            total_comparisons=("significant", "size"),
            median_t_pvalue=("t_pvalue", "median"),
            median_wilcoxon_pvalue=("wilcoxon_pvalue", "median"),
        )
        .reset_index()
    )

    grouped["Significant / Total"] = grouped.apply(
        lambda r: f"{r['significant_count']} / {r['total_comparisons']}", axis=1
    )
    grouped["Median t-test p"] = grouped["median_t_pvalue"].apply(format_p)
    grouped["Median Wilcoxon p"] = grouped["median_wilcoxon_pvalue"].apply(format_p)

    table = grouped[[
        "scenario",
        "metric",
        "Significant / Total",
        "Median t-test p",
        "Median Wilcoxon p",
    ]].copy()

    table.columns = [
        "Scenario",
        "Metric",
        "Significant / Total",
        "Median t-test p",
        "Median Wilcoxon p",
    ]

    scenario_order = ["LOEO", "LOPO", "LOGO", "LOCO(k=10)", "SparseX", "SparseY"]
    metric_order = ["MSE", "MAE"]

    table["Scenario"] = pd.Categorical(table["Scenario"], categories=scenario_order, ordered=True)
    table["Metric"] = pd.Categorical(table["Metric"], categories=metric_order, ordered=True)

    table = table.sort_values(["Scenario", "Metric"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)

    print("\nCompact Table 3 preview:\n")
    print(table.to_string(index=False))
    print(f"\nSaved to: {output_csv}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]

    input_csv = repo_root / "results" / "ood" / "table3_rf_vs_xgb_significance.csv"
    output_csv = repo_root / "figures" / "paper" / "tables" / "table3_significance_compact.csv"

    if input_csv.exists():
        build_table3_compact(input_csv, output_csv)
    else:
        print(f"File not found: {input_csv}")
