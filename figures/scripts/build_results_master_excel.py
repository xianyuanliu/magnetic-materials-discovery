from pathlib import Path
import pandas as pd

# =========================
# Paths
# =========================
# Script location: magnetic-materials-discovery/figures/scripts/build_results_master_excel.py
# parents[2] = magnetic-materials-discovery
repo_root = Path(__file__).resolve().parents[2]

results_dir = repo_root / "results"
output_excel = results_dir / "magnetic_materials_results_master.xlsx"

paper_tables_dir = repo_root / "figures" / "paper" / "tables"
paper_tables_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# =========================
# Settings
# =========================
scenario_order = ["LOEO", "LOPO", "LOGO", "LOCO(k=10)"]
model_order = ["Random Forest", "XGBoost"]


# =========================
# Helpers
# =========================
def load_csv(path: Path, dataset_name=None):
    if not path.exists():
        print(f"[WARNING] Missing file: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if dataset_name and "dataset" not in df.columns:
        df.insert(0, "dataset", dataset_name)

    return df


def make_fig_mse_source(ood_t4: pd.DataFrame) -> pd.DataFrame:
    if ood_t4.empty:
        return pd.DataFrame()

    df = ood_t4.copy()
    df = df[
        df["scenario"].isin(scenario_order)
        & df["model"].isin(model_order)
    ].copy()

    df["MSE_mean"] = df["MSE"].astype(str).str.split("±").str[0].str.strip()
    df["MSE_std"] = df["MSE"].astype(str).str.split("±").str[1].str.strip()

    return df


# =========================
# Load all data
# =========================

# ID CV baseline results
cv_novamag = load_csv(results_dir / "novamag_cv10_runs.csv", "novamag")
cv_mp = load_csv(results_dir / "mp_cv10_runs.csv", "mp")
cv_sig = load_csv(results_dir / "rf_vs_xgb_significance.csv")

# OOD Novamag
ood_nova_t1 = load_csv(results_dir / "ood" / "table1_splits_summary.csv", "novamag")
ood_nova_t2 = load_csv(results_dir / "ood" / "table2_metrics_by_model.csv", "novamag")
ood_nova_t3 = load_csv(results_dir / "ood" / "table3_rf_vs_xgb_significance.csv", "novamag")
ood_nova_t4 = load_csv(results_dir / "ood" / "table4_combined_comparison.csv", "novamag")

# OOD MP
ood_mp_t1 = load_csv(results_dir / "mp_ood" / "table1_splits_summary.csv", "mp")
ood_mp_t2 = load_csv(results_dir / "mp_ood" / "table2_metrics_by_model.csv", "mp")
ood_mp_t3 = load_csv(results_dir / "mp_ood" / "table3_rf_vs_xgb_significance.csv", "mp")
ood_mp_t4 = load_csv(results_dir / "mp_ood" / "table4_combined_comparison.csv", "mp")

# =========================
# Combined outputs
# =========================
cv_all = pd.concat([cv_novamag, cv_mp], ignore_index=True)

ood_t1 = pd.concat([ood_nova_t1, ood_mp_t1], ignore_index=True)
ood_t2 = pd.concat([ood_nova_t2, ood_mp_t2], ignore_index=True)
ood_t3 = pd.concat([ood_nova_t3, ood_mp_t3], ignore_index=True)
ood_t4 = pd.concat([ood_nova_t4, ood_mp_t4], ignore_index=True)

fig_mse = make_fig_mse_source(ood_t4)

# Paper Table 3: combined OOD performance
paper_table3 = ood_t4.copy()
if not paper_table3.empty:
    paper_table3 = paper_table3[
        paper_table3["scenario"].isin(scenario_order)
        & paper_table3["model"].isin(model_order)
    ].copy()
    paper_table3 = paper_table3.rename(
        columns={
            "dataset": "Dataset",
            "scenario": "Scenario",
            "model": "Model",
        }
    )

# Paper Table 4: combined significance summary
paper_table4 = ood_t3.copy()
if not paper_table4.empty:
    paper_table4 = paper_table4[
        paper_table4["scenario"].isin(scenario_order)
    ].copy()

    paper_table4 = (
        paper_table4
        .groupby(["dataset", "scenario", "metric"], observed=True)
        .agg(
            **{
                "Significant / Total": (
                    "significant",
                    lambda x: f"{int(x.sum())} / {len(x)}",
                ),
                "Median t-test p": ("t_pvalue", "median"),
                "Median Wilcoxon p": ("wilcoxon_pvalue", "median"),
            }
        )
        .reset_index()
        .rename(
            columns={
                "dataset": "Dataset",
                "scenario": "Scenario",
                "metric": "Metric",
            }
        )
    )

# =========================
# README / Notes
# =========================
readme = pd.DataFrame(
    {
        "Sheet": [
            "01_ID_CV_Novamag",
            "02_ID_CV_MP",
            "03_OOD_Novamag_Table1_Splits",
            "04_OOD_Novamag_Table2_Metrics",
            "05_OOD_Novamag_Table3_Significance",
            "06_OOD_Novamag_Table4_Combined",
            "07_OOD_MP_Table1_Splits",
            "08_OOD_MP_Table2_Metrics",
            "09_OOD_MP_Table3_Significance",
            "10_OOD_MP_Table4_Combined",
            "11_Fig4_Source_MSE",
            "12A_Paper_Table3_OOD",
            "12B_Paper_Table4_Signif",
            "13_Notes",
            "14_ID_CV_Significance",
        ],
        "Description": [
            "In-distribution 10-fold CV results for Novamag",
            "In-distribution 10-fold CV results for Materials Project",
            "Novamag OOD split summary",
            "Novamag OOD metrics by model",
            "Novamag RF vs XGB OOD significance",
            "Novamag OOD combined comparison",
            "Materials Project OOD split summary",
            "Materials Project OOD metrics by model",
            "Materials Project RF vs XGB OOD significance",
            "Materials Project OOD combined comparison",
            "Source data for Figure 4 MSE comparison",
            "Combined paper Table 3: OOD performance",
            "Combined paper Table 4: RF vs XGB significance",
            "Notes and interpretation",
            "In-distribution CV RF vs XGB significance",
        ],
    }
)

notes = pd.DataFrame(
    {
        "Item": [
            "ID CV",
            "OOD",
            "Important distinction",
            "Figure 4",
            "Paper Table 3",
            "Paper Table 4",
        ],
        "Note": [
            "Random 10-fold cross-validation within the same distribution.",
            "Structured held-out splits: LOEO, LOPO, LOGO, LOCO(k=10).",
            "CV results are ID baseline results, not OOD results.",
            "Uses MSE mean and standard deviation from combined OOD Table 4 outputs.",
            "Combines Novamag and Materials Project OOD performance.",
            "Combines Novamag and Materials Project RF vs XGB significance.",
        ],
    }
)

# =========================
# Write Excel
# =========================
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    readme.to_excel(writer, sheet_name="00_README", index=False)

    cv_novamag.to_excel(writer, sheet_name="01_ID_CV_Novamag", index=False)
    cv_mp.to_excel(writer, sheet_name="02_ID_CV_MP", index=False)

    ood_nova_t1.to_excel(writer, sheet_name="03_Nova_OOD_Splits", index=False)
    ood_nova_t2.to_excel(writer, sheet_name="04_Nova_OOD_Metrics", index=False)
    ood_nova_t3.to_excel(writer, sheet_name="05_Nova_OOD_Signif", index=False)
    ood_nova_t4.to_excel(writer, sheet_name="06_Nova_OOD_Combined", index=False)

    ood_mp_t1.to_excel(writer, sheet_name="07_MP_OOD_Splits", index=False)
    ood_mp_t2.to_excel(writer, sheet_name="08_MP_OOD_Metrics", index=False)
    ood_mp_t3.to_excel(writer, sheet_name="09_MP_OOD_Signif", index=False)
    ood_mp_t4.to_excel(writer, sheet_name="10_MP_OOD_Combined", index=False)

    fig_mse.to_excel(writer, sheet_name="11_Fig4_Source_MSE", index=False)

    paper_table3.to_excel(writer, sheet_name="12A_Paper_Table3_OOD", index=False)
    paper_table4.to_excel(writer, sheet_name="12B_Paper_Table4_Signif", index=False)

    notes.to_excel(writer, sheet_name="13_Notes", index=False)
    cv_sig.to_excel(writer, sheet_name="14_ID_CV_Significance", index=False)

# =========================
# Export paper-ready CSVs
# =========================
paper_table3.to_csv(
    paper_tables_dir / "table3_combined_ood_performance.csv",
    index=False,
)
paper_table4.to_csv(
    paper_tables_dir / "table4_combined_rf_xgb_significance.csv",
    index=False,
)
fig_mse.to_csv(
    paper_tables_dir / "figure4_source_mse.csv",
    index=False,
)

print(f"\n Master Excel file created at:\n{output_excel}")
print(f" Paper Table 3 CSV:\n{paper_tables_dir / 'table3_combined_ood_performance.csv'}")
print(f" Paper Table 4 CSV:\n{paper_tables_dir / 'table4_combined_rf_xgb_significance.csv'}")
print(f" Figure 4 source CSV:\n{paper_tables_dir / 'figure4_source_mse.csv'}")
