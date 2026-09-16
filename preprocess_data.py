"""Build the featurized CSVs that main.py reads, from the raw Novamag and Materials Project collections.

Run once per data refresh. Every experiment afterwards re-enters at loaddata/featurized_csv.py, which reads what this
script saves instead of re-parsing the raw collections.
"""

import pandas as pd

from loaddata.element_properties import load_element_properties
from loaddata.materials_project import load_materials_project
from loaddata.novamag import read_novamag_records
from loaddata.unified import select_unified_columns
from prepdata.modeling_table import build_modeling_table

NOVAMAG_RAW_DIR = "./data/novamag/Novamag_Data_Files/"
NOVAMAG_RECORDS_PATH = "./data/novamag/novamag-raw.csv"
NOVAMAG_OUTPUT_PATH = "./data/novamag-magnetism.csv"
MP_CSV_PATH = "./data/materials_project/mp-data.csv"
MP_OUTPUT_PATH = "./data/mp-magnetism.csv"


def _save_modeling_table(data: pd.DataFrame, pt: pd.DataFrame, mm: pd.DataFrame, output_path: str) -> None:
    """Featurize one dataset's unified frame and write the modeling table."""
    table, _ = build_modeling_table(data, pt, mm)
    print(f"  {len(table)} samples after featurization and filtering -> {output_path}")
    table.to_csv(output_path, index=True)


def main():
    """Build novamag-magnetism.csv and mp-magnetism.csv from the raw source data."""
    periodic_table, miedema_weight = load_element_properties()

    print("Processing Novamag dataset...")
    # Read once: the full records are saved for reference, then narrowed to the unified (formula, target) frame.
    novamag_records = read_novamag_records(NOVAMAG_RAW_DIR)
    novamag_records.to_csv(NOVAMAG_RECORDS_PATH, index=True)
    novamag = select_unified_columns(novamag_records)
    _save_modeling_table(novamag, periodic_table, miedema_weight, NOVAMAG_OUTPUT_PATH)

    print("\nProcessing Materials Project dataset...")
    _save_modeling_table(load_materials_project(MP_CSV_PATH), periodic_table, miedema_weight, MP_OUTPUT_PATH)


if __name__ == "__main__":
    main()
