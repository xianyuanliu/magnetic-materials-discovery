"""Build the raw Novamag/Materials Project magnetism CSVs used by main.py."""

from loaddata.raw_loaders import load_novamag_raw_data, load_mp_raw_data
from prepdata.build_features import process_data


def main():
    """Build novamag-magnetism.csv and mp-magnetism.csv from the raw source data."""
    pt_path = "./data/Periodic-table/periodic_table.xlsx"
    mm_path = "./data/Miedema-model/Miedema-model-reduced.xlsx"

    ### 1. Process Novamag dataset
    print("Processing Novamag dataset...")
    novamag_dir = "./data/novamag/Novamag_Data_Files/"
    save_novamag_raw_data = True

    novamag_raw = load_novamag_raw_data(novamag_dir)

    # Save raw Novamag data for reference
    if save_novamag_raw_data:
        novamag_raw.to_csv("./data/novamag/novamag-raw.csv", index=True)

    novamag_raw = novamag_raw[["chemical formula", "saturation magnetization"]]
    novamag_mag = process_data(novamag_raw, pt_path, mm_path)

    novamag_mag.to_csv("./data/novamag-magnetism.csv", index=True)


    # Process Materials Project dataset
    print("\nProcessing Materials Project dataset...")
    mp_csv_path = "./data/materials_project/mp-data.csv"
    mp_raw = load_mp_raw_data(mp_csv_path)
    mp_mag = process_data(mp_raw, pt_path, mm_path)

    mp_mag.to_csv("./data/mp-magnetism.csv", index=True)


if __name__ == "__main__":
    main()
