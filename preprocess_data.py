import numpy as np
import pandas as pd
import alloys


# ====== Shared elemental data loader ======

def load_elemental_data(
    pt_path = "./data/Periodic-table/periodic_table.xlsx",
    mm_path = "./data/Miedema-model/Miedema-model-reduced.xlsx",
):
    """Load the periodic table and the Miedema model."""
    periodic_table = alloys.import_periodic_table(pt_path)
    miedema_weight = alloys.import_miedema_weight(mm_path)
    return periodic_table, miedema_weight


# ====== Raw data loaders with basic cleaning ======

def load_novamag_raw_data(novamag_dir):
    """
    Load and prune raw Novamag data to chemical formula and saturation magnetization.
    """
    data = alloys.importNovamag(novamag_dir)

    # Normalize literal 'none' entries
    data = data.replace({None: np.nan, "none": np.nan, "None": np.nan})
    data = data.infer_objects(copy=False)

    # # Find columns with missing values
    # na_cols = [col for col in data.columns if data[col].isna().any()]
    # print(
    #     "The number of features with at least one NaN value is "
    #     f"{len(na_cols)}"
    # )

    # # Drop columns with more than 10 NaN values
    # dropped_cols = []
    # for col in na_cols:
    #     count = data[col].isna().sum()
    #     print(f"Column '{col}' has {count} nan values")
    #     if count > 10:
    #         print(f"Dropping column '{col}'")
    #         dropped_cols.append(col)
    #         data = data.drop(col, axis=1).copy()
    # print(f"Number of dropped features is: {len(dropped_cols)}")

    # Drop all columns except target and chemical formula
    return data


def load_mp_raw_data(csv_path):
    """
    Read the Materials Project CSV export, convert magnetization units, and filter invalid entries.
    """
    data = pd.read_csv(csv_path)
    data = data.rename(columns={"composition": "chemical formula"})
    print(f"The total number of imported features is {len(data.columns)}")

    # Convert target to saturation magnetization, μB/Å^3 → A/m → μ_0 M（T）
    print("Converting total magnetization to saturation magnetization...")
    data["total_magnetization_normalized_vol"] = pd.to_numeric(data["total_magnetization_normalized_vol"], errors='coerce')
    mu_B = 9.274e-24  # A·m^2
    angstrom3_to_m3 = 1e-30 # m^3
    mu_0 = 4 * np.pi * 1e-7  # T·m/A
    factor = (mu_B / angstrom3_to_m3) * mu_0  # ~= 11.65 T per (μB/Å^3)
    data["saturation magnetization"] = data["total_magnetization_normalized_vol"] * factor

    # Keep only magnetic systems
    initial_rows = len(data)
    data = data[data["is_magnetic"]]
    print(f"Filtered to magnetic systems: {len(data)} of {initial_rows} rows remaining")

    # Remove entries containing rare-earth elements
    rare_earth_elements = [
        "La",
        "Sc",
        "Dy",
        "Sm",
        "Lu",
        "Er",
        "Y",
        "Pr",
        "Nd",
        "Gd",
        "Tm",
        "Pm",
        "Ce",
        "Tb",
        "Eu",
        "Ho",
        "Yb",
    ]
    before_rare_earth = len(data)
    for element in rare_earth_elements:
        data = data[~data["chemical formula"].str.contains(element)]
    print(f"Removed rare-earth entries: {before_rare_earth - len(data)} rows dropped")

    # Remove uncommon or commercially unavailable elements
    non_commercial_elements = [
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
    ]
    before_non_commercial = len(data)
    for element in non_commercial_elements:
        data = data[~data["chemical formula"].str.contains(element)]
    print(f"Removed non-commercial entries: {before_non_commercial - len(data)} rows dropped")

    data = data[["chemical formula", "saturation magnetization"]]
    print(f"Final MP raw dataset size: {data.shape[0]} material samples x {data.shape[1]} features")
    return data


# ====== Shared alloy feature builder ======

def build_features(raw_data: pd.DataFrame, pt: pd.DataFrame, mm: pd.DataFrame):
    """
    Engineer alloy features shared across Novamag and Materials Project data.
    """
    data = raw_data.copy()

    # compoundradix: number of species in the compound (binary, ternary, etc.)
    data["compoundradix"] = alloys.get_CompoundRadix(data)

    # stoichiometric array
    stoich_array = alloys.get_stoich_array(data, pt)

    # Compute element-weighted features from the stoichiometric array
    data["stoicentw"] = alloys.get_StoicEntw(stoich_array)           # mixing entropy
    data["Zw"] = alloys.get_Zw(pt, stoich_array)                     # atomic weight
    data["periodw"] = alloys.get_Periodw(pt, stoich_array)           # period
    data["groupw"] = alloys.get_Groupw(pt, stoich_array)             # group
    data["meltingTw"] = alloys.get_MeltingTw(pt, stoich_array)       # melting point
    data["miedemaH"] = alloys.get_Miedemaw(mm, stoich_array)         # Miedema mixing enthalpy
    data["valencew"] = alloys.get_Valencew(pt, stoich_array)         # valence
    data["electronegw"] = alloys.get_Electronegw(pt, stoich_array)   # electronegativity

    feature_columns = [
        "compoundradix",
        "stoicentw",
        "Zw",
        "periodw",
        "groupw",
        "meltingTw",
        "miedemaH",
        "valencew",
        "electronegw",
    ]

    # Remove rows with missing target
    data.dropna(
        axis=0,
        subset=["saturation magnetization"] + feature_columns,
        inplace=True,
    )

    # Drop alloys which are below magnetic cutoff (determined through prior model optimization)
    data.drop(data[data["saturation magnetization"] < 0.18].index, axis=0, inplace=True)

    # # Round the saturation magnetization to two decimal places
    # data["saturation magnetization"] = pd.to_numeric(data["saturation magnetization"]).round(decimals=2)

    # Collapse duplicate chemical formulas by taking the median feature values
    data = data.groupby(by="chemical formula").median()
   
    return data, feature_columns


# ====== Shared feature engineering ======

def process_data(raw_data, pt_path, mm_path):
    """
    Load elemental data and engineer alloy features for a raw magnetism dataset.
    """
    periodic_table, miedema_weight = load_elemental_data(pt_path, mm_path)
    data, _ = build_features(raw_data, periodic_table, miedema_weight)
    print(f"The total number of samples after cleaning is {len(data)}")
    
    return data


# ===== Main execution ======

def main():
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