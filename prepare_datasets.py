"""Prepare feature CSVs once; main.py reuses them for training and evaluation.

Run again when raw data, features or selection settings change. Dataset readers handle source formats; prepdata selects
magnetic records before aggregating normalized compositions. This script orchestrates those steps and saves the output.
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from loaddata.tabular_access import load_element_properties
from loaddata.materials_project import DEFAULT_CSV_PATH, load_materials_project
from loaddata.novamag import DEFAULT_ROOT_DIR, load_novamag
from prepdata.modeling_table import NON_COMMERCIAL_ELEMENTS, RARE_EARTH_ELEMENTS, build_modeling_table


def parse_args() -> argparse.Namespace:
    """Expose data sources, output location and experiment-specific selection settings."""
    parser = argparse.ArgumentParser(description="Prepare composition features from local magnetic-material datasets")
    parser.add_argument("--dataset", choices=("all", "novamag", "mp"), default="all")
    parser.add_argument("--novamag-dir", default=DEFAULT_ROOT_DIR, help="Directory containing Novamag JSON files")
    parser.add_argument("--mp-csv", default=DEFAULT_CSV_PATH, help="Materials Project CSV export")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--min-target", type=float, default=0.18, help="Inclusive record threshold in tesla, before median",
    )
    parser.add_argument(
        "--exclude-elements", nargs="*", default=None, metavar="ELEMENT",
        help="Override element exclusions for selected datasets; no values disables exclusions. "
             "Default: original rare-earth/actinide exclusions for MP, none for Novamag.",
    )
    parser.add_argument(
        "--mp-include-nonmagnetic", action="store_true",
        help="Disable MP's is_magnetic label filter; the target threshold still applies",
    )
    return parser.parse_args()


def _save_modeling_table(
    data: pd.DataFrame, pt: pd.DataFrame, mm: pd.DataFrame, output_path: Path,
    *, min_target: float, excluded_elements: Optional[Iterable[str]], magnetic_only: bool = False,
) -> None:
    """Select records, aggregate equivalent compositions and save the feature table."""
    table, _ = build_modeling_table(
        data, pt, mm, min_target=min_target, excluded_elements=excluded_elements, magnetic_only=magnetic_only,
    )
    print(f"  {len(data)} loaded records -> {len(table)} compositions -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=True)


def main() -> None:
    """Generate prepared data independently of model training."""
    args = parse_args()
    periodic_table, miedema_weight = load_element_properties()

    if args.dataset in ("all", "novamag"):
        print("Processing Novamag dataset...")
        novamag = load_novamag(args.novamag_dir)
        records_path = args.output_dir / "novamag" / "novamag-raw.csv"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        novamag.to_csv(records_path, index=False)
        _save_modeling_table(
            novamag, periodic_table, miedema_weight, args.output_dir / "novamag-magnetism.csv",
            min_target=args.min_target, excluded_elements=args.exclude_elements,
        )

    if args.dataset in ("all", "mp"):
        print("Processing Materials Project dataset...")
        excluded = args.exclude_elements
        if excluded is None:
            excluded = RARE_EARTH_ELEMENTS + NON_COMMERCIAL_ELEMENTS
        _save_modeling_table(
            load_materials_project(args.mp_csv), periodic_table, miedema_weight,
            args.output_dir / "mp-magnetism.csv", min_target=args.min_target,
            excluded_elements=excluded, magnetic_only=not args.mp_include_nonmagnetic,
        )


if __name__ == "__main__":
    main()
