from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RAW_DIRECTORY = Path("data/raw/2025-01")
REPORT_DIRECTORY = Path("reports")


def load_raw_pages(directory: Path) -> pd.DataFrame:
    """
    Read every page_*.json file and combine them into one DataFrame.
    """

    page_files = sorted(directory.glob("page_*.json"))

    if not page_files:
        raise FileNotFoundError(
            f"No page files were found inside {directory}."
        )

    dataframes: list[pd.DataFrame] = []

    for page_file in page_files:
        print(f"Reading {page_file}...")

        with page_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)

        page_dataframe = pd.DataFrame.from_records(records)
        dataframes.append(page_dataframe)

    combined_dataframe = pd.concat(
        dataframes,
        ignore_index=True,
    )

    return combined_dataframe


def create_column_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a summary describing every column.
    """

    profile = pd.DataFrame(
        {
            "data_type": dataframe.dtypes.astype(str),
            "row_count": len(dataframe),
            "non_null_count": dataframe.notna().sum(),
            "null_count": dataframe.isna().sum(),
            "null_percentage": (
                dataframe.isna().mean() * 100
            ).round(2),
            "unique_count": dataframe.nunique(
                dropna=True
            ),
        }
    )

    return profile.sort_values(
        "null_percentage",
        ascending=False,
    )


def main() -> None:
    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_raw_pages(RAW_DIRECTORY)

    print("\nData loaded successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head())

    print("\nExact duplicate rows:")
    print(df.duplicated().sum())

    if "unique_key" in df.columns:
        duplicate_keys = df.duplicated(
            subset=["unique_key"],
            keep=False,
        ).sum()

        print("\nRows with duplicated unique keys:")
        print(duplicate_keys)

    profile = create_column_profile(df)

    report_file = REPORT_DIRECTORY / "raw_column_profile.csv"
    profile.to_csv(report_file)

    print(f"\nColumn profile saved to {report_file}")


if __name__ == "__main__":
    main()