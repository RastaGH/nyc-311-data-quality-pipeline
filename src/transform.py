from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RAW_DIRECTORY = Path("data/raw/2025-01")
PROCESSED_DIRECTORY = Path("data/processed")
REPORT_DIRECTORY = Path("reports")


DATE_COLUMNS = [
    "created_date",
    "closed_date",
    "due_date",
    "resolution_action_updated_date",
]

TEXT_COLUMNS = [
    "unique_key",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "city",
    "status",
    "community_board",
    "council_district",
    "police_precinct",
    "borough",
    "open_data_channel_type",
]

EXPECTED_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "city",
    "status",
    "due_date",
    "resolution_action_updated_date",
    "community_board",
    "council_district",
    "police_precinct",
    "borough",
    "open_data_channel_type",
    "latitude",
    "longitude",
]


def load_raw_pages(directory: Path) -> pd.DataFrame:
    page_files = sorted(directory.glob("page_*.json"))

    if not page_files:
        raise FileNotFoundError(
            f"No raw page files were found in {directory}."
        )

    frames: list[pd.DataFrame] = []

    for page_file in page_files:
        with page_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)

        frames.append(
            pd.DataFrame.from_records(records)
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def ensure_expected_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add any expected column that is absent.
    Missing columns are filled with null values.
    """

    result = dataframe.copy()

    for column in EXPECTED_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    return result[EXPECTED_COLUMNS]


def clean_text_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    for column in TEXT_COLUMNS:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    return result


def clean_dates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    for column in DATE_COLUMNS:
        result[column] = pd.to_datetime(
            result[column],
            errors="coerce",
        )

    return result


def clean_coordinates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result["latitude"] = pd.to_numeric(
        result["latitude"],
        errors="coerce",
    )

    result["longitude"] = pd.to_numeric(
        result["longitude"],
        errors="coerce",
    )

    result["valid_latitude"] = (
        result["latitude"].isna()
        | result["latitude"].between(-90, 90)
    )

    result["valid_longitude"] = (
        result["longitude"].isna()
        | result["longitude"].between(-180, 180)
    )

    result["has_valid_coordinates"] = (
        result["latitude"].notna()
        & result["longitude"].notna()
        & result["valid_latitude"]
        & result["valid_longitude"]
    )

    return result


def clean_zip_codes(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result["incident_zip"] = (
        result["incident_zip"]
        .astype("string")
        .str.extract(r"(\d{5})", expand=False)
    )

    result["valid_zip_format"] = (
        result["incident_zip"].isna()
        | result["incident_zip"].str.fullmatch(
            r"\d{5}"
        )
    )

    return result


def standardize_categories(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    # Keep copies of original values.
    result["borough_raw"] = result["borough"]
    result["status_raw"] = result["status"]

    borough_mapping = {
        "BRONX": "Bronx",
        "BROOKLYN": "Brooklyn",
        "MANHATTAN": "Manhattan",
        "QUEENS": "Queens",
        "STATEN ISLAND": "Staten Island",
        "UNSPECIFIED": "Unspecified",
    }

    normalized_borough = (
        result["borough"]
        .str.upper()
        .str.strip()
    )

    result["borough"] = normalized_borough.map(
        borough_mapping
    )

    result["status"] = (
        result["status"]
        .str.strip()
        .str.title()
    )

    result["open_data_channel_type"] = (
        result["open_data_channel_type"]
        .str.strip()
        .str.title()
    )

    return result


def create_date_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result["created_year"] = (
        result["created_date"].dt.year
    )

    result["created_month_number"] = (
        result["created_date"].dt.month
    )

    result["created_month_name"] = (
        result["created_date"].dt.month_name()
    )

    result["created_day_name"] = (
        result["created_date"].dt.day_name()
    )

    result["created_hour"] = (
        result["created_date"].dt.hour
    )

    result["created_date_only"] = (
        result["created_date"].dt.date
    )

    return result


def create_resolution_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result["is_closed"] = (
        result["closed_date"].notna()
    )

    result["invalid_date_sequence"] = (
        result["closed_date"].notna()
        & result["created_date"].notna()
        & (
            result["closed_date"]
            < result["created_date"]
        )
    )

    result["resolution_hours"] = (
        (
            result["closed_date"]
            - result["created_date"]
        ).dt.total_seconds()
        / 3600
    )

    # Invalid negative durations should not be analyzed.
    result.loc[
        result["invalid_date_sequence"],
        "resolution_hours",
    ] = pd.NA

    result["resolution_days"] = (
        result["resolution_hours"] / 24
    )

    result["sla_evaluable"] = (
        result["closed_date"].notna()
        & result["due_date"].notna()
    )

    result["sla_met"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )

    evaluable = result["sla_evaluable"]

    result.loc[evaluable, "sla_met"] = (
        result.loc[evaluable, "closed_date"]
        <= result.loc[evaluable, "due_date"]
    )

    return result


def handle_duplicates(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = dataframe.copy()

    duplicate_mask = result.duplicated(
        subset=["unique_key"],
        keep=False,
    )

    duplicate_records = result[
        duplicate_mask
    ].copy()

    # Sort so the most recently updated version is last.
    result = result.sort_values(
        by=[
            "unique_key",
            "resolution_action_updated_date",
        ],
        na_position="first",
    )

    result = result.drop_duplicates(
        subset=["unique_key"],
        keep="last",
    )

    return result, duplicate_records


def create_quality_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    total_rows = len(dataframe)

    quality_results = [
        {
            "rule": "unique_key_missing",
            "failure_count": int(
                dataframe["unique_key"].isna().sum()
            ),
        },
        {
            "rule": "created_date_missing",
            "failure_count": int(
                dataframe["created_date"].isna().sum()
            ),
        },
        {
            "rule": "agency_missing",
            "failure_count": int(
                dataframe["agency"].isna().sum()
            ),
        },
        {
            "rule": "complaint_type_missing",
            "failure_count": int(
                dataframe["complaint_type"].isna().sum()
            ),
        },
        {
            "rule": "invalid_date_sequence",
            "failure_count": int(
                dataframe[
                    "invalid_date_sequence"
                ].sum()
            ),
        },
        {
            "rule": "missing_borough",
            "failure_count": int(
                dataframe["borough"].isna().sum()
            ),
        },
        {
            "rule": "invalid_or_missing_coordinates",
            "failure_count": int(
                (
                    ~dataframe[
                        "has_valid_coordinates"
                    ]
                ).sum()
            ),
        },
    ]

    report = pd.DataFrame(quality_results)

    report["total_rows"] = total_rows

    report["failure_percentage"] = (
        report["failure_count"]
        / total_rows
        * 100
    ).round(2)

    return report


def main() -> None:
    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading raw data...")
    df = load_raw_pages(RAW_DIRECTORY)

    print(f"Raw rows: {len(df):,}")

    df = ensure_expected_columns(df)
    df = clean_text_columns(df)
    df = clean_dates(df)
    df = clean_coordinates(df)
    df = clean_zip_codes(df)
    df = standardize_categories(df)
    df = create_date_features(df)
    df = create_resolution_features(df)

    df, duplicate_records = handle_duplicates(df)

    print(f"Clean rows: {len(df):,}")
    print(
        "Duplicate records saved for review:",
        len(duplicate_records),
    )

    clean_output = (
        PROCESSED_DIRECTORY
        / "nyc_311_2025_01_clean.parquet"
    )

    sample_output = (
        Path("data/samples")
        / "nyc_311_2025_01_clean_sample.csv"
    )

    duplicate_output = (
        REPORT_DIRECTORY
        / "duplicate_unique_keys.csv"
    )

    quality_output = (
        REPORT_DIRECTORY
        / "data_quality_report.csv"
    )

    df.to_parquet(
        clean_output,
        index=False,
    )

    df.head(500).to_csv(
        sample_output,
        index=False,
    )

    duplicate_records.to_csv(
        duplicate_output,
        index=False,
    )

    quality_report = create_quality_report(df)

    quality_report.to_csv(
        quality_output,
        index=False,
    )

    print(f"Clean data saved to {clean_output}")
    print(f"Sample saved to {sample_output}")
    print(f"Duplicates saved to {duplicate_output}")
    print(f"Quality report saved to {quality_output}")


if __name__ == "__main__":
    main()