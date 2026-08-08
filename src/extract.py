from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sodapy import Socrata


DOMAIN = "data.cityofnewyork.us"
DATASET_ID = "erm2-nwe9"

# Start with a smaller page size while learning.
PAGE_SIZE = 10_000

SELECT_COLUMNS = """
    unique_key,
    created_date,
    closed_date,
    agency,
    agency_name,
    complaint_type,
    descriptor,
    location_type,
    incident_zip,
    city,
    status,
    due_date,
    resolution_action_updated_date,
    community_board,
    council_district,
    police_precinct,
    borough,
    open_data_channel_type,
    latitude,
    longitude
"""


def download_month(
    year: int,
    month: int,
    next_year: int,
    next_month: int,
) -> None:
    """
    Download one month of NYC 311 records in separate JSON pages.
    """

    load_dotenv()

    app_token = os.getenv("SOCRATA_APP_TOKEN")

    if not app_token:
        raise RuntimeError(
            "SOCRATA_APP_TOKEN is missing from the .env file."
        )

    start_date = f"{year:04d}-{month:02d}-01T00:00:00.000"
    end_date = (
    f"{next_year:04d}-{next_month:02d}-01T00:00:00.000"
)

    where_clause = f"""
        created_date >= '{start_date}'
        AND created_date < '{end_date}'
    """

    output_directory = Path(
        f"data/raw/{year:04d}-{month:02d}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    client = Socrata(
        DOMAIN,
        app_token,
        timeout=180,
    )

    offset = 0
    page_number = 1
    total_records = 0

    try:
        while True:
            print(
                f"Downloading page {page_number} "
                f"starting at offset {offset:,}..."
            )

            rows = client.get(
                DATASET_ID,
                select=SELECT_COLUMNS,
                where=where_clause,
                order="created_date ASC, unique_key ASC",
                limit=PAGE_SIZE,
                offset=offset,
            )

            if not rows:
                print("No additional records were returned.")
                break

            output_file = (
                output_directory
                / f"page_{page_number:04d}.json"
            )

            with output_file.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    rows,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            rows_downloaded = len(rows)
            total_records += rows_downloaded

            print(
                f"Saved {rows_downloaded:,} records to "
                f"{output_file}"
            )

            if rows_downloaded < PAGE_SIZE:
                print("Reached the final page.")
                break

            offset += PAGE_SIZE
            page_number += 1

    finally:
        client.close()

    print("\nDownload complete.")
    print(f"Total records: {total_records:,}")
    print(f"Output folder: {output_directory}")


if __name__ == "__main__":
    download_month(
        year=2025,
        month=1,
        next_year=2025,
        next_month=2,
    )