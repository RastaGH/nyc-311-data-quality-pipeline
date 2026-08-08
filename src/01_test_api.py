import os

import pandas as pd
from dotenv import load_dotenv
from sodapy import Socrata


# Load variables from the .env file.
load_dotenv()

# Read the application token.
app_token = os.getenv("SOCRATA_APP_TOKEN")

# Basic API information.
domain = "data.cityofnewyork.us"
dataset_id = "erm2-nwe9"

# Create the API connection.
client = Socrata(
    domain,
    app_token,
    timeout=120,
)

# Request 100 records from January 1, 2025.
results = client.get(
    dataset_id,
    select="""
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
        borough,
        open_data_channel_type,
        latitude,
        longitude
    """,
    where="""
        created_date >= '2025-01-01T00:00:00.000'
        AND created_date < '2025-01-02T00:00:00.000'
    """,
    order="created_date ASC, unique_key ASC",
    limit=100,
)

# Convert the results into a Pandas DataFrame.
df = pd.DataFrame.from_records(results)

# Print basic information.
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst five rows:")
print(df.head())


from pathlib import Path


# Create the samples folder when it does not already exist.
output_directory = Path("data/samples")
output_directory.mkdir(parents=True, exist_ok=True)

# Save the sample as CSV.
output_file = output_directory / "nyc_311_sample_100.csv"
df.to_csv(output_file, index=False)

print(f"\nSample saved to: {output_file}")


# Close the client connection.
client.close()