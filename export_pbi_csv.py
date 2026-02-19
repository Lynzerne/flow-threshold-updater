"""
export_pbi_csv.py

Creates a PowerBI-friendly CSV from the master parquet.
- Filters to selected stations (05CC set + 05CB007)
- Keeps ALL dates
- Retains all threshold + flow columns automatically
- Ensures numeric fields are clean for PowerBI
"""

import pandas as pd

MASTER_PARQUET = "data/WS_R_master_daily.parquet"
OUT_CSV = "data/pbi_flow_thresholds_05CC_plus_05CB007.csv"

STATIONS = {
    "05CC001",
    "05CC002",
    "05CC007",
    "05CC008",
    "05CC011",
    "05CC013",
    "05CB007",
}

def main():
    # Load master dataset
    df = pd.read_parquet(MASTER_PARQUET)

    # Normalize station number
    df["station_no"] = df["station_no"].astype(str).str.strip().str.upper()

    # Normalize Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Filter to desired stations
    df = df[df["station_no"].isin(STATIONS)].copy()

    if df.empty:
        print("No matching stations found. CSV not written.")
        return

    # Identify metadata columns
    metadata_cols = {"station_no", "station_name", "Date", "lon", "lat"}

    # Everything else is time-series / threshold data
    ts_cols = [c for c in df.columns if c not in metadata_cols]

    keep_cols = [c for c in df.columns if c in metadata_cols or c in ts_cols]
    df = df[keep_cols]

    # Clean numeric columns for PowerBI
    for col in df.columns:
        if col in {"station_no", "station_name", "Date"}:
            continue
        if col == "is_revised":
            df[col] = df[col].astype("boolean")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort cleanly
    df = df.sort_values(["station_no", "Date"])

    # Write CSV
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {OUT_CSV} with {len(df)} rows and {len(df.columns)} columns.")

if __name__ == "__main__":
    main()
