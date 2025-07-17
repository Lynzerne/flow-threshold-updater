import os
import json
import requests
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, timedelta, date

try:
    import pyarrow
except ImportError:
    raise ImportError("pyarrow is required to read/write Parquet files. Please install it via pip.")
pd.options.io.parquet.engine = "pyarrow"


# --- PATCH START ---
# New update function to keep original published data for finalized days,
# allow filling nulls anytime, and track revisions with revision_number.

def update_master_dataset(master_df, new_data_df, ts_cols, merge_keys=['station_no', 'Date']):
    today = datetime.today().date()
    finalization_cutoff = today - timedelta(days=2)

    def is_nan_or_null(x):
        return pd.isna(x) or (isinstance(x, str) and x.strip() == "")

    updated_rows = []
    master_indices_to_keep = []

    master_indexed = master_df.set_index(merge_keys)

    for _, new_row in new_data_df.iterrows():
        station = new_row['station_no']
        date = pd.to_datetime(new_row['Date']).date()
        key = tuple(new_row[k] for k in merge_keys)

        if key not in master_indexed.index:
            # New record, add as-is
            new_row['is_revised'] = False
            new_row['revision_number'] = 0
            updated_rows.append(new_row)
            continue

        old_row = master_indexed.loc[key]
        if isinstance(old_row, pd.DataFrame):
            old_row = old_row.iloc[0]

        # Allow overwriting null/blank values anytime
        fill_null_allowed = any(
            is_nan_or_null(old_row.get(col)) and not is_nan_or_null(new_row.get(col))
            for col in ts_cols
        )
        if fill_null_allowed:
            # Just overwrite nulls with new data, no revision mark
            new_row['is_revised'] = old_row.get('is_revised', False)
            new_row['revision_number'] = old_row.get('revision_number', 0)
            updated_rows.append(new_row)
            continue

        # If date is recent (<= 2 days old), overwrite freely, no revision
        if date > finalization_cutoff:
            new_row['is_revised'] = False
            new_row['revision_number'] = 0
            updated_rows.append(new_row)
            continue

        # For finalized data, detect revisions and keep original + add revised row
        revised_fields = []
        for col in ts_cols:
            old_val = old_row.get(col)
            new_val = new_row.get(col)
            if pd.isna(old_val) and pd.notna(new_val):
                revised_fields.append(f"{col}: null → {new_val}")
            elif pd.notna(old_val) and pd.notna(new_val) and old_val != new_val:
                revised_fields.append(f"{col}: {old_val} → {new_val}")

        if revised_fields:
            # Keep original row by not removing it here (master_indices_to_keep)
            # Add new revised row with revision_number incremented
            new_row['is_revised'] = True

            existing_revs = master_df[
                (master_df['station_no'] == station) & 
                (master_df['Date'] == new_row['Date']) & 
                (master_df['is_revised'] == True)
            ]
            max_rev = existing_revs['revision_number'].max() if not existing_revs.empty else 0
            new_row['revision_number'] = max_rev + 1

            updated_rows.append(new_row)
            print(f"🔁 Revision for {station} on {date}: {'; '.join(revised_fields)}")
        else:
            # No revision, keep old row
            idx = master_df[
                (master_df['station_no'] == station) & (master_df['Date'] == new_row['Date'])
            ].index[0]
            master_indices_to_keep.append(idx)

    # Keep original rows that were not replaced by revisions
    master_rows_to_keep = master_df.loc[master_indices_to_keep]
    updated_df = pd.DataFrame(updated_rows)

    final_master_df = pd.concat([master_rows_to_keep, updated_df], ignore_index=True)
    final_master_df.sort_values(by=merge_keys, inplace=True)
    return final_master_df
# --- PATCH END ---
# --- Config ---
station_list_csv = "data/AB_WS_R_StationList.csv"
output_parquet = "data/WS_R_master_daily.parquet"
base_url_template = "https://rivers.alberta.ca/apps/Basins/data/figures/river/abrivers/stationdata/WS_R_{}_table.json"

# --- Date Window (last 7 days; source JSON provides last 7 days of data) ---
today = datetime.today().date()
lookback_days = 7
date_window = [today - timedelta(days=i) for i in range(lookback_days)]

# --- Load Stations ---
stns = pd.read_csv(station_list_csv)
required_cols = ['WSC', 'LAT', 'LON']
for c in required_cols:
    if c not in stns.columns:
        raise ValueError(f"Missing column '{c}' in station list CSV")

# --- Load Existing Master Data ---
if os.path.exists(output_parquet):
    master_df = pd.read_parquet(output_parquet, engine="pyarrow")
    master_df['Date'] = pd.to_datetime(master_df['Date']).dt.date
    print(f"Loaded existing master dataset with {len(master_df)} rows.")
else:
    master_df = pd.DataFrame()
    print("No existing master dataset found; starting fresh.")

all_data = []

for _, row in stns.iterrows():
    station_id = row['WSC']
    url = base_url_template.format(station_id)
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Failed to fetch data for {station_id}: {e}")
        continue

    site_name = data.get('station_name', station_id)

    columns = ['Date']
    for md in data.get('ts_metadata', []):
        columns.append(md.get('ts_label', md.get('parameter_name', 'unknown')))

    # Deduplicate columns
    label_counts = Counter()
    deduped_columns = []
    for col in columns:
        count = label_counts[col]
        label_counts[col] += 1
        if count == 0:
            deduped_columns.append(col)
        else:
            deduped_columns.append(f"{col}_{count}")

    rows = []
    for entry in data.get('data', []):
        rows.append(entry.get('values', []))

    if not rows:
        print(f"No data rows for {station_id}, skipping.")
        continue

    df = pd.DataFrame(rows, columns=deduped_columns)
    if 'Date' not in df.columns:
        print(f"No 'Date' column in data for {station_id}, skipping.")
        continue

    df['Date'] = pd.to_datetime(df['Date']).dt.date
    # Remove future-dated entries
    df = df[df['Date'] <= today]

    # Don't filter by date_window - keep all up to 7 days

    if df.empty:
        print(f"No recent data for {station_id}, skipping.")
        continue

    df['station_no'] = station_id
    df['station_name'] = site_name
    df['lon'] = row['LON']
    df['lat'] = row['LAT']

    all_data.append(df)

# --- Merge/Update Master Dataset ---
if all_data:
    new_data_df = pd.concat(all_data, ignore_index=True)

    merge_keys = ['station_no', 'Date']
    metadata_cols = ['station_no', 'station_name', 'Date', 'lon', 'lat']
    ts_cols = [col for col in new_data_df.columns if col not in metadata_cols]

    if not master_df.empty:
        # --- PATCH START ---
        master_df = update_master_dataset(master_df, new_data_df, ts_cols, merge_keys=merge_keys)
        # --- PATCH END ---
    else:
        new_data_df['is_revised'] = False
        new_data_df['revision_number'] = 0
        master_df = new_data_df

    master_df.sort_values(['station_no', 'Date'], inplace=True)
    master_df.to_parquet(output_parquet, index=False, engine="pyarrow")
    print(f"Master dataset saved to {output_parquet}")

else:
    print("No new data collected from any stations.")
#############################Stitch#############################
import pandas as pd
import json
import os

# --- Paths & Date ---
iday = date.today().strftime('%Y-%m-%d')  # Today's date
station_list_csv = "data/AB_WS_R_StationList.csv"
daily_parquet_path = f"data/AB_WS_R_Flows_{iday}.parquet"
master_parquet_path = "data/WS_R_master_daily.parquet"
master_geojson_path = "data/AB_WS_R_stations.geojson"
geojson_updated_path = f"data/AB_WS_R_stations_{iday}.geojson"

# --- Load Station List ---
stns = pd.read_csv(station_list_csv)
stns['WSC'] = stns['WSC'].astype(str).str.strip()

# --- Load Master GeoJSON or Create Skeleton ---
if os.path.exists(master_geojson_path):
    print(f"📄 Loading existing GeoJSON: {master_geojson_path}")
    with open(master_geojson_path, 'r') as f:
        geojson = json.load(f)
else:
    print("⚙️ GeoJSON skeleton not found. Building it...")
    geojson = {"type": "FeatureCollection", "features": []}
    for _, row in stns.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['LON'], row['LAT']]
            },
            "properties": {
                "station_no": row['WSC'],
                "station_name": None,
                "time_series": []
            }
        }
        geojson['features'].append(feature)
    print(f"✅ GeoJSON skeleton created.")

# --- Load Full Master Daily Data to Get All Timeseries ---
master_df = pd.read_parquet(master_parquet_path, engine="pyarrow")
master_df['station_no'] = master_df['station_no'].astype(str).str.strip()

# Columns to exclude from timeseries (metadata)
metadata_cols = {'station_no', 'station_name', 'Date', 'lon', 'lat'}

# Columns to include inside timeseries (parameters)
ts_cols = [col for col in master_df.columns if col not in metadata_cols]

# --- Build a dict for fast feature lookup ---
feature_map = {feat['properties']['station_no']: feat for feat in geojson['features']}

# --- Update features with full timeseries ---
for stn_id, group_df in master_df.groupby('station_no'):
    if stn_id not in feature_map:
        continue
    feat = feature_map[stn_id]

    # Clean out all keys except static metadata & time_series to avoid garbage data
    static_keys = {'station_no', 'station_name', 'lat', 'lon', 'time_series'}
    feat['properties'] = {k: v for k, v in feat['properties'].items() if k in static_keys}

    latest_record = group_df.iloc[-1]
    feat['properties']['station_no'] = stn_id
    feat['properties']['station_name'] = str(latest_record.get('station_name', stn_id))
    feat['properties']['lat'] = float(latest_record.get('lat', feat['properties'].get('lat', 0)))
    feat['properties']['lon'] = float(latest_record.get('lon', feat['properties'].get('lon', 0)))

    # --- Build timeseries list with date, parameter values, and is_revised ---
    timeseries = []
    for _, row in group_df.iterrows():
        ts_entry = {'date': row['Date'].strftime('%Y-%m-%d')}
        for col in ts_cols:
            val = row[col]
            if pd.notnull(val):
                ts_entry[col] = val
        # Add is_revised flag if it exists, else default to False
        ts_entry['is_revised'] = bool(row.get('is_revised', False))
        # Add revision_number for insight
        ts_entry['revision_number'] = int(row.get('revision_number', 0))

        timeseries.append(ts_entry)

    # Defensive cleanup: remove NaN keys inside timeseries dicts (optional)
    for ts_entry in timeseries:
        keys_to_del = [k for k, v in ts_entry.items() if pd.isna(v)]
        for k in keys_to_del:
            del ts_entry[k]

    feat['properties']['time_series'] = timeseries

# --- Save Updated GeoJSON ---
with open(geojson_updated_path, 'w') as f:
    json.dump(geojson, f, indent=2)
print(f"📍 Daily GeoJSON snapshot saved: {geojson_updated_path}")

# --- Update Rolling Master GeoJSON ---
with open(master_geojson_path, 'w') as f:
    json.dump(geojson, f, indent=2)
print(f"🔁 Rolling master GeoJSON updated: {master_geojson_path}")
