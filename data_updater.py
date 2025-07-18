import os
import json
import requests
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta, date

try:
    import pyarrow
except ImportError:
    raise ImportError("pyarrow is required to read/write Parquet files. Please install it via pip.")
pd.options.io.parquet.engine = "pyarrow"

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


    # Don't filter by date_window - keep all dates from source JSON (up to 7 days)
    # This allows backfill of missing days automatically.

    if df.empty:
        print(f"No recent data for {station_id}, skipping.")
        continue

    df['station_no'] = station_id
    df['station_name'] = site_name
    df['lon'] = row['LON']
    df['lat'] = row['LAT']

    all_data.append(df)

# --- Merge/Update Master Dataset (Revised for Date-Based Logic - FIX for KeyError: 'Date') ---
if all_data:
    new_data_df = pd.concat(all_data, ignore_index=True)

    merge_keys = ['station_no', 'Date']
    metadata_cols = ['station_no', 'station_name', 'Date', 'lon', 'lat']
    ts_cols = [col for col in new_data_df.columns if col not in metadata_cols]

    if not master_df.empty:
        # Create a temporary DataFrame to hold the updated rows
        updated_rows_list = []
        
        # Convert master_df to a dictionary for faster lookups
        master_dict = master_df.set_index(merge_keys).to_dict('index')

        # Use a combined set of (station_no, Date) pairs to ensure we cover all existing and new dates
        # This handles cases where a station-date exists in master_df but not in new_data_df (e.g., no recent scrape data for it)
        # Or vice versa.
        # Ensure 'Date' column exists before dropping duplicates, if new_data_df is empty.
        all_unique_keys_df = pd.DataFrame(columns=merge_keys)
        if not master_df.empty:
            all_unique_keys_df = pd.concat([all_unique_keys_df, master_df[merge_keys]], ignore_index=True)
        if not new_data_df.empty: # Only add new_data_df keys if it's not empty
            all_unique_keys_df = pd.concat([all_unique_keys_df, new_data_df[merge_keys]], ignore_index=True)
        all_unique_keys_df.drop_duplicates(inplace=True)


        for _, row_key in all_unique_keys_df.iterrows():
            stn = row_key['station_no']
            dt = row_key['Date']

            existing_master_row_data = master_dict.get((stn, dt)) # Use .get() to safely retrieve None if not found
            new_scrape_row_data = new_data_df[(new_data_df['station_no'] == stn) & (new_data_df['Date'] == dt)].iloc[0].to_dict() if not new_data_df[(new_data_df['station_no'] == stn) & (new_data_df['Date'] == dt)].empty else None

            # Initialize the row to be added to the temporary master.
            # Start with existing master data if available, otherwise new scrape data, otherwise just the key info.
            if existing_master_row_data is not None:
                current_row_data = existing_master_row_data.copy()
            elif new_scrape_row_data is not None:
                current_row_data = new_scrape_row_data.copy()
            else:
                # If neither existing nor new data, it means this key came from master_df but has no corresponding scrape.
                # In this case, we still need to preserve it. Initialize with merge_keys.
                current_row_data = {'station_no': stn, 'Date': dt, 'is_revised': False}
                # Add other metadata if known from master_df, otherwise they might be NaN initially.
                # This ensures at least 'Date' and 'station_no' exist.


            is_row_revised = False # Flag for this entire row

            # --- Update Time-Series Columns ---
            for col in ts_cols: # Iterate through all possible time-series columns
                old_val = existing_master_row_data.get(col) if existing_master_row_data is not None else None
                new_val = new_scrape_row_data.get(col) if new_scrape_row_data is not None and col in new_scrape_row_data else None

                # Only update if the old value was NaN AND the new value is not NaN
                if pd.isna(old_val) and pd.notna(new_val):
                    current_row_data[col] = new_val # Fill the blank
                    is_row_revised = True # Mark as revised if a blank was filled
                elif pd.notna(old_val): # If old value exists and is not NaN, keep it
                    current_row_data[col] = old_val
                else: # Both old and new are NaN, or no new value; ensure it's in current_row_data
                    current_row_data[col] = new_val # This will make it NaN if new_val is NaN or None

            # Update metadata (station_name, lat, lon) from the *latest* scrape, if available,
            # otherwise keep existing, or make NaN if neither.
            if new_scrape_row_data is not None:
                current_row_data['station_name'] = new_scrape_row_data.get('station_name', current_row_data.get('station_name'))
                current_row_data['lon'] = new_scrape_row_data.get('lon', current_row_data.get('lon'))
                current_row_data['lat'] = new_scrape_row_data.get('lat', current_row_data.get('lat'))
            elif existing_master_row_data is not None: # If only existing master data, use its metadata
                current_row_data['station_name'] = existing_master_row_data.get('station_name')
                current_row_data['lon'] = existing_master_row_data.get('lon')
                current_row_data['lat'] = existing_master_row_data.get('lat')
            else: # If a completely new record was constructed from all_unique_keys_df, ensure these are present
                current_row_data['station_name'] = None
                current_row_data['lon'] = None
                current_row_data['lat'] = None


            # Set the is_revised flag for the row
            current_row_data['is_revised'] = is_row_revised

            updated_rows_list.append(current_row_data)

        # Convert list of dicts to DataFrame
        updated_df = pd.DataFrame(updated_rows_list)
        # Ensure correct dtypes, especially for 'Date' and numeric columns
        # Explicitly ensure merge_keys and metadata_cols are present even if all NaN.
        for col in merge_keys + metadata_cols:
            if col not in updated_df.columns:
                updated_df[col] = None # Or pd.NA, or a default value

        updated_df['Date'] = pd.to_datetime(updated_df['Date']).dt.date
        for col in ts_cols: # Convert time-series cols to numeric, coercing errors
             updated_df[col] = pd.to_numeric(updated_df[col], errors='coerce')


        # Reconstruct master_df: remove old versions of updated rows, then concat
        updated_keys_df = updated_df[merge_keys].drop_duplicates()
        master_df = master_df[~master_df.set_index(merge_keys).index.isin(updated_keys_df.set_index(merge_keys).index)]
        master_df = pd.concat([master_df, updated_df], ignore_index=True)

    else:
        # Master_df was empty, so simply initialize it with new_data_df
        new_data_df['is_revised'] = False
        master_df = new_data_df

    master_df.sort_values(merge_keys, inplace=True)

    master_df.to_parquet(output_parquet, index=False, engine="pyarrow")
    print(f"Master dataset saved to {output_parquet}")

    # --- Save Daily Snapshot for Most Recent Date Available ---
    # This snapshot will reflect the data as of today's run based on the master_df
    # and is suitable for displaying the current state in the app.
    # Check if master_df is empty before trying to get max date
    if not master_df.empty:
        iday = master_df['Date'].max()
        daily_snapshot_df = master_df[master_df['Date'] == iday]

        daily_parquet_path = f"data/AB_WS_R_Flows_{iday}.parquet"
        daily_snapshot_df.to_parquet(daily_parquet_path, index=False, engine="pyarrow")
        print(f"Daily snapshot Parquet saved to {daily_parquet_path}")
    else:
        print("Master dataset is empty, no daily snapshot saved.")

else:
    print("No new data collected from any stations.")
#############################Stitch#############################
from datetime import date
import pandas as pd
import json
import os
import geopandas as gpd # <--- ADD THIS IMPORT

# --- Paths & Date ---
iday = date.today().strftime('%Y-%m-%d')
station_list_csv = "data/AB_WS_R_StationList.csv"
daily_parquet_path = f"data/AB_WS_R_Flows_{iday}.parquet" # This is a daily snapshot from updater
master_parquet_path = "data/WS_R_master_daily.parquet" # This is the full, continuously updated daily data
master_geojson_path = "data/AB_WS_R_stations.geojson" # This is the rolling master GeoJSON
geojson_updated_path = f"data/AB_WS_R_stations_{iday}.geojson" # This is the daily GeoJSON snapshot

# --- NEW: Define the path for the PARQUET file the app will read ---
output_app_parquet_path = "data/AB_WS_R_stations.parquet" # <--- NEW PATH DEFINITION

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

    # Build timeseries list with date, parameter values, and is_revised
    timeseries = []
    for _, row in group_df.iterrows():
        ts_entry = {'date': row['Date'].strftime('%Y-%m-%d')}
        for col in ts_cols:
            val = row[col]
            if pd.notnull(val) and str(val).strip() != '':
                try:
                    ts_entry[col] = float(val)
                except (ValueError, TypeError):
                    pass
        ts_entry['is_revised'] = bool(row.get('is_revised', False))
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

# --- NEW: Convert GeoJSON back to GeoDataFrame and save as AB_WS_R_stations.parquet ---
# This is the file your Streamlit app loads!
try:
    # Convert the geojson dictionary (FeatureCollection) into a GeoDataFrame
    # This requires 'geometry' column to be properly formatted (e.g., from WKT or dicts)
    # The 'geojson' variable here is already the full FeatureCollection dictionary.
    # geopandas.GeoDataFrame.from_features expects a list of features or a dict with 'features' key.
    stations_gdf = gpd.GeoDataFrame.from_features(geojson['features'])

    # Ensure the 'geometry' column is correctly set if it's not automatically inferred
    # In your case, it should be fine since you set geometry.coordinates
    
    # Save to Parquet
    stations_gdf.to_parquet(output_app_parquet_path, index=False, engine="pyarrow")
    print(f"✅ Stations GeoDataFrame saved to {output_app_parquet_path} for Streamlit app.")

except Exception as e:
    print(f"❌ Error saving GeoDataFrame to Parquet: {e}")

