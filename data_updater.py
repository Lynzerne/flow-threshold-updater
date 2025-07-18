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

# --- Merge/Update Master Dataset (Revised for Auditing) ---
if all_data:
    new_data_df = pd.concat(all_data, ignore_index=True)
    new_data_df['Date'] = pd.to_datetime(new_data_df['Date']).dt.date # Ensure date type consistency

    merge_keys = ['station_no', 'Date']
    metadata_cols = ['station_no', 'station_name', 'Date', 'lon', 'lat']
    ts_cols = [col for col in new_data_df.columns if col not in metadata_cols]

    if not master_df.empty:
        # Create a combined DataFrame of existing master and new data
        # This will contain duplicate rows for existing (station, date) pairs
        combined_df = pd.concat([master_df, new_data_df], ignore_index=True)

        # Sort by station, date, and then potentially by a timestamp of collection (if you had one)
        # For now, we'll rely on the existing data being the "older" one if there are duplicates.
        combined_df.sort_values(merge_keys, inplace=True)

        # Drop duplicates, keeping the FIRST occurrence (i.e., the original master_df record)
        # This ensures that if data for a (station, date, column) already exists and is not NaN,
        # it is NOT overwritten by a new non-NaN value from new_data_df.
        # This is the core change for compliance.
        # We also want to update metadata from the latest scrape, as it doesn't represent "data."
        # So we'll update those separately.

        # First, handle the time-series data: only update NaNs
        # Group by merge_keys and fill NaNs forward from the combined (old then new) data
        # This ensures that if the old value was NaN and the new one has a value, it's filled.
        # If old value was already non-NaN, it remains.
        filled_ts_data = combined_df.groupby(merge_keys)[ts_cols].apply(
            lambda x: x.bfill().iloc[0] if not x.isnull().all().all() else x.iloc[0]
        ).reset_index()

        # Re-merge the metadata (station_name, lat, lon) from the latest available scrape (new_data_df)
        # and ensure original is_revised flag is handled carefully.
        # We need to preserve the is_revised flag from the master_df if the value wasn't revised
        # to ensure historical accuracy, or set it if a blank was filled.

        # Identify which values were actually "revised" (i.e., blank filled)
        # Create a temporary DataFrame of old values for comparison
        temp_master_for_comparison = master_df.set_index(merge_keys)
        
        # This list will store the final, unique rows for the new master_df
        final_master_rows = []

        # Iterate through unique station-date combinations from the combined data
        for (stn, dt), group in combined_df.groupby(merge_keys):
            existing_master_row = temp_master_for_comparison.loc[(stn, dt)] if (stn, dt) in temp_master_for_comparison.index else None
            new_scrape_row = new_data_df[(new_data_df['station_no'] == stn) & (new_data_df['Date'] == dt)].iloc[0] if not new_data_df[(new_data_df['station_no'] == stn) & (new_data_df['Date'] == dt)].empty else None

            # Start with new scrape row as the base, then overlay existing master data
            # for actual flow values IF they existed and were non-NaN.
            final_row_data = new_scrape_row.to_dict() if new_scrape_row is not None else {}
            
            is_revised_flag = False
            for col in ts_cols:
                new_val = new_scrape_row[col] if new_scrape_row is not None and col in new_scrape_row else None
                
                # Check if this specific value was "filled in" (was NaN in master_df, now has value)
                if existing_master_row is not None and col in existing_master_row:
                    old_val = existing_master_row[col]
                    if pd.isna(old_val) and pd.notna(new_val):
                        # This specific parameter was revised (filled from NaN)
                        is_revised_flag = True
                        final_row_data[col] = new_val # Take the new value as it was filling a blank
                    elif pd.notna(old_val):
                        # If old value was already NOT NaN, keep the OLD value for audit compliance
                        final_row_data[col] = old_val
                        # The 'is_revised' flag on the old row might indicate it was blank-filled previously.
                        # For audit, we only care if *we* just revised it.
                        # You might need a separate mechanism if you need to track *every* revision.
                        # For now, if old value was present, we assume it's the "official" version.
                    else: # old_val was NaN, new_val is also NaN
                        final_row_data[col] = new_val # Keep it NaN
                else: # New (station, date) combination, or new column for existing (station, date)
                    final_row_data[col] = new_val # Take the new value

            # Set 'is_revised' based on if *we* just filled a blank from the scrape
            final_row_data['is_revised'] = is_revised_flag
            
            # Reconstruct the row for the final DataFrame
            final_master_rows.append(final_row_data)

        master_df = pd.DataFrame(final_master_rows)
        master_df = master_df.drop_duplicates(subset=merge_keys, keep='first') # Just in case to ensure unique rows
        
    else: # master_df is empty, so all new_data_df are new records
        new_data_df['is_revised'] = False
        master_df = new_data_df

    master_df.sort_values(merge_keys, inplace=True) # Sort by WSC and Date

    master_df.to_parquet(output_parquet, index=False, engine="pyarrow")
    print(f"Master dataset saved to {output_parquet}")

    # --- Save Daily Snapshot for Most Recent Date Available ---
    # This snapshot will reflect the data as of today's run based on the master_df
    # and is suitable for displaying the current state in the app.
    iday = master_df['Date'].max() # Get the latest date available in the master data
    daily_snapshot_df = master_df[master_df['Date'] == iday]

    daily_parquet_path = f"data/AB_WS_R_Flows_{iday}.parquet"
    daily_snapshot_df.to_parquet(daily_parquet_path, index=False, engine="pyarrow")
    print(f"Daily snapshot Parquet saved to {daily_parquet_path}")

else:
    print("No new data collected from any stations.")

#############################Stitch#############################
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

