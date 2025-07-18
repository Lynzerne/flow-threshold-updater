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
today = datetime.today().date() # Current date of script execution
lookback_days = 7

# --- Load Stations ---
stns = pd.read_csv(station_list_csv)
required_cols = ['WSC', 'LAT', 'LON']
for c in required_cols:
    if c not in stns.columns:
        raise ValueError(f"Missing column '{c}' in station list CSV")

# --- Load Existing Master Data ---
if os.path.exists(output_parquet):
    master_df = pd.read_parquet(output_parquet, engine="pyarrow")
    # Ensure 'Date' column is consistently a datetime.date object upon load
    # Convert to datetime64[ns] first, then to date objects, handling potential NaNs
    master_df['Date'] = pd.to_datetime(master_df['Date'], errors='coerce').dt.date
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

    # Ensure 'Date' column is converted to date objects early
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

    # Filter out any future dates from the scraped data here
    df = df[df['Date'] <= today]
    # Filter out rows where 'Date' became NaT due to coercion errors
    df = df.dropna(subset=['Date'])
    
    if df.empty:
        print(f"No recent (today or past) data for {station_id}, skipping.")
        continue

    df['station_no'] = station_id
    df['station_name'] = site_name
    df['lon'] = row['LON']
    df['lat'] = row['LAT']

    all_data.append(df)

# --- Merge/Update Master Dataset (Revised for Date-Based Logic - FIX for KeyError: 'Date' - Version 2) ---
if all_data:
    new_data_df = pd.concat(all_data, ignore_index=True)

    merge_keys = ['station_no', 'Date']
    # Ensure all merge_keys are in new_data_df after concat.
    # This might already be handled if df = df[df['Date'] <= today] ensures Date exists.
    # But it's good to be defensive.
    for col in merge_keys:
        if col not in new_data_df.columns:
            new_data_df[col] = pd.NA

    metadata_cols = ['station_no', 'station_name', 'Date', 'lon', 'lat']
    # Dynamically determine all possible columns from both master and new_data_df
    # This ensures that `ts_cols` includes all potential columns that might appear.
    all_possible_columns = list(set(master_df.columns.tolist() + new_data_df.columns.tolist()))
    ts_cols = [col for col in all_possible_columns if col not in metadata_cols]

    if not master_df.empty:
        updated_rows_list = []

        master_dict = master_df.set_index(merge_keys).to_dict('index')

        # Ensure consistency in date types for keys before creating all_unique_keys_df
        master_df_keys = master_df[merge_keys].copy()
        master_df_keys['Date'] = pd.to_datetime(master_df_keys['Date'], errors='coerce').dt.date.dropna() # Drop NaTs here
        
        new_data_df_keys = new_data_df[merge_keys].copy()
        new_data_df_keys['Date'] = pd.to_datetime(new_data_df_keys['Date'], errors='coerce').dt.date.dropna() # Drop NaTs here


        all_unique_keys_df = pd.DataFrame(columns=merge_keys)
        if not master_df_keys.empty:
            all_unique_keys_df = pd.concat([all_unique_keys_df, master_df_keys], ignore_index=True)
        if not new_data_df_keys.empty:
            all_unique_keys_df = pd.concat([all_unique_keys_df, new_data_df_keys], ignore_index=True)
        all_unique_keys_df.drop_duplicates(inplace=True)


        for _, row_key in all_unique_keys_df.iterrows():
            stn = row_key['station_no']
            dt = row_key['Date'] 

            existing_master_row_data = master_dict.get((stn, dt))
            new_scrape_row_data = new_data_df[(new_data_df['station_no'] == stn) & (new_data_df['Date'] == dt)].iloc[0].to_dict() if not new_data_df[(new_data_df['station_no'] == stn) & (new_data_df['Date'] == dt)].empty else None

            # Initialize current_row_data with ALL potential columns set to None/NaN
            current_row_data = {col: None for col in all_possible_columns}
            current_row_data['is_revised'] = False # Default

            # Overlay with existing master data if available
            if existing_master_row_data is not None:
                current_row_data.update(existing_master_row_data)

            # Overlay with new scraped data, prioritizing metadata from new_scrape_row_data
            if new_scrape_row_data is not None:
                current_row_data.update(new_scrape_row_data) # This will overwrite if keys match

            # Re-assert essential merge keys and metadata for safety, ensure correct type for 'Date'
            current_row_data['station_no'] = stn
            current_row_data['Date'] = dt # Already a date object from all_unique_keys_df loop

            is_row_revised = False 

            # --- Update Time-Series Columns ---
            for col in ts_cols:
                old_val = current_row_data.get(col) # This is the value from master_df (or initialized None)
                new_val = new_scrape_row_data.get(col) if new_scrape_row_data is not None else None

                if pd.isna(old_val) and pd.notna(new_val):
                    current_row_data[col] = new_val
                    is_row_revised = True
                # Else: if old_val is not NaN, we keep it (desired audit behavior) - current_row_data already holds it.
                # If both are NaN, it remains NaN.

            # Metadata (station_name, lon, lat) should always come from the latest available source (new_scrape_row_data if present)
            # If not in new_scrape_row_data, it remains what was in existing_master_row_data or None.
            if new_scrape_row_data is not None:
                current_row_data['station_name'] = new_scrape_row_data.get('station_name', current_row_data.get('station_name'))
                current_row_data['lon'] = new_scrape_row_data.get('lon', current_row_data.get('lon'))
                current_row_data['lat'] = new_scrape_row_data.get('lat', current_row_data.get('lat'))
            
            current_row_data['is_revised'] = is_row_revised
            updated_rows_list.append(current_row_data)

        updated_df = pd.DataFrame(updated_rows_list)

        # FINAL DTYPE CONVERSIONS FOR THE ENTIRE DATAFRAME
        # Convert 'Date' column to datetime.date objects, handling NaTs
        updated_df['Date'] = pd.to_datetime(updated_df['Date'], errors='coerce').dt.date
        
        # Convert numeric columns, coercing errors to NaN
        for col in ts_cols:
            if col in updated_df.columns: 
                updated_df[col] = pd.to_numeric(updated_df[col], errors='coerce')
            else:
                updated_df[col] = pd.NA # Add column if it doesn't exist

        # Ensure essential metadata columns are present, initialized to pd.NA if missing
        for col in ['station_no', 'station_name', 'lon', 'lat', 'is_revised']:
            if col not in updated_df.columns:
                updated_df[col] = pd.NA


        # Reconstruct master_df
        updated_keys_df = updated_df[merge_keys].drop_duplicates()
        master_df = master_df[~master_df.set_index(merge_keys).index.isin(updated_keys_df.set_index(merge_keys).index)]
        master_df = pd.concat([master_df, updated_df], ignore_index=True)

    else:
        new_data_df['is_revised'] = False
        master_df = new_data_df
        master_df['Date'] = pd.to_datetime(master_df['Date'], errors='coerce').dt.date

    master_df.sort_values(merge_keys, inplace=True)

    master_df.to_parquet(output_parquet, index=False, engine="pyarrow")
    print(f"Master dataset saved to {output_parquet}")

    # --- Save Daily Snapshot ---
    # Final check for master_df and 'Date' column validity before trying to get max date
    if not master_df.empty and 'Date' in master_df.columns and not master_df['Date'].isnull().all():
        iday = master_df['Date'].max()
        daily_snapshot_df = master_df[master_df['Date'] == iday]

        daily_parquet_path = f"data/AB_WS_R_Flows_{iday}.parquet"
        daily_snapshot_df.to_parquet(daily_parquet_path, index=False, engine="pyarrow")
        print(f"Daily snapshot Parquet saved to {daily_parquet_path}")
    else:
        print("Master dataset is empty or 'Date' column is problematic, no daily snapshot saved.")

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

