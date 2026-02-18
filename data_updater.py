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
today = datetime.today().date()  # Current date of script execution
lookback_days = 7  # (kept for readability; filtering is done by <= today)

# --- Load Stations ---
stns = pd.read_csv(station_list_csv)
required_cols = ['WSC', 'LAT', 'LON']
for c in required_cols:
    if c not in stns.columns:
        raise ValueError(f"Missing column '{c}' in station list CSV")

# --- Load Existing Master Data ---
if os.path.exists(output_parquet):
    master_df = pd.read_parquet(output_parquet, engine="pyarrow")
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

# --- Merge/Update Master Dataset ---
if all_data:
    new_data_df = pd.concat(all_data, ignore_index=True)

    merge_keys = ['station_no', 'Date']
    for col in merge_keys:
        if col not in new_data_df.columns:
            new_data_df[col] = pd.NA

    metadata_cols = ['station_no', 'station_name', 'Date', 'lon', 'lat']

    all_possible_columns = list(set(master_df.columns.tolist() + new_data_df.columns.tolist()))
    ts_cols = [col for col in all_possible_columns if col not in metadata_cols]

    if not master_df.empty:
        updated_rows_list = []

        master_dict = master_df.set_index(merge_keys).to_dict('index')

        master_df_keys = master_df[merge_keys].copy()
        master_df_keys['Date'] = pd.to_datetime(master_df_keys['Date'], errors='coerce').dt.date
        master_df_keys = master_df_keys.dropna(subset=['Date'])

        new_data_df_keys = new_data_df[merge_keys].copy()
        new_data_df_keys['Date'] = pd.to_datetime(new_data_df_keys['Date'], errors='coerce').dt.date
        new_data_df_keys = new_data_df_keys.dropna(subset=['Date'])

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

            new_row_df = new_data_df[(new_data_df['station_no'] == stn) & (new_data_df['Date'] == dt)]
            new_scrape_row_data = new_row_df.iloc[0].to_dict() if not new_row_df.empty else None

            current_row_data = {col: None for col in all_possible_columns}
            current_row_data['is_revised'] = False

            if existing_master_row_data is not None:
                current_row_data.update(existing_master_row_data)

            if new_scrape_row_data is not None:
                current_row_data.update(new_scrape_row_data)

            current_row_data['station_no'] = stn
            current_row_data['Date'] = dt

            is_row_revised = False

            # --- Update Time-Series Columns ---
            for col in ts_cols:
                old_val = current_row_data.get(col)
                new_val = new_scrape_row_data.get(col) if new_scrape_row_data is not None else None

                if pd.isna(old_val) and pd.notna(new_val):
                    current_row_data[col] = new_val
                    is_row_revised = True

            if new_scrape_row_data is not None:
                current_row_data['station_name'] = new_scrape_row_data.get('station_name', current_row_data.get('station_name'))
                current_row_data['lon'] = new_scrape_row_data.get('lon', current_row_data.get('lon'))
                current_row_data['lat'] = new_scrape_row_data.get('lat', current_row_data.get('lat'))

            current_row_data['is_revised'] = is_row_revised
            updated_rows_list.append(current_row_data)

        updated_df = pd.DataFrame(updated_rows_list)

        updated_df['Date'] = pd.to_datetime(updated_df['Date'], errors='coerce').dt.date

        for col in ts_cols:
            if col in updated_df.columns:
                updated_df[col] = pd.to_numeric(updated_df[col], errors='coerce')
            else:
                updated_df[col] = pd.NA

        for col in ['station_no', 'station_name', 'lon', 'lat', 'is_revised']:
            if col not in updated_df.columns:
                updated_df[col] = pd.NA

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
    # ##DISABLED FOR STORAGE SPACE##
    # This creates a new parquet every day and will balloon repo/storage over time.
    #
    # if not master_df.empty and 'Date' in master_df.columns and not master_df['Date'].isnull().all():
    #     iday = master_df['Date'].max()
    #     daily_snapshot_df = master_df[master_df['Date'] == iday]
    #     daily_parquet_path = f"data/AB_WS_R_Flows_{iday}.parquet"
    #     daily_snapshot_df.to_parquet(daily_parquet_path, index=False, engine="pyarrow")
    #     print(f"Daily snapshot Parquet saved to {daily_parquet_path}")
    # else:
    #     print("Master dataset is empty or 'Date' column is problematic, no daily snapshot saved.")

else:
    print("No new data collected from any stations.")

#############################Stitch#############################
from datetime import date
import pandas as pd
import json
import os

# If you want geopandas later, you can keep it installed,
# but we are disabling the parquet export from GeoDataFrame for storage.
# import geopandas as gpd  # ##DISABLED FOR STORAGE SPACE##

# --- Paths & Date ---
iday = date.today().strftime('%Y-%m-%d')
station_list_csv = "data/AB_WS_R_StationList.csv"
master_parquet_path = "data/WS_R_master_daily.parquet"     # rolling appended master daily (keep)
master_geojson_path = "data/AB_WS_R_stations.geojson"      # rolling GeoJSON (app uses THIS)

# Daily geojson snapshot (disabled)
geojson_updated_path = f"data/AB_WS_R_stations_{iday}.geojson"  # ##DISABLED FOR STORAGE SPACE##

# Parquet for app (disabled; app reads geojson)
output_app_parquet_path = "data/AB_WS_R_stations.parquet"       # ##DISABLED FOR STORAGE SPACE##

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
            "geometry": {"type": "Point", "coordinates": [row['LON'], row['LAT']]},
            "properties": {
                "station_no": row['WSC'],
                "station_name": None,
                "time_series": []
            }
        }
        geojson['features'].append(feature)
    print("✅ GeoJSON skeleton created.")

# --- Load Full Master Daily Data (source of timeseries) ---
master_df = pd.read_parquet(master_parquet_path, engine="pyarrow")
master_df['station_no'] = master_df['station_no'].astype(str).str.strip()

# --- App-only whitelist: keep ONLY what the Streamlit app reads/uses ---
TS_KEEP = [
    # Flows
    "Daily flow",
    "Calculated flow",

    # SWA thresholds (you plot Q80/Q90/Q95; compliance uses Q80 & Q95)
    "Q80",
    "Q90",
    "Q95",

    # WMP thresholds (your extract_thresholds() keyset)
    "WCO",
    "IO",
    "Minimum flow",
    "Industrial IO",
    "Non-industrial IO",
    "IFN",

    # Optional small flag
    "is_revised",
]

# Metadata stored on feature properties (not per-timeseries entry)
metadata_cols = {"station_no", "station_name", "Date", "lon", "lat"}

# Only keep columns that exist in master_df
ts_cols = [c for c in TS_KEEP if c in master_df.columns]

# --- Build a dict for fast feature lookup ---
feature_map = {feat['properties']['station_no']: feat for feat in geojson['features']}

# --- Update features with reduced timeseries payload ---
for stn_id, group_df in master_df.groupby('station_no'):
    if stn_id not in feature_map:
        continue
    feat = feature_map[stn_id]

    # Clean out all keys except static metadata & time_series to avoid garbage data
    static_keys = {'station_no', 'station_name', 'lat', 'lon', 'time_series'}
    feat['properties'] = {k: v for k, v in feat['properties'].items() if k in static_keys}

    # Ensure group is sorted by Date (so "latest_record" is truly latest)
    if 'Date' in group_df.columns:
        group_df = group_df.copy()
        group_df['Date'] = pd.to_datetime(group_df['Date'], errors='coerce')
        group_df = group_df.dropna(subset=['Date']).sort_values('Date')

    if group_df.empty:
        continue

    latest_record = group_df.iloc[-1]

    feat['properties']['station_no'] = stn_id
    feat['properties']['station_name'] = str(latest_record.get('station_name', stn_id))
    feat['properties']['lat'] = float(latest_record.get('lat', feat['properties'].get('lat', 0)))
    feat['properties']['lon'] = float(latest_record.get('lon', feat['properties'].get('lon', 0)))

    # Build reduced timeseries list (ONLY what app needs)
    timeseries = []
    for _, row in group_df.iterrows():
        # Date formatting
        d = row.get('Date', None)
        if pd.isna(d):
            continue
        d_str = pd.to_datetime(d).strftime('%Y-%m-%d')

        ts_entry = {'date': d_str}

        # Add only whitelisted columns if present + non-null
        for col in ts_cols:
            val = row.get(col)
            if pd.notnull(val) and str(val).strip() != '':
                if col == 'is_revised':
                    ts_entry[col] = bool(val)
                else:
                    try:
                        ts_entry[col] = float(val)
                    except (ValueError, TypeError):
                        # If something weird slips through, just skip
                        pass

        timeseries.append(ts_entry)

    feat['properties']['time_series'] = timeseries

# --- Save Rolling Master GeoJSON (app reads THIS) ---
with open(master_geojson_path, 'w') as f:
    json.dump(geojson, f, indent=2)
print(f"🔁 Rolling master GeoJSON updated: {master_geojson_path}")

# --- Save Daily GeoJSON Snapshot (disabled) ---
# ##DISABLED FOR STORAGE SPACE##
# This creates a new geojson every day and will balloon repo/storage over time.
#
# with open(geojson_updated_path, 'w') as f:
#     json.dump(geojson, f, indent=2)
# print(f"📍 Daily GeoJSON snapshot saved: {geojson_updated_path}")

# --- Convert GeoJSON to GeoDataFrame and save Parquet (disabled) ---
# ##DISABLED FOR STORAGE SPACE##
# App reads the rolling geojson directly; exporting parquet adds another big artifact.
#
# try:
#     stations_gdf = gpd.GeoDataFrame.from_features(geojson['features'])
#     stations_gdf.to_parquet(output_app_parquet_path, index=False, engine="pyarrow")
#     print(f"✅ Stations GeoDataFrame saved to {output_app_parquet_path}")
# except Exception as e:
#     print(f"❌ Error saving GeoDataFrame to Parquet: {e}")
