import streamlit as st
import pandas as pd
import geopandas as gpd
import json
from dateutil.parser import parse
import os
from datetime import datetime, date

# --- Paths ---
DATA_DIR = "data"
STATION_CSV = os.path.join(DATA_DIR, "AB_WS_R_StationList.csv")
GEOJSON_FILE = os.path.join(DATA_DIR, "AB_WS_R_stations.geojson")

@st.cache_data
def load_data():
    # Load CSV and GeoJSON
    station_list = pd.read_csv(STATION_CSV)
    geo_data = gpd.read_file(GEOJSON_FILE)

    # Rename to match merge key
    if 'station_no' in geo_data.columns:
        geo_data = geo_data.rename(columns={'station_no': 'WSC'})
    
    # Debug: print columns for verification
    st.write("station_list columns:", station_list.columns.tolist())
    st.write("geo_data columns:", geo_data.columns.tolist())

    # Check station_list columns to pick merge key
    if 'WSC' not in station_list.columns:
        st.error("station_list missing 'WSC' column for merge")
        st.stop()

    # Debug: Check raw dates in geojson before merge
    all_dates_geojson = set()
    for ts_str in geo_data['time_series']:
        try:
            ts = json.loads(ts_str) if isinstance(ts_str, str) else ts_str
            for item in ts:
                if 'date' in item:
                    d = parse(item['date']).date()
                    all_dates_geojson.add(d)
        except Exception as e:
            st.write(f"Error parsing time_series item in geojson: {e}")
    if all_dates_geojson:
        st.write(f"Earliest date in geojson time_series: {min(all_dates_geojson)}")
        st.write(f"Latest date in geojson time_series: {max(all_dates_geojson)}")
    else:
        st.write("No valid dates found in geojson time_series")

    # Merge on 'WSC'
    merged = pd.merge(station_list, geo_data, on='WSC', how='inner')

    # Parse time_series JSON strings safely
    def safe_parse(val):
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if not parsed:
                    st.write("Empty time_series JSON string found:", val)
                return parsed
            except Exception as e:
                st.write(f"JSON parse error in time_series: {e}, val: {val[:100]}")
                return []
        return val

    merged['time_series'] = merged['time_series'].apply(safe_parse)

    # Debug: print sample dates per station after parsing
    for idx, ts in enumerate(merged['time_series']):
        if not isinstance(ts, list):
            continue
        dates = []
        for item in ts:
            try:
                if 'date' in item:
                    dates.append(parse(item['date']).date())
            except Exception as e:
                st.write(f"Error parsing date in station time_series: {e}")
        if dates:
            st.write(f"Station {merged.iloc[idx]['WSC']} sample dates: {sorted(dates)[:5]} ... total dates: {len(dates)}")

    return merged

def get_valid_dates(merged):
    dates = set()
    for ts in merged['time_series']:
        if not isinstance(ts, list):
            continue
        for item in ts:
            if 'date' in item and 'Daily flow' in item:
                try:
                    flow = item['Daily flow']
                    if flow is not None and str(flow).strip() != '':
                        d = parse(item['date']).date()
                        dates.add(d)
                except Exception:
                    pass
    return sorted(dates)

# --- Main app ---

st.title("Date Range Test")

merged = load_data()

valid_dates = get_valid_dates(merged)

if not valid_dates:
    st.error("No valid flow dates found!")
    st.stop()

st.write(f"Total valid flow dates found: {len(valid_dates)}")
st.write(f"Earliest date: {valid_dates[0]}")
st.write(f"Latest date: {valid_dates[-1]}")

# Sidebar date pickers
min_date = valid_dates[0]
max_date = valid_dates[-1]

start_date = st.sidebar.date_input("Start date", value=max_date - pd.Timedelta(days=7), min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("Start date must be before or equal to End date")
    st.stop()

st.write(f"Selected date range: {start_date} to {end_date}")

# Filter the valid_dates within selected range
selected_dates = [d for d in valid_dates if start_date <= d <= end_date]
st.write(f"Dates in selection ({len(selected_dates)}):")
st.write(selected_dates)
