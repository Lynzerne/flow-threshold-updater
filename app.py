from datetime import datetime 
import streamlit as st
import pandas as pd
import geopandas as gpd
import json
from datetime import datetime, timedelta
import folium
from folium import IFrame
from folium.plugins import Fullscreen
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from dateutil.parser import parse
import os
st.cache_data.clear()
st.set_page_config(layout="wide")

# --- Paths ---
DATA_DIR = "data"
DIVERSION_DIR = os.path.join(DATA_DIR, "DiversionTables")
STREAM_CLASS_FILE = os.path.join(DATA_DIR, "StreamSizeClassification.csv")
CSV_FILE = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")],
    reverse=True
)[0]  # Use most recent CSV by name

# --- Load data ---
@st.cache_data
@st.cache_data
def load_data():
    # Load station list with required columns including WSC, PolicyType, StreamSize, LAT, LON
    station_list = pd.read_csv(os.path.join(DATA_DIR, "AB_WS_R_StationList.csv"))
    
    # Load geo data (make sure column names align for merging)
    geo_data = gpd.read_file(os.path.join(DATA_DIR, "AB_WS_R_stations.geojson"))
    geo_data = geo_data.rename(columns={'station_no': 'WSC'})  # rename for merge
    
    # Merge on 'WSC'
    merged = pd.merge(station_list, geo_data, on='WSC', how='inner')
    
    # Parse 'time_series' JSON strings safely
    def safe_parse(val):
        if isinstance(val, str):
            try:
                return json.loads(val)
            except:
                return []
        return val
    
    merged['time_series'] = merged['time_series'].apply(safe_parse)
    
    return merged
merged = load_data()

# --- Load diversion tables ---
@st.cache_data


def load_diversion_tables():
    diversion_tables = {}
    diversion_labels = {}

    for f in os.listdir(DIVERSION_DIR):
        if f.endswith(".xlsx"):
            wsc = f.split("_")[0]
            df = pd.read_excel(os.path.join(DIVERSION_DIR, f))
            df.columns = df.columns.str.strip()

            # Convert 'Date' column to datetime, coercing errors to NaT
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()

            # Debug: print any problematic date values
            print(f"Checking 'Date' column values in file: {f}")
            for d in df['Date']:
                if not (isinstance(d, (pd.Timestamp, datetime)) and pd.notna(d)):
                    print(f"Problematic date value: {d} (type: {type(d)})")

            # Safer date replacement with try-except to catch errors
            def safe_replace_year(d):
                try:
                    if isinstance(d, (pd.Timestamp, datetime)) and pd.notna(d):
                        return d.replace(year=1900)
                    else:
                        return pd.NaT
                except Exception as e:
                    print(f"Error replacing year in date {d}: {e}")
                    return pd.NaT

            df['Date'] = df['Date'].apply(safe_replace_year)

            diversion_tables[wsc] = df
            diversion_labels[wsc] = f"{wsc} diversion table"

    return diversion_tables, diversion_labels

diversion_tables, diversion_labels = load_diversion_tables()

# --- Helper functions ---
def extract_daily_data(time_series, date_str):
    for item in time_series:
        if item.get("date") == date_str:
            return item
    return {}

def extract_thresholds(entry):
    keys = {'WCO', 'IO', 'Minimum flow', 'Industrial IO', 'Non-industrial IO', 'IFN'}
    return {k: v for k, v in entry.items() if k in keys and v is not None}

def compliance_color_WMP(flow, thresholds):
    if flow is None or pd.isna(flow) or not thresholds:
        return 'gray'
    for val in thresholds.values():
        if val is not None and flow < val:
            return 'red'
    return 'green'

def compliance_color_SWA(stream_size, flow, q80, q95):
    if any(x is None or pd.isna(x) for x in [flow, q80]):
        return 'gray'
    if flow > q80:
        return 'green'
    if stream_size == 'Large':
        return 'yellow'
    elif stream_size == 'Medium':
        if q95 is None or pd.isna(q95):
            return 'gray'
        return 'yellow' if flow > q95 else 'red'
    elif stream_size == 'Small':
        return 'red'
    return 'gray'

def get_color_for_date(row, date):
    daily = extract_daily_data(row['time_series'], date)
    flow_daily = daily.get('Daily flow')
    flow_calc = daily.get('Calculated flow')

    flow = max(filter(pd.notna, [flow_daily, flow_calc]), default=None)

    policy = row['PolicyType']
    if policy == 'SWA':
        return compliance_color_SWA(row.get('StreamSize'), flow, daily.get('Q80'), daily.get('Q95'))
    elif policy == 'WMP':
        return compliance_color_WMP(flow, extract_thresholds(daily))
    return 'gray'

def get_valid_dates(merged):
    dates = set()
    for ts in merged['time_series']:
        for item in ts:
            if 'date' in item and 'Daily flow' in item:
                try:
                    d = parse(item['date']).strftime('%Y-%m-%d')
                    if item['Daily flow'] is not None:
                        dates.add(d)
                except:
                    pass
    return sorted(dates)

valid_dates = get_valid_dates(merged)

# --- Sidebar ---
st.sidebar.header("Date Range")
min_date = datetime.strptime(valid_dates[0], "%Y-%m-%d").date()
max_date = datetime.strptime(valid_dates[-1], "%Y-%m-%d").date()
start_date = st.sidebar.date_input("Start", value=max_date - timedelta(days=7), min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End", value=max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

selected_dates = [d for d in valid_dates if start_date.strftime('%Y-%m-%d') <= d <= end_date.strftime('%Y-%m-%d')]
show_diversion = st.sidebar.checkbox("Show Diversion Tables", value=True)

# --- Popup HTML builder ---
def make_popup_html_with_plot(row, selected_dates, show_diversion):
    # (you can paste your existing make_popup_html_with_plot function here, unchanged)
    return "<b>Popup here</b>"  # Placeholder for now

# --- Map rendering ---
def get_most_recent_valid_date(row, dates):
    for d in sorted(dates, reverse=True):
        daily = extract_daily_data(row['time_series'], d)
        if any(pd.notna(daily.get(k)) for k in ['Daily flow', 'Calculated flow']):
            return d
    return None

def render_map():
    # Use LAT and LON from AB_WS_R_StationList.csv columns for map center
    m = folium.Map(
        location=[merged['LAT'].mean(), merged['LON'].mean()],
        zoom_start=6,
        width='100%',
        height='100%'
    )
    Fullscreen().add_to(m)
    
    st.write("Merged columns:", merged.columns.tolist())  # for debug
    
    for _, row in merged.iterrows():
        coords = [row['LAT'], row['LON']]
        
        date = get_most_recent_valid_date(row, selected_dates)
        if not date:
            continue
        
        color = get_color_for_date(row, date)
        
        popup_html = make_popup_html_with_plot(row, selected_dates, show_diversion=False)
        iframe = IFrame(html=popup_html, width=700, height=500)
        popup = folium.Popup(iframe)
        
        folium.CircleMarker(
            location=coords,
            radius=7,
            color='black',
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup,
            tooltip=row['station_name']  # or adjust depending on your columns
        ).add_to(m)
    
    return m


# --- Display ---
st.title("Alberta Flow Threshold Viewer")
st.caption(f"Using data from: `{CSV_FILE}`")

if not selected_dates:
    st.warning("No data available for the selected date range.")
else:
    m = render_map()
    map_html = m._repr_html_()
    st.components.v1.html(map_html, height=800, scrolling=True)

