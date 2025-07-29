from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import geopandas as gpd
import json
import folium
from folium import IFrame
from folium.plugins import Fullscreen
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from dateutil.parser import parse
import os
from streamlit_js_eval import streamlit_js_eval
import plotly.graph_objects as go
from streamlit_folium import st_folium

st.cache_data.clear()
st.set_page_config(layout="wide")

# Get station selection from URL query params
query_params = st.query_params
selected_station = query_params.get('station', [None])[0]

if selected_station:
    st.session_state.selected_station = selected_station.strip().upper()
else:
    if 'selected_station' not in st.session_state:
        st.session_state.selected_station = None

def sync_url_to_session():
    query_params = st.query_params  # <-- changed here
    station_from_url = query_params.get('station', [None])[0]
    if station_from_url:
        if st.session_state.selected_station != station_from_url.strip().upper():
            st.session_state.selected_station = station_from_url.strip().upper()
            st.experimental_rerun()

sync_url_to_session()

# --- Paths ---
DATA_DIR = "data"
DIVERSION_DIR = os.path.join(DATA_DIR, "DiversionTables")
STREAM_CLASS_FILE = os.path.join(DATA_DIR, "StreamSizeClassification.csv")
CSV_FILE = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")], reverse=True)[0]

# --- Load data ---
def make_df_hashable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list columns to tuples for Streamlit caching compatibility.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].apply(lambda x: isinstance(x, list)).any():
            df_copy[col] = df_copy[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    return df_copy

@st.cache_data
def load_data():
    station_list = pd.read_csv(os.path.join(DATA_DIR, "AB_WS_R_StationList.csv"))
    geo_data = gpd.read_file(os.path.join(DATA_DIR, "AB_WS_R_stations.geojson"))
    geo_data = geo_data.rename(columns={'station_no': 'WSC'})
    merged = pd.merge(station_list, geo_data, on='WSC', how='inner')

    def safe_parse(val):
        if isinstance(val, str):
            try:
                return json.loads(val)
            except:
                return []
        return val

    merged['time_series'] = merged['time_series'].apply(safe_parse)
    merged = make_df_hashable(merged)  # <-- Add this line here
    merged = make_df_hashable(merged)  # <-- call here
    return merged

# Call load_data and assign merged here
merged = load_data()

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

if not valid_dates:
    st.error("No valid flow data found. Please check your data files.")
    st.stop()



# --- Load diversion tables ---
@st.cache_data
def load_diversion_tables():
    diversion_tables = {}
    diversion_labels = {}

    for f in os.listdir(DIVERSION_DIR):
        if f.endswith(".parquet"):
            wsc = f.split("_")[0]
            file_path = os.path.join(DIVERSION_DIR, f)
            df = pd.read_parquet(file_path)

            # Rename the first three columns
            standard_columns = ['Date', 'Cutback1', 'Cutback2']
            if len(df.columns) == 4:
                # Preserve whatever the third cutback label is
                third_label = df.columns[3]
                df.columns = standard_columns + [third_label]
                diversion_labels[wsc] = third_label
            else:
                # Fallback if format isn't as expected
                diversion_labels[wsc] = "Cutback3"
                df.columns = standard_columns + ['Cutback3']

            # Normalize and fix date format
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()

            def safe_replace_year(d):
                try:
                    if isinstance(d, (pd.Timestamp, datetime)) and pd.notna(d):
                        return d.replace(year=1900)
                    else:
                        return pd.NaT
                except:
                    return pd.NaT

            df['Date'] = df['Date'].apply(safe_replace_year)

            diversion_tables[wsc] = df
        # ✅ Normalize keys
    diversion_tables = {k.strip().upper(): v for k, v in diversion_tables.items()}
    diversion_labels = {k.strip().upper(): v for k, v in diversion_labels.items()}

    return diversion_tables, diversion_labels

with st.spinner("Loading... this may take a few minutes"):
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
    

# --- Streamlit Sidebar Elements ---
with st.sidebar.expander("🚨 Note from Developer", expanded=False):
    st.markdown("""
    <div style='color: red'>
        This app pre-computes charts and tables for all stations before displaying the map.  
        That means loading can take **2-3 minutes**, depending on your date range and device.
    </div>
    <div style='margin-top: 8px;'>
        We're working on making this faster and more responsive. Thanks for your patience!
    </div>
    """, unsafe_allow_html=True)

st.sidebar.header("Date Range")
# Ensure min_date and max_date are actual date objects from the valid_dates string list
min_date = datetime.strptime(valid_dates[0], "%Y-%m-%d").date()
max_date = datetime.strptime(valid_dates[-1], "%Y-%m-%d").date()
st.sidebar.write(f"Current data collection range: {min_date} to {max_date}")
start_date = st.sidebar.date_input("Start", value=max_date - timedelta(days=7), min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End", value=max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Filter `valid_dates` (which are strings) based on the selected `start_date` and `end_date` (date objects)
selected_dates = [d for d in valid_dates if start_date.strftime('%Y-%m-%d') <= d <= end_date.strftime('%Y-%m-%d')]

with st.sidebar.expander("ℹ️ About this App"):
    st.markdown("""
    **🔍 What is this?** This tool visualizes flow data from Alberta water stations and evaluates compliance with flow thresholds used in water policy decisions.

    **📊 Data Sources:** - **Hydrometric data** and  **Diversion thresholds** from Alberta River Basins Water Conservation layer (Rivers.alberta.ca)
    - Alberta has over 400 hydrometric stations operated by both the Alberta provincial government and the federal Water Survey of Canada, which provides near real time flow and water level monitoring data. For the purpose of this app, flow in meters cubed per second is used.
    - **Diversion Tables** from current provincial policy and regulations - use layer toggles on the right to swap between diversion tables and other thresholds for available stations.
    - **Stream size and policy type** from Alberta Environment and Protected Areas and local (Survace Water Allocation Directive) and local jurisdictions (Water Management Plans)

    **📏 Threshold Definitions:** - **WCO (Water Conservation Objective):** Target flow for ecosystem protection - sometimes represented as a percentage of "Natural Flow" (ie 45%), which is a theoretical value depicting what the flow of a system would be if there were no diversions
    - **IO (Instream Objective):** Minimum flow below which withdrawals are restricted  
    - **IFN (Instream Flow Need):** Ecological flow requirement for sensitive systems  
    - **Q80/Q95:** Statistical low flows based on historical comparisons; Q80 means flow is exceeded 80% of the time - often used as a benchmark for the low end of "typical flow".  
    - Q90: The flow value exceeded 90% of the time. This means the river flow is above this level 90% of the time—representing a more extreme low flow than Q80.
    - Q95: The flow exceeded 95% of the time, meaning the river is flowing above this very low level 95% of the time.  This is often considered a critical threshold for ecological health.
    - **Cutbacks 1/2/3:** Phased reduction thresholds for diversions - can represent cutbacks in rate of diversion or daily limits

    **🟢 Color Codes in Map:** - 🟢 Flow meets all thresholds  
    - 🔴 Flow below one or more thresholds  
    - 🟡 Intermediate (depends on stream size & Q-values)  
    - ⚪ Missing or insufficient data
    - 🔵 **Blue border**: Station has a Diversion Table (click layer on right for additional thresholds)

    _🚧 This app is under development. Thanks for your patience — and coffee! ☕ - Lyndsay Greenwood_
    """)
with st.sidebar.expander("ℹ️ Who Cares?"):
    st.markdown("""
    **❓ Why does this matter?** Water is a shared resource, and limits must exist to ensure fair and equitable access. It is essential to environmental health, human life, and economic prosperity.  
    However, water supply is variable—and increasingly under pressure from many angles: natural seasonal fluctuations, shifting climate and weather patterns, and changing socio-economic factors such as population growth and energy demand.
    
    In Alberta, many industries—from agriculture and manufacturing to energy production and resource extraction—depend heavily on water. Setting clear limits and thresholds on water diversions helps protect our waterways from overuse by establishing enforceable cutoffs. These limits are often written directly into water diversion licenses issued by the provincial government.
    
    While water conservation is a personal responsibility we all share, ensuring that diversion limits exist—and are respected—is a vital tool in protecting Alberta’s water systems and ecosystems for generations to come.

    """)


import hashlib

def make_dates_hash(selected_dates):
    return hashlib.md5(str(selected_dates).encode()).hexdigest()

current_dates_hash = make_dates_hash(selected_dates)

def make_popup_html(row, selected_dates, show_diversion):
    font_size = '16px'
    padding = '6px'
    border = '2px solid black'

    flows, calc_flows = [], []
    threshold_sets = []
    threshold_labels = set()
    show_daily_flow = show_calc_flow = False

    selected_dates = sorted(selected_dates, key=pd.to_datetime)

    for d in selected_dates:
        daily = extract_daily_data(row['time_series'], d)
        df = daily.get('Daily flow', float('nan'))
        cf = daily.get('Calculated flow', float('nan'))
        flows.append(df)
        calc_flows.append(cf)

        if pd.notna(df): show_daily_flow = True
        if pd.notna(cf): show_calc_flow = True

        if show_diversion and row['WSC'] in diversion_tables:
            diversion_df = diversion_tables[row['WSC']]
            target_day = pd.to_datetime(d).replace(year=1900).normalize()
            div_row = diversion_df[diversion_df['Date'] == target_day]
            if not div_row.empty:
                div = div_row.iloc[0]
                third_label = diversion_labels.get(row['WSC'], 'Cutback3')
                thresholds = {
                    'Cutback1': div.get('Cutback1', float('nan')),
                    'Cutback2': div.get('Cutback2', float('nan')),
                    third_label: div.get(third_label, float('nan'))
                }
                threshold_sets.append(thresholds)
                threshold_labels.update(thresholds.keys())
                continue

        if row['PolicyType'] == 'WMP':
            thresholds = extract_thresholds(daily)
        elif row['PolicyType'] == 'SWA':
            thresholds = {k: daily.get(k, float('nan')) for k in ['Q80', 'Q90', 'Q95']}
        else:
            thresholds = {}

        threshold_sets.append(thresholds)
        threshold_labels.update(thresholds.keys())

    threshold_labels = sorted(threshold_labels)

    html = f"<div style='max-width: 100%;'><h4 style='font-size:{font_size};'>{row['station_name']}</h4>"
    html += f"<table style='border-collapse: collapse; border: {border}; font-size:{font_size};'>"
    html += "<tr><th style='padding:{0}; border:{1};'>Metric (m³/s)</th>{2}</tr>".format(
        padding, border,
        ''.join([f"<th style='padding:{padding}; border:{border};'>{d}</th>" for d in selected_dates])
    )

    if show_daily_flow:
        html += "<tr><td style='padding:{0}; border:{1}; font-weight:bold;'>Daily Flow</td>".format(padding, border)
        html += ''.join([
            f"<td style='padding:{padding}; border:{border};'>{f'{v:.2f}' if pd.notna(v) else 'NA'}</td>"
            for v in flows
        ])
        html += "</tr>"

    if show_calc_flow and any(pd.notna(val) for val in calc_flows):
        html += "<tr><td style='padding:{0}; border:{1}; font-weight:bold;'>Calculated Flow</td>".format(padding, border)
        html += ''.join([
            f"<td style='padding:{padding}; border:{border};'>{f'{v:.2f}' if pd.notna(v) else 'NA'}</td>"
            for v in calc_flows
        ])
        html += "</tr>"

    for label in threshold_labels:
        html += f"<tr><td style='padding:{padding}; border:{border}; font-weight:bold;'>{label}</td>"
        html += ''.join([
            f"<td style='padding:{padding}; border:{border};'>" + (f"{t.get(label):.2f}" if pd.notna(t.get(label)) else "NA") + "</td>"
            for t in threshold_sets
        ])
        html += "</tr>"

    html += "</table></div>"

    return html

import hashlib

def generate_all_popups(merged_df, selected_dates):
    popup_cache_no_diversion = {}
    popup_cache_diversion = {}

    for _, row in merged_df.iterrows():
        wsc = row['WSC']
        try:
            popup_cache_no_diversion[wsc] = make_popup_html(row, selected_dates, show_diversion=False)
            popup_cache_diversion[wsc] = make_popup_html(row, selected_dates, show_diversion=True)
        except Exception as e:
            st.exception(e)
            popup_cache_no_diversion[wsc] = "<p>Error generating popup</p>"
            popup_cache_diversion[wsc] = "<p>Error generating popup</p>"

    return popup_cache_no_diversion, popup_cache_diversion

def get_date_hash(dates):
    """Create a short unique hash string for a list of dates."""
    date_str = ",".join(sorted(dates))
    return hashlib.md5(date_str.encode()).hexdigest()

# Pre-generate both popup caches upfront

def get_most_recent_valid_date(row, dates):
    for d in sorted(dates, reverse=True):
        daily = extract_daily_data(row['time_series'], d)
        if any(pd.notna(daily.get(k)) for k in ['Daily flow', 'Calculated flow']):
            return d
    return None
if ('popup_cache_no_diversion' not in st.session_state or
    'popup_cache_diversion' not in st.session_state or
    st.session_state.get('cached_dates_hash', '') != current_dates_hash):

    with st.spinner("🚧 App is loading... Grab a coffee while we fire it up ☕"):
        no_diversion_cache, diversion_cache = generate_all_popups(merged, selected_dates)
        st.session_state.popup_cache_no_diversion = no_diversion_cache
        st.session_state.popup_cache_diversion = diversion_cache
        st.session_state.cached_dates_hash = current_dates_hash

else:
    cached_dates_hash = st.session_state.get('cached_dates_hash', '')
    if cached_dates_hash != current_dates_hash:
        with st.spinner("Updating popup caches for new date range..."):
            no_diversion_cache, diversion_cache = generate_all_popups(merged, selected_dates)
            st.session_state.popup_cache_no_diversion = no_diversion_cache
            st.session_state.popup_cache_diversion = diversion_cache
            st.session_state.cached_dates_hash = current_dates_hash

@st.cache_data(show_spinner=True)

def render_map_clickable(merged, selected_dates):
    mean_lat = merged['lat'].mean() if 'lat' in merged.columns else merged['LAT'].mean()
    mean_lon = merged['lon'].mean() if 'lon' in merged.columns else merged['LON'].mean()

    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6, width='100%', height='1000px')
    Fullscreen().add_to(m)

    fg_all = folium.FeatureGroup(name='All Stations')
    fg_diversion = folium.FeatureGroup(name='Diversion Stations')

    for _, row in merged.iterrows():
        coords = [row['LAT'], row['LON']]
        wsc = row['WSC'].strip().upper()

        date = get_most_recent_valid_date(row, selected_dates)
        color = get_color_for_date(row, date)

        # Determine border color based on diversion presence
        border_color = 'blue' if wsc in diversion_tables else 'black'
    
        marker = folium.CircleMarker(
            location=coords,
            radius=7,
            color=border_color,
            weight=3,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=row['station_name']
        )
        marker.add_to(fg_all)
    
        # JS for updating URL query param on click
        marker.add_child(folium.Element(f"""
            <script>
            var marker = {marker.get_name()};
            marker.on('click', function(e) {{
                const wsc = '{wsc}';
                const url = new URL(window.location);
                url.searchParams.set('station', wsc);
                window.history.pushState({{}}, '', url);
                window.dispatchEvent(new Event('popstate'));
            }});
            </script>
        """))

        # Marker for diversion stations (blue border)
        if wsc in diversion_tables:
            marker2 = folium.CircleMarker(
                location=coords,
                radius=7,
                color='blue',
                weight=3,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=row['station_name']
            )
            marker2.add_to(fg_diversion)

            marker2.add_child(folium.Element(f"""
                <script>
                var marker = {marker2.get_name()};
                marker.on('click', function(e) {{
                    const wsc = '{wsc}';
                    const url = new URL(window.location);
                    url.searchParams.set('station', wsc);
                    window.history.pushState({{}}, '', url);
                    window.dispatchEvent(new Event('popstate'));
                }});
                </script>
            """))

    fg_all.add_to(m)
    fg_diversion.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

# --- Plotly chart function for selected station ---

def plot_station_chart(wsc, merged, selected_dates):
    row = merged[merged['WSC'].str.strip().str.upper() == wsc]
    if row.empty:
        st.write("Station not found.")
        return
    row = row.iloc[0]

    dates = pd.to_datetime(selected_dates)
    flows = []
    calc_flows = []
    for d in selected_dates:
        daily = extract_daily_data(row['time_series'], d)
        flows.append(daily.get('Daily flow', None))
        calc_flows.append(daily.get('Calculated flow', None))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=flows, mode='lines+markers', name='Daily Flow'))
    fig.add_trace(go.Scatter(x=dates, y=calc_flows, mode='lines+markers', name='Calculated Flow'))

    fig.update_layout(title=f"Flow Data for {row['station_name']}",
                      xaxis_title='Date',
                      yaxis_title='Flow (m³/s)',
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- Layout ---

st.title("Alberta Flow Threshold Viewer")

col1, col2 = st.columns([2, 1])

with col1:
    # Create your folium map like before
    mean_lat = merged['LAT'].mean()
    mean_lon = merged['LON'].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6)

    for _, row in merged.iterrows():
        coords = [row['LAT'], row['LON']]
        wsc = row['WSC'].strip().upper()
        folium.CircleMarker(
            location=coords,
            radius=7,
            color='black',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            tooltip=row['station_name'],
            popup=wsc  # Use the WSC as popup to identify station
        ).add_to(m)

    # Render the map and capture click info
    clicked_data = st_folium(m, height=600)

with col2:
    # Handle clicks and display charts here
    if clicked_data and clicked_data.get('last_object_clicked'):
        selected_wsc = clicked_data['last_object_clicked'].get('popup')
        if selected_wsc:
            st.session_state.selected_station = selected_wsc
    if st.session_state.get('selected_station'):
        plot_station_chart(st.session_state.selected_station, merged, selected_dates)
    else:
        st.write("Click a station on the map to see its flow chart here.")
# Always compute the current hash


st.title("Alberta Flow Threshold Viewer")




