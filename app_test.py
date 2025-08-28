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

# --- Mobile Detection ---
# Get screen width using JavaScript evaluation for responsive layout
# Only run once at the start, or on specific events to avoid constant re-evaluation
# --- Initializing session state variables early ---
if 'map_height_pixels' not in st.session_state:
    st.session_state.map_height_pixels = 1200 # Default to desktop height

if 'selected_station' not in st.session_state:
    st.session_state.selected_station = None

if 'show_station_details_expander' not in st.session_state:
    st.session_state.show_station_details_expander = False

def sync_url_to_session():
    query_params = st.query_params  # <-- changed here
    station_from_url = query_params.get('station', [None])[0]
    if station_from_url:
        if st.session_state.selected_station != station_from_url.strip().upper():
            st.session_state.selected_station = station_from_url.strip().upper()
            st.experimental_rerun()
# --- Get screen width using streamlit_js_eval ---
# Do this early so is_mobile is determined before layout decisions
# Add a key to streamlit_js_eval for stability
browser_width = streamlit_js_eval(js_expressions='screen.width', want_output=True, key='browser_width_eval')

# Set is_mobile based on the returned width, or default to False if None (initial load)
if browser_width is not None:
    is_mobile = browser_width < 768
else:
    is_mobile = False # Default to desktop layout if width is not yet available
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
        # Normalize keys
    diversion_tables = {k.strip().upper(): v for k, v in diversion_tables.items()}
    diversion_labels = {k.strip().upper(): v for k, v in diversion_labels.items()}

    return diversion_tables, diversion_labels
    stations_with_diversion = set(diversion_tables.keys())

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

    if stream_size == 'Large':
        return 'green' if flow > q80 else 'yellow'

    elif stream_size == 'Medium':
        if q95 is None or pd.isna(q95):
            return 'gray'
        if flow > q80:
            return 'green'
        elif flow > q95:
            return 'yellow'
        else:
            return 'red'

    elif stream_size == 'Small':
        return 'green' if flow > q80 else 'red'

    return 'gray'

def compliance_color_diversion(flow, thresholds):
    if flow is None or pd.isna(flow):
        return 'gray'

    cb1 = thresholds.get('Cutback1')
    cb2 = thresholds.get('Cutback2')
    cb3 = thresholds.get('Cutback3') or thresholds.get('Cutoff')

    # Ensure at least one threshold exists
    if cb1 is None and cb2 is None and cb3 is None:
        return 'gray'

    try:
        if cb1 is not None and flow > cb1:
            return 'green'
        elif cb2 is not None and flow > cb2:
            return 'yellow'
        elif cb3 is not None and flow > cb3:
            return 'orange'
        else:
            return 'red'
    except:
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
        This app is a prototye and currently under construction!
        This app dynamically loads in station data and renders color for the markers as you scroll around. You may experience some loading and re-loading as you are viewing it. 
    </div>
    <div style='margin-top: 8px;'>
        We're working on making this less disruptive and more responsive. Thanks for your patience! - Lyndsay Greenwood
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
    **What is this?** This tool visualizes flow data from Alberta water stations and evaluates compliance with flow thresholds used in water policy decisions.

    **Data Sources:** - **Hydrometric data** and  **Diversion thresholds** from Alberta River Basins Water Conservation layer (Rivers.alberta.ca)
    - Alberta has over 400 hydrometric stations operated by both the Alberta provincial government and the federal Water Survey of Canada, which provides near real time flow and water level monitoring data. For the purpose of this app, flow in meters cubed per second is used.
    - **Diversion Tables** data from current provincial policy and regulations - use layer toggles on the right to swap between diversion tables and other thresholds for available stations, toggle will appear above charts with diversion table data to swap between cutbacks and quartiles.

    ** Threshold Definitions:** 
    - **WCO (Water Conservation Objective):** Target flow for ecosystem protection - sometimes represented as a percentage of "Natural Flow" (ie 45%), which is a theoretical value depicting what the flow of a system would be if there were no diversions
    - **IO (In-stream Objective):** Minimum flow below which withdrawals are restricted  
    - **IFN (In-stream Flow Need):** Ecological flow requirement for sensitive systems  
    - **Q80/Q95:** Statistical low flows based on historical comparisons;
    - Q80: Flow is exceeded 80% of the time historically - often used as a benchmark for the low end of "typical flow".  
    - Q90: The flow value exceeded 90% of the time. This means the river flow historically is above this level 90% of the time—representing a more extreme low flow than Q80.
    - Q95: The flow exceeded 95% of the time, meaning the river is flowing above this very low level 95% of the time.  This is often considered a critical threshold for ecological health.
    - **Cutbacks 1/2/3:** Phased reduction thresholds for diversions - can represent cutbacks in rate of diversion per second, daily limits or complete cutoff of diversion

    **🟢 Color Codes in Map:** - 🟢 Flow meets all thresholds  
    - 🔴 Flow below one or more thresholds  
    - 🟡 Intermediate - Flow may be below one threshold but above others (depends on stream size & Q-values)  
    - ⚪ Missing or insufficient data
    - 🔵 **Blue border**: Station has a Diversion Table (click toggle on right above data table for additional thresholds)

    _🚧 This app is under development. Thanks for your patience — and coffee! ☕ - Lyndsay Greenwood_
    """)
with st.sidebar.expander("ℹ️ Who Cares?"):
    st.markdown("""
    **❓ Why does this matter❓** 
    
    Water is a shared resource, and limits on it's extraction and use must exist to ensure fair and equitable access. Water is essential to environmental health, human life, and economic prosperity.  
    Wwater supply, however, is variable—and increasingly under pressure from many angles: natural seasonal fluctuations, shifting climate and weather patterns, and changing socio-economic factors such as population growth and energy demand.
    
    In Alberta, many industries—from agriculture and manufacturing to energy production and resource extraction—depend heavily on water. Setting clear limits and thresholds on water diversions helps protect our waterways from overuse by establishing enforceable cutoffs. These limits are often written directly into water diversion licenses issued by the provincial government.
    
    While water conservation is a responsibility we all share, ensuring that diversion limits exist—and are respected—is a vital tool in protecting Alberta’s water systems and ecosystems for generations to come.

    """)


import hashlib

def make_dates_hash(selected_dates):
    return hashlib.md5(str(selected_dates).encode()).hexdigest()

current_dates_hash = make_dates_hash(selected_dates)

import hashlib



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




def render_map_clickable(merged, selected_dates):
    mean_lat = merged['lat'].mean() if 'lat' in merged.columns else merged['LAT'].mean()
    mean_lon = merged['lon'].mean() if 'lon' in merged.columns else merged['LON'].mean()

     # Adjust map height based on mobile detection
    map_height_pixels = 300 if is_mobile else 1200
    m = folium.Map(location=[50.5, -114], zoom_start=6, width='100%', height=f'{map_height_pixels}px')
    st.session_state.map_height_pixels = map_height_pixels # Store in session state for st_folium
    Fullscreen().add_to(m)

    fg_all = folium.FeatureGroup(name='All Stations')
    fg_diversion = folium.FeatureGroup(name='Diversion Stations')

    for _, row in merged.iterrows():
        coords = [row['LAT'], row['LON']]
        wsc = row['WSC'].strip().upper()
        station_name = row.get("station_name", "Unknown Station")  # Add a default fallback
        date = get_most_recent_valid_date(row, selected_dates)
        compliance_color = get_color_for_date(row, date)
        tooltip_text = f"{wsc} {station_name}"

        border_color = 'blue' if wsc in diversion_tables else 'black'
        # Marker with tooltip only (station code) — NO popup here
        marker = folium.CircleMarker(
            location=coords,
            radius=7,
            color=border_color,
            weight=3,
            fill=True,
            fill_color=compliance_color,
            fill_opacity=0.7,
            tooltip=tooltip_text
        )
        marker.add_to(fg_all)

        if wsc in diversion_tables:
            marker2 = folium.CircleMarker(
                location=coords,
                radius=7,
                color='blue',
                weight=3,
                fill=True,
                fill_color=compliance_color,
                fill_opacity=0.7,
                tooltip=tooltip_text
            )
            marker2.add_to(fg_diversion)

    fg_all.add_to(m)
    fg_diversion.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m

# --- Plotly chart function for selected station ---

@st.cache_data
def has_diversion(wsc):
    return wsc.strip().upper() in diversion_tables  # or use stations_with_diversion if precomputed

if 'clicked_station' in st.session_state:
    row = st.session_state['clicked_station']
    wsc = row['WSC'].strip().upper()

    st.subheader(row.get('station_name', wsc))

    # Show toggle ONLY if diversion data exists for this station

    if has_diversion(wsc):
        st.write(f"Diversion data available for {wsc}")
        show_div = st.toggle("Show diversion thresholds", value=False)
    else:
        st.write(f"No diversion data for {wsc}")
        show_div = False

    # Render table and chart, passing show_div to control diversion display
    st.markdown(render_station_table(row, selected_dates, show_diversion=show_div), unsafe_allow_html=True)
    plot_station_chart(wsc, merged, selected_dates, show_diversion=show_div)


def plot_station_chart(wsc, merged, selected_dates, show_diversion=False):
    row = merged[merged['WSC'].str.strip().str.upper() == wsc]
    if row.empty:
        st.write("Station not found.")
        return
    row = row.iloc[0]

    dates = pd.to_datetime(selected_dates)
    flows = []
    thresholds_list = []

    for d in selected_dates:
        daily = extract_daily_data(row['time_series'], d)
        # Use daily flow if present else calculated flow
        flow = daily.get('Daily flow')
        if flow is None or pd.isna(flow):
            flow = daily.get('Calculated flow')
        flows.append(flow)

        # Extract thresholds for this date depending on policy
        if show_diversion and wsc in diversion_tables:
            diversion_df = diversion_tables[wsc]
            target_day = pd.to_datetime(d).replace(year=1900).normalize()
            div_row = diversion_df[diversion_df['Date'] == target_day]
            if not div_row.empty:
                div = div_row.iloc[0]
                third_label = diversion_labels.get(wsc, 'Cutback3')
                thresholds = {
                    'Cutback1': div.get('Cutback1', None),
                    'Cutback2': div.get('Cutback2', None),
                    third_label: div.get(third_label, None)
                }
            else:
                thresholds = {}
        else:
            # For non-diversion stations, fallback thresholds by policy
            if row['PolicyType'] == 'WMP':
                thresholds = extract_thresholds(daily)
            elif row['PolicyType'] == 'SWA':
                thresholds = {k: daily.get(k) for k in ['Q80', 'Q90', 'Q95']}
            else:
                thresholds = {}

        thresholds_list.append(thresholds)

        threshold_colors = {
        'Cutback1': 'yellow',
        'Cutback2': 'orange',
        'Cutback3': 'red',
        'Cutoff': 'red',
        'IO': 'red',
        'WCO': 'orange',
        'Q80': 'green',
        'Q90': 'yellow',
        'Q95': 'red',
        'Minimum flow': 'red',
        'Industrial IO': 'yellow',
        'Non-industrial IO': 'orange',
        'IFN': 'orange',
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=flows, mode='lines+markers', name='Flow'))


    # Gather all unique threshold labels across dates
    all_threshold_labels = set()
    for thres in thresholds_list:
        all_threshold_labels.update(thres.keys())
    all_threshold_labels = sorted(all_threshold_labels)

    # For each threshold, plot a line with values across dates
    for label in all_threshold_labels:
        vals = [thres.get(label, None) if thres else None for thres in thresholds_list]
        if any(v is not None and not pd.isna(v) for v in vals):
            color = threshold_colors.get(label, 'gray')  # default to gray if unknown
            fig.add_trace(go.Scatter(
                x=dates,
                y=vals,
                mode='lines+markers',
                name=label,
                line=dict(color=color)
            ))



    st.plotly_chart(fig, use_container_width=True)

# --- Layout ---

st.title("Alberta Flow Threshold Viewer")
def get_text_color(bg_color):
    c = str(bg_color).lower()
    # Define which backgrounds should have black text for contrast
    light_colors = ['yellow', 'gold', '#ffff00', '#ffd700', 'lightyellow']
    if c in light_colors or c.startswith('#ff'):
        return 'black'
    return 'white'

def render_station_table(row, selected_dates, show_diversion=False):
    policy = row.get('PolicyType', '')
    stream_size = row.get('StreamSize', '')

    flows, calc_flows = [], []
    daily_colors, calc_colors = [], []
    threshold_sets = []
    threshold_labels = set()

    selected_dates = sorted(selected_dates, key=pd.to_datetime)

    for d in selected_dates:
        daily = extract_daily_data(row['time_series'], d)
        df = daily.get('Daily flow')
        cf = daily.get('Calculated flow')

        flows.append(df)
        calc_flows.append(cf)

        wsc = row.get("WSC", "").strip().upper()
        if show_diversion and wsc in diversion_tables:
            diversion_df = diversion_tables[wsc]
            target_day = pd.to_datetime(d).replace(year=1900).normalize()
            div_row = diversion_df[diversion_df['Date'] == target_day]
            if not div_row.empty:
                div = div_row.iloc[0]
                third_label = diversion_labels.get(wsc, 'Cutback3')
                thresholds = {
                    'Cutback1': div.get('Cutback1', None),
                    'Cutback2': div.get('Cutback2', None),
                    third_label: div.get(third_label, None)
                }
            else:
                thresholds = {}
        elif policy == 'WMP':
            thresholds = extract_thresholds(daily)
        elif policy == 'SWA':
            thresholds = {k: daily.get(k) for k in ['Q80', 'Q90', 'Q95']}
        else:
            thresholds = {}

        threshold_sets.append(thresholds)
        threshold_labels.update(thresholds.keys())

        if show_diversion and wsc in diversion_tables:
            daily_color = compliance_color_diversion(df, thresholds)
            calc_color = compliance_color_diversion(cf, thresholds)
        else:
            daily_color = (
                compliance_color_SWA(stream_size, df, daily.get('Q80'), daily.get('Q95'))
                if policy == 'SWA' else compliance_color_WMP(df, thresholds)
            )
            calc_color = (
                compliance_color_SWA(stream_size, cf, daily.get('Q80'), daily.get('Q95'))
                if policy == 'SWA' else compliance_color_WMP(cf, thresholds)
            )
        
        daily_colors.append(daily_color)
        calc_colors.append(calc_color)

    threshold_labels = sorted(threshold_labels)

    html = f"<h4>{row['station_name']}</h4>"

    # Add scrollable container for wide tables
    html += """
    <div style="overflow-x: auto; max-width: 100%;">
      <table style='border-collapse: collapse; width: max-content; border: 1px solid black;'>
    """

    # Header row with dates
    html += "<tr><th style='padding: 6px; border: 1px solid black;'>Metric (m³/s)</th>"
    html += ''.join([f"<th style='padding: 6px; border: 1px solid black;'>{d}</th>" for d in selected_dates])
    html += "</tr>"

    # Daily Flow row
    if any(pd.notna(v) for v in flows):
        html += "<tr><td style='padding: 6px; border: 1px solid black; font-weight: bold;'>Daily Flow</td>"
        for val, color in zip(flows, daily_colors):
            display_val = f"{val:.2f}" if pd.notna(val) else "NA"
            text_color = get_text_color(color)
            html += f"<td style='padding: 6px; border: 1px solid black; background-color: {color}; color: {text_color}; text-align: center;'>{display_val}</td>"
        html += "</tr>"

    # Calculated Flow row
    if any(pd.notna(v) for v in calc_flows):
        html += "<tr><td style='padding: 6px; border: 1px solid black; font-weight: bold;'>Calculated Flow</td>"
        for val, color in zip(calc_flows, calc_colors):
            display_val = f"{val:.2f}" if pd.notna(val) else "NA"
            text_color = get_text_color(color)
            html += f"<td style='padding: 6px; border: 1px solid black; background-color: {color}; color: {text_color}; text-align: center;'>{display_val}</td>"
        html += "</tr>"

    # Threshold rows (no color)
    for label in threshold_labels:
        html += f"<tr><td style='padding: 6px; border: 1px solid black; font-weight: bold;'>{label}</td>"
        for thresholds in threshold_sets:
            val = thresholds.get(label) if thresholds else None
            display_val = f"{val:.2f}" if pd.notna(val) else "NA"
            html += f"<td style='padding: 6px; border: 1px solid black; text-align: center;'>{display_val}</td>"
        html += "</tr>"

    html += "</table></div>"

    return html
    st.markdown(html, unsafe_allow_html=True)

if is_mobile:
    # --- Mobile layout ---
    st.markdown("#### Interactive Map")
    st.markdown(
        "<small>⬅️ Select desired date range / view map info<br>⬇️ Click a station or enter a station number to view details</small>",
        unsafe_allow_html=True
    )

    manual_wsc = st.text_input("Enter station number:", key="manual_wsc_input_mobile")
    if manual_wsc:
        st.session_state.selected_station = manual_wsc.strip().upper()
        st.session_state.show_station_details_expander = True

    st.markdown("---")

    if 'show_station_details_expander' not in st.session_state:
        st.session_state.show_station_details_expander = False

    m = render_map_clickable(merged, selected_dates)
    clicked_data = st_folium(
        m,
        height=st.session_state.map_height_pixels,
        use_container_width=True,
        key="mobile_folium_map"
    )

    if clicked_data and clicked_data.get('last_object_clicked_tooltip'):
        tooltip_text = clicked_data['last_object_clicked_tooltip']
        if tooltip_text:
            selected_wsc = tooltip_text.split(" ")[0].strip().upper()
            st.session_state.selected_station = selected_wsc
            st.session_state.show_station_details_expander = True

    st.markdown("---")

    with st.expander("Station Details", expanded=st.session_state.show_station_details_expander):
        if st.session_state.get('selected_station'):
            station_code = st.session_state.selected_station
            row = merged[merged['WSC'].str.strip().str.upper() == station_code]

            if not row.empty:
                row = row.iloc[0]
                has_div = station_code in diversion_tables
                if has_div:
                    toggle_key = f"show_diversion_{station_code}_mobile"
                    if toggle_key not in st.session_state:
                        st.session_state[toggle_key] = False
                    show_diversion = st.checkbox(
                        "Show diversion table thresholds",
                        value=st.session_state[toggle_key],
                        key=toggle_key
                    )
                else:
                    show_diversion = False

                html_table = render_station_table(row, selected_dates, show_diversion=show_diversion)
                st.markdown(html_table, unsafe_allow_html=True)
                plot_station_chart(station_code, merged, selected_dates, show_diversion=show_diversion)
            else:
                st.write("Station data not found.")
        else:
            st.write("Click a station on the map or enter a number above.")

        
 
else:
    # --- Desktop layout ---
    col1, col2 = st.columns([5, 2])
    
    with col1:
        st.markdown("### Interactive Map - Click a station or enter a station number below:")
    
        manual_input = st.text_input(
            "Enter station number:",
            key="manual_wsc_input_top"
        )
    
        # Render map
        m = render_map_clickable(merged, selected_dates)
        clicked_data = st_folium(
            m,
            height=1200,
            use_container_width=True,
            key="desktop_folium_map"
        )
    
        # Update selected_station from map click
        if clicked_data and clicked_data.get('last_object_clicked_tooltip'):
            tooltip_text = clicked_data['last_object_clicked_tooltip']
            if tooltip_text:
                st.session_state.selected_station = tooltip_text.split(" ")[0].strip().upper()
                st.session_state.show_station_details_expander = True
    
        # Update selected_station from manual input
        if manual_input:
            st.session_state.selected_station = manual_input.strip().upper()
            st.session_state.show_station_details_expander = True
    
    with col2:
        # Render the station table & chart
        if st.session_state.get('selected_station'):
            station_code = st.session_state.selected_station
            row = merged[merged['WSC'].str.strip().str.upper() == station_code]
    
            if not row.empty:
                row = row.iloc[0]
    
                # Diversion toggle
                has_div = station_code in diversion_tables
                toggle_key = f"show_diversion_{station_code}_desktop"
                if has_div and toggle_key not in st.session_state:
                    st.session_state[toggle_key] = False
                show_diversion = st.checkbox(
                    "Show diversion table thresholds",
                    value=st.session_state.get(toggle_key, False),
                    key=toggle_key
                ) if has_div else False
    
                # Render table and chart
                html_table = render_station_table(row, selected_dates, show_diversion=show_diversion)
                st.markdown(html_table, unsafe_allow_html=True)
                plot_station_chart(station_code, merged, selected_dates, show_diversion=show_diversion)
    
                # NOW clear input safely
                st.session_state.manual_wsc_input_top = ""
    
            else:
                st.write("Station data not found.")
        else:
            st.write("Click a station on the map or enter a station number above.")










