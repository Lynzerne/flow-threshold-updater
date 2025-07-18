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

# Clear cache only for development. Consider removing in production for performance.
st.cache_data.clear()
st.set_page_config(layout="wide")

# --- Paths ---
DATA_DIR = "data"
DIVERSION_DIR = os.path.join(DATA_DIR, "DiversionTables")
STREAM_CLASS_FILE = os.path.join(DATA_DIR, "StreamSizeClassification.csv")

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
    # Load spatial GeoData
    geo_data = gpd.read_parquet(os.path.join(DATA_DIR, "AB_WS_R_stations.parquet"))
    geo_data = geo_data.rename(columns={'station_no': 'WSC'})

    # Load station attributes from CSV (contains PolicyType, StreamSize, etc.)
    station_info = pd.read_csv(os.path.join(DATA_DIR, "AB_WS_R_StationList.csv"))

    # Merge in additional attributes
    geo_data = geo_data.merge(
        station_info[['WSC', 'PolicyType', 'StreamSize', 'LAT', 'LON']],
        on='WSC', how='left'
    )

    # Convert geometry to WKT (safe for caching)
    geo_data['geometry_wkt'] = geo_data.geometry.apply(lambda g: g.wkt if g else None)
    geo_data = geo_data.drop(columns=['geometry'])

    # Parse time_series safely
    def safe_parse(val):
        if isinstance(val, str):
            try:
                return json.loads(val)
            except:
                return []
        return val

    geo_data['time_series'] = geo_data['time_series'].apply(safe_parse)

    # Make compatible with streamlit cache
    geo_data = make_df_hashable(geo_data)
    # Debugging: print("Columns in merged DataFrame:", geo_data.columns.tolist())
    return geo_data

# Call load_data and assign merged here
merged = load_data()

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
            df.columns = df.columns.str.strip()

            # Expected columns: ['Date', 'Cutback1', 'Cutback2', 'Cutback3 or Cutoff']
            # Handle missing or renamed last column:
            standard_columns = ['Date', 'Cutback1', 'Cutback2']
            if len(df.columns) == 4:
                third_label = df.columns[3]
                df.columns = standard_columns + [third_label]
                diversion_labels[wsc] = third_label
            else:
                diversion_labels[wsc] = "Cutback3"
                df.columns = standard_columns + ['Cutback3']

            # Normalize and fix date format (year replaced by 1900)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()

            def safe_replace_year(d):
                try:
                    if isinstance(d, (pd.Timestamp, datetime)) and pd.notna(d):
                        return d.replace(year=1900)
                except:
                    return pd.NaT
                return pd.NaT

            df['Date'] = df['Date'].apply(safe_replace_year)
            diversion_tables[wsc] = df
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

def get_valid_dates(merged_df, debug=False):
    """
    Extracts all unique dates from the 'time_series' column of the merged DataFrame
    where either 'Daily flow' or 'Calculated flow' has a non-NaN value.
    Returns dates as sorted strings in 'YYYY-MM-DD' format.
    """
    dates = set()
    skipped_entries = 0

    for idx, ts in enumerate(merged_df['time_series']):
        if not isinstance(ts, (list, tuple)): # Ensure it's iterable
            if debug:
                st.warning(f"Skipping row {idx} with non-list time_series: {type(ts)}")
            continue

        for item in ts:
            if 'date' in item:
                try:
                    # Attempt to parse as datetime, then format as string
                    d_obj = parse(item['date']).date() # Get only the date part
                    d_str = d_obj.strftime('%Y-%m-%d')

                    # Check for valid flow data before adding the date
                    daily_flow = item.get('Daily flow')
                    calculated_flow = item.get('Calculated flow')

                    if (daily_flow is not None and not pd.isna(daily_flow)) or \
                       (calculated_flow is not None and not pd.isna(calculated_flow)):
                        dates.add(d_str)
                    else:
                        if debug:
                            print(f"Skipped date {d_str} for station {merged_df.loc[idx, 'WSC']} — no valid flow data")
                        skipped_entries += 1
                except Exception as e:
                    if debug:
                        st.error(f"Date parse error for station {merged_df.loc[idx, 'WSC']}: {item.get('date')} -> {e}")
                    skipped_entries += 1
            else:
                if debug:
                    st.warning(f"Skipped time_series item without 'date' key for station {merged_df.loc[idx, 'WSC']}: {item}")
    
    if debug:
        st.info(f"✅ Total valid dates found: {len(dates)}")
        st.info(f"❌ Skipped entries (parsing or no flow data): {skipped_entries}")
    
    return sorted(list(dates))

# Placeholder for get_color_for_date (you need to implement your logic here)
def get_color_for_date(row, date_str):
    # This is a simplified placeholder.
    # You need to implement your actual logic based on compliance_color_WMP/SWA
    # and the specific date's flow and thresholds.
    daily = extract_daily_data(row['time_series'], date_str)
    flow = daily.get('Daily flow') or daily.get('Calculated flow') # Use either for color
    
    if row['PolicyType'] == 'WMP':
        thresholds = extract_thresholds(daily)
        return compliance_color_WMP(flow, thresholds)
    elif row['PolicyType'] == 'SWA':
        q80 = daily.get('Q80')
        q95 = daily.get('Q95')
        return compliance_color_SWA(row.get('StreamSize'), flow, q80, q95)
    return 'gray' # Default for stations with no policy type or no data

# --- Critical: Define valid_dates AFTER merged is loaded and before sidebar uses it ---
valid_dates = get_valid_dates(merged, debug=True) # Set debug=True to see more info in the console/logs

# --- Sidebar ---
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

# Ensure valid_dates is not empty before accessing its elements
if valid_dates:
    min_date = datetime.strptime(valid_dates[0], "%Y-%m-%d").date()
    max_date = datetime.strptime(valid_dates[-1], "%Y-%m-%d").date()
else:
    # Fallback if no valid dates are found in the data
    st.sidebar.warning("No valid dates found in the data. Date range picker will use default values.")
    min_date = datetime.now().date() - timedelta(days=365)
    max_date = datetime.now().date()

start_date = st.sidebar.date_input("Start", value=max_date - timedelta(days=7), min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End", value=max_date, min_value=min_date, max_value=max_date)

# Convert selected dates to string format for consistent comparison with time_series data
# This is crucial for matching dates.
selected_dates_list = [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start_date, end=end_date).date]
# Debugging: Show the selected dates being used for the map generation
st.sidebar.subheader("Selected Dates for Map:")
st.sidebar.write(f"Start: {start_date} | End: {end_date}")
st.sidebar.write(f"Number of days selected: {len(selected_dates_list)}")
st.sidebar.write(f"First 5 selected dates: {selected_dates_list[:5]}")
st.sidebar.write(f"Last 5 selected dates: {selected_dates_list[-5:]}")


with st.sidebar.expander("ℹ️ About this App"):
    st.markdown("""
    **🔍 What is this?**

    This tool visualizes flow data from Alberta water stations and evaluates compliance with flow thresholds used in water policy decisions.

    **📊 Data Sources:**

    - **Hydrometric data** and **Diversion thresholds** from Alberta River Basins Water Conservation layer (Rivers.alberta.ca)
    - Alberta has over 400 hydrometric stations operated by both the Alberta provincial government and the federal Water Survey of Canada, which provides near real time flow and water level monitoring data. For the purpose of this app, flow in meters cubed per second is used.
    - **Diversion Tables** from current provincial policy and regulations - use layer toggles on the right to swap between diversion tables and other thresholds for available stations.
    - **Stream size and policy type** from Alberta Environment and Protected Areas and local (Survace Water Allocation Directive) and local jurisdictions (Water Management Plans)

    **📏 Threshold Definitions:**

    - **WCO (Water Conservation Objective):** Target flow for ecosystem protection - sometimes represented as a percentage of "Natural Flow" (ie 45%), which is a theoretical value depicting what the flow of a system would be if there were no diversions
    - **IO (Instream Objective):** Minimum flow below which withdrawals are restricted
    - **IFN (Instream Flow Need):** Ecological flow requirement for sensitive systems
    - **Q80/Q95:** Statistical low flows based on historical comparisons; Q80 means flow is exceeded 80% of the time - often used as a benchmark for the low end of "typical flow".
    - Q90: The flow value exceeded 90% of the time. This means the river flow is above this level 90% of the time—representing a more extreme low flow than Q80.
    - Q95: The flow exceeded 95% of the time, meaning the river is flowing above this very low level 95% of the time. This is often considered a critical threshold for ecological health.
    - **Cutbacks 1/2/3:** Phased reduction thresholds for diversions - can represent cutbacks in rate of diversion or daily limits

    **🟢 Color Codes in Map:**

    - 🟢 Flow meets all thresholds
    - 🔴 Flow below one or more thresholds
    - 🟡 Intermediate (depends on stream size & Q-values)
    - ⚪ Missing or insufficient data
    - 🔵 **Blue border**: Station has a Diversion Table (click layer on right for additional thresholds)

    _🚧 This app is under development. Thanks for your patience — and coffee! ☕ - Lyndsay Greenwood_
    """)
with st.sidebar.expander("ℹ️ Who Cares?"):
    st.markdown("""
    **❓ Why does this matter?**

    Water is a shared resource, and limits must exist to ensure fair and equitable access. It is essential to environmental health, human life, and economic prosperity.
    However, water supply is variable—and increasingly under pressure from many angles: natural seasonal fluctuations, shifting climate and weather patterns, and changing socio-economic factors such as population growth and energy demand.

    In Alberta, many industries—from agriculture and manufacturing to energy production and resource extraction—depend heavily on water. Setting clear limits and thresholds on water diversions helps protect our waterways from overuse by establishing enforceable cutoffs. These limits are often written directly into water diversion licenses issued by the provincial government.

    While water conservation is a personal responsibility we all share, ensuring that diversion limits exist—and are respected—is a vital tool in protecting Alberta’s water systems and ecosystems for generations to come.

    """)

# Pre-generate both popup caches upfront
def get_most_recent_valid_date(row, dates):
    # Ensure 'dates' is sorted in descending order to find the most recent quickly
    for d_str in sorted(dates, reverse=True):
        daily = extract_daily_data(row['time_series'], d_str)
        if any(pd.notna(daily.get(k)) for k in ['Daily flow', 'Calculated flow']):
            return d_str
    return None

import hashlib

def get_date_hash(dates):
    """Create a short unique hash string for a list of dates."""
    date_str = ",".join(sorted(dates))
    return hashlib.md5(date_str.encode()).hexdigest()

@st.cache_data(show_spinner=True)
def generate_all_popups(merged_df, selected_dates_tuple, diversion_tables, diversion_labels): # Pass these in
    selected_dates = list(selected_dates_tuple)  # Convert tuple back to list for processing

    popup_cache_no_diversion = {}
    popup_cache_diversion = {}

    for index, row in merged_df.iterrows(): # Use index here to avoid direct modification of row
        wsc = row['WSC']
        try:
            # Pass diversion_tables and diversion_labels to make_popup_html_with_plot
            popup_cache_no_diversion[wsc] = make_popup_html_with_plot(row, selected_dates, show_diversion=False)
            popup_cache_diversion[wsc] = make_popup_html_with_plot(row, selected_dates, show_diversion=True)
        except Exception as e:
            st.error(f"Error generating popup for station {wsc}: {e}") # Show specific error
            popup_cache_no_diversion[wsc] = "<p>Error generating popup</p>"
            popup_cache_diversion[wsc] = "<p>Error generating popup</p>"

    return popup_cache_no_diversion, popup_cache_diversion

@st.cache_data(show_spinner=True)
def render_map_two_layers(merged_df, selected_dates_for_map): # Pass selected_dates here
    m = folium.Map(
        location=[merged_df['LAT'].mean(), merged_df['LON'].mean()],
        zoom_start=6,
        width='100%',
        height='1200px'
    )

    # Add responsive popup size script
    from branca.element import Element

    # Responsive popup width JS
    popup_resize_script = Element("""
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        const resizePopups = () => {
            const popups = document.querySelectorAll('.leaflet-popup-content');
            popups.forEach(p => {
                if (window.innerWidth < 500) {
                    p.style.width = '320px';
                    p.style.maxHeight = '90vh';
                    p.style.overflow = 'auto';
                } else {
                    p.style.width = '650px';
                    p.style.maxHeight = '600px';
                    p.style.overflow = 'auto';
                }
            });
        };
        const observer = new MutationObserver(resizePopups);
        observer.observe(document.body, { childList: true, subtree: true });
        resizePopups();
    });
    </script>
    """)
    m.get_root().html.add_child(popup_resize_script)

    m.get_root().html.add_child(folium.Element("""
        <style>
            /* Ensure the body (and thus the map) allows touch actions for zooming */
            body {
                touch-action: pan-x pan-y pinch-zoom !important;
            }
        </style>
    """))
    Fullscreen().add_to(m)

    # FeatureGroups for two modes
    fg_all = folium.FeatureGroup(name='All Stations')
    fg_diversion = folium.FeatureGroup(name='Diversion Stations')

    for _, row in merged_df.iterrows(): # Use merged_df passed as argument
        coords = [row['LAT'], row['LON']]

        # IMPORTANT: Use selected_dates_for_map here
        date = get_most_recent_valid_date(row, selected_dates_for_map)
        if not date:
            # st.write(f"No valid date found for {row['WSC']} within selected range.") # Debugging
            continue

        color = get_color_for_date(row, date)

        wsc = row['WSC']
        popup_html_no_diversion = st.session_state.popup_cache_no_diversion.get(wsc, "<p>No data</p>")
        popup_html_diversion = st.session_state.popup_cache_diversion.get(wsc, "<p>No data</p>")

        iframe_no_diversion = IFrame(html=popup_html_no_diversion, width=650, height=500)
        popup_no_diversion = folium.Popup(iframe_no_diversion, max_width=700)

        iframe_diversion = IFrame(html=popup_html_diversion, width=650, height=500)
        popup_diversion = folium.Popup(iframe_diversion, max_width=700)

        # Marker for ALL stations (show no diversion popup)
        folium.CircleMarker(
            location=coords,
            radius=7,
            color='black',
            weight=3,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_no_diversion,
            tooltip=row['station_name']
        ).add_to(fg_all)

        # Marker for diversion stations only (show diversion popup)
        if wsc in diversion_tables:
            folium.CircleMarker(
                location=coords,
                radius=7,
                color='blue',
                weight=3,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_diversion,
                tooltip=row['station_name']
            ).add_to(fg_diversion)

    # Add both layers to map
    fg_all.add_to(m)
    fg_diversion.add_to(m)

    # Add layer control to toggle between groups
    folium.LayerControl(collapsed=False).add_to(m)

    return m

# --- Display ---
st.title("Alberta Flow Threshold Viewer")

with st.spinner("🚧 App is loading... Grab a coffee while we fire it up ☕"):
    # The merged and diversion_tables are already loaded at the top-level.

    # Ensure selected_dates_list is defined before use
    # It's defined above in the sidebar section, so it should be available.

    # Always compute the current hash based on the selected_dates_list
    current_dates_hash = get_date_hash(selected_dates_list)

    # Check if popups need to be regenerated
    if ('popup_cache_no_diversion' not in st.session_state or
        'popup_cache_diversion' not in st.session_state or
        st.session_state.get('cached_dates_hash', '') != current_dates_hash):

        st.info("🔄 Regenerating popups for the selected date range...") # Debugging info
        no_diversion_cache, diversion_cache = generate_all_popups(merged, tuple(selected_dates_list), diversion_tables, diversion_labels)
        st.session_state.popup_cache_no_diversion = no_diversion_cache
        st.session_state.popup_cache_diversion = diversion_cache
        st.session_state.cached_dates_hash = current_dates_hash
    else:
        st.info("✅ Using cached popups.") # Debugging info

    # Render and display the map, passing the current selected_dates_list
    m = render_map_two_layers(merged, selected_dates_list)
    map_html = m.get_root().render()

    # Inject mobile-friendly viewport settings into <head>
    map_html = map_html.replace(
        "<head>",
        "<head><meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=5.0, minimum-scale=0.1'>"
    )

    # Display map
    st.components.v1.html(map_html, height=1200, scrolling=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Debugging: Station Data Viewer")
station_options = merged['WSC'].tolist()
selected_station_debug = st.sidebar.selectbox("Select a station to inspect its raw data:", station_options)

# Extract and display the time_series entries for this station
if selected_station_debug:
    station_row_debug = merged[merged['WSC'] == selected_station_debug].iloc[0]
    time_series_debug = station_row_debug['time_series']

    st.sidebar.write(f"Full time_series for **{selected_station_debug}**:")
    
    # Filter time_series to show only relevant dates for debugging
    filtered_time_series_debug = [
        item for item in time_series_debug if item.get('date') and 
        start_date <= datetime.strptime(item['date'], '%Y-%m-%d').date() <= end_date
    ]
    
    sorted_filtered_series_debug = sorted(filtered_time_series_debug, key=lambda x: x['date'], reverse=True)
    
    if sorted_filtered_series_debug:
        st.sidebar.write(f"Showing {len(sorted_filtered_series_debug)} entries within the selected date range ({start_date} to {end_date}):")
        st.sidebar.json(sorted_filtered_series_debug)
    else:
        st.sidebar.info("No data entries found for this station within the selected date range.")

    st.sidebar.write(f"Latest 5 entries in raw time_series (regardless of selected range):")
    sorted_all_series_debug = sorted(time_series_debug, key=lambda x: x['date'], reverse=True)
    st.sidebar.json(sorted_all_series_debug[:5])

    st.sidebar.write("---")
    st.sidebar.write(f"Valid dates identified for all stations (from get_valid_dates):")
    st.sidebar.write(f"Min Valid Date: {valid_dates[0] if valid_dates else 'N/A'}")
    st.sidebar.write(f"Max Valid Date: {valid_dates[-1] if valid_dates else 'N/A'}")
    st.sidebar.write(f"Total valid dates: {len(valid_dates)}")
