from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import geopandas as gpd # Keep this for general use, but you'll use json for the main file
import json
import folium
from folium import IFrame
from folium.plugins import Fullscreen
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from dateutil.parser import parse
import os

st.cache_data.clear() # Good, this will clear previous caches
st.set_page_config(layout="wide")

# --- Paths ---
DATA_DIR = "data"
DIVERSION_DIR = os.path.join(DATA_DIR, "DiversionTables")
STREAM_CLASS_FILE = os.path.join(DATA_DIR, "StreamSizeClassification.csv")

# REMOVE OR COMMENT OUT THIS LINE:
# geo_data = gpd.read_parquet(os.path.join(DATA_DIR, "AB_WS_R_stations.parquet"))


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
    # Load spatial GeoJSON data
    GEOJSON_FILE_PATH = os.path.join(DATA_DIR, "AB_WS_R_stations.geojson")

    try:
        with open(GEOJSON_FILE_PATH, 'r') as f:
            geo_json_raw = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: GeoJSON file not found at {GEOJSON_FILE_PATH}")
        return pd.DataFrame() # Return empty DataFrame on error

    properties_list = [feature['properties'] for feature in geo_json_raw['features']]
    geo_data_df = pd.DataFrame(properties_list)
     # --- ADD THIS SECTION ---
   
    if 'station_no' in geo_data_df.columns:
        geo_data_df = geo_data_df.rename(columns={'station_no': 'WSC'})
        
    elif 'WSC' not in geo_data_df.columns:
        

    # Load station attributes from CSV (contains PolicyType, StreamSize, etc.)
    station_info = pd.read_csv(os.path.join(DATA_DIR, "AB_WS_R_StationList.csv"))

  
    # Strip whitespace from all column names
    station_info.columns = station_info.columns.str.strip()
   

    # Check for 'WSC' or 'station_no'
    if 'WSC' in station_info.columns:
       
    elif 'station_no' in station_info.columns:
        station_info = station_info.rename(columns={'station_no': 'WSC'})
       
  

    # Merge in additional attributes
    geo_data_df = geo_data_df.merge(
        station_info[['WSC', 'PolicyType', 'StreamSize', 'LAT', 'LON']],
        on='WSC', how='left'
    )


    def safe_parse(val):
        if isinstance(val, str):
            try:
                # If it's a string, try to parse it as JSON
                return json.loads(val)
            except json.JSONDecodeError:
                # If it's a malformed JSON string, return an empty list or handle as needed
                return []
        # If it's already a list (or anything else), return as is
        return val

    geo_data_df['time_series'] = geo_data_df['time_series'].apply(safe_parse)


    # Make compatible with streamlit cache
    geo_data_df = make_df_hashable(geo_data_df)
    print("Columns in merged DataFrame:", geo_data_df.columns.tolist())
    st.write("DEBUG: Columns in merged DataFrame (from app):", geo_data_df.columns.tolist())
    st.write("DEBUG: Sample of time_series from app:", geo_data_df['time_series'].head(1).iloc[0])


    return geo_data_df


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


def make_popup_html_with_plot(row, selected_dates, show_diversion):
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
    plot_dates = pd.to_datetime(selected_dates)

    daily_colors, calc_colors = [], []
    for d in selected_dates:
        daily = extract_daily_data(row['time_series'], d)
        flow_daily = daily.get('Daily flow')
        flow_calc = daily.get('Calculated flow')
        thresholds = extract_thresholds(daily) if row['PolicyType'] == 'WMP' else {}
        q80 = daily.get('Q80')
        q95 = daily.get('Q95')

        daily_colors.append(
            compliance_color_SWA(row.get('StreamSize'), flow_daily, q80, q95)
            if row['PolicyType'] == 'SWA' else compliance_color_WMP(flow_daily, thresholds)
        )

        calc_colors.append(
            compliance_color_SWA(row.get('StreamSize'), flow_calc, q80, q95)
            if row['PolicyType'] == 'SWA' else compliance_color_WMP(flow_calc, thresholds)
        )

    # Mobile-friendly scrollable popup wrapper
    html = f"""
    <style>
      @media (max-width: 500px) {{
        .leaflet-popup-content {{
          width: auto !important;
          max-width: 95vw !important;
          max-width: 95vw !important; /* Allow it to be almost full width of the viewport */
          min-width: 10px !important;
      /* Base styles for the popup content - apply to all screen sizes first */
      .leaflet-popup-content {{
          padding: 0 !important;
          margin: 0 !important;
        }}
    
        .leaflet-popup-content-wrapper {{
          box-sizing: border-box; /* Crucial for consistent sizing */
      }}

      .leaflet-popup-content-wrapper {{
          padding: 4px !important;
        }}
    
        .leaflet-popup-content > div {{
          background: #fff; /* Ensure white background */
          border-radius: 12px;
      }}

      .leaflet-popup-content > div {{
          width: 100% !important;
          width: 100% !important; /* Ensure direct child takes full width */
        }}
    
        .popup-wrapper {{
          width: 95vw !important;
          max-width: 100vw !important;
          width: 100% !important; /* Ensure this takes full available width from parent */
          max-width: 100% !important; /* Cap it at 100% of its parent, not viewport */
      }}

      .popup-wrapper {{
          width: 100% !important;
          max-width: 100% !important;
          min-width: 10px !important;
          max-height: 85vh !important;
          overflow-x: auto !important;
          overflow-y: auto !important;
          overflow-x: auto !important; /* Enable horizontal scrolling if content overflows */
          overflow-y: auto !important; /* Enable vertical scrolling */
          height: auto; /* Let content dictate height */
          min-width: 10px; /* Smallest possible width */
          overflow-x: auto; /* Enable horizontal scrolling for overflow */
          overflow-y: auto; /* Enable vertical scrolling */
          -webkit-overflow-scrolling: touch;
          touch-action: pan-x pan-y;
          box-sizing: border-box;
          padding: 5px; /* Add some internal padding */
          padding: 5px;
        }}
    
        .popup-wrapper table, .popup-wrapper h4 {{
          font-size: 12px !important;
        }}
    
        .popup-wrapper table {{
          width: 100% !important;
          table-layout: fixed !important;
          table-layout: auto !important; /* Change to auto for better content fitting */
          table-layout: auto !important;
          padding: 5px; /* Internal padding for content */
      }}

      .popup-wrapper h4 {{
          font-size: {font_size};
          margin-top: 0; /* Remove default margin */
          margin-bottom: 10px;
          text-align: center; /* Center the title */
      }}

      .popup-wrapper table {{
          border-collapse: collapse;
          border: {border};
          font-size: {font_size};
          width: 100% !important; /* Table must take full width of its container */
          table-layout: auto; /* Allow columns to size based on content */
          word-wrap: break-word;
          min-width: 280px; /* Give it a minimum width to prevent excessive squishing */
          min-width: 280px;
        }}
    
        .popup-wrapper img {{
          min-width: 280px; /* Minimum width for table on mobile */
          margin-bottom: 15px; /* Space below table */
      }}

      .popup-wrapper th, .popup-wrapper td {{
          padding: {padding};
          border: {border};
          text-align: center; /* Center text in cells */
          vertical-align: middle;
      }}

      .popup-wrapper th {{
          background-color: #f2f2f2; /* Light grey background for headers */
      }}

      .popup-wrapper img {{
          max-width: 100% !important;
          height: auto !important;
          display: block !important;
          margin: 0 auto !important;
          min-width: 280px; /* Give plot image a minimum width */
          min-width: 280px;
        }}
          min-width: 280px; /* Minimum width for image on mobile */
      }}
    
      /* Styles for larger screens to ensure consistency */
      /* Styles for larger screens to ensure consistency and proper sizing */
      @media (min-width: 501px) {{

      /* Mobile-specific adjustments */
      @media (max-width: 500px) {{
        .leaflet-popup-content {{
            width: 600px !important; /* Adjust as needed for desktop */
            max-width: 90vw !important;
            /* The overall popup container width will be largely controlled by folium.Popup max_width */
            /* We'll let the content wrapper fill that space */
            width: auto !important; /* Allow content to dictate width up to max_width */
            max-width: 700px !important; /* Should match folium.Popup max_width or slightly less */
            min-width: 600px !important; /* Ensure a minimum size on desktop */
          max-width: 95vw !important; /* Allow almost full viewport width */
        }}
        .popup-wrapper {{
            width: 100% !important; /* Ensure content fills available popup width */
            width: 100% !important; /* Ensure content fills available popup width within the IFrame */
            max-width: 100% !important;
            max-height: 600px !important; /* Desktop max height */
            max-height: 500px !important; /* Should match IFrame height or slightly less for padding */
            overflow-x: hidden !important; /* No horizontal scroll on desktop normally */
            overflow-y: auto !important;
            padding: 10px;
            padding: 10px; /* More padding for desktop view */
          max-height: 85vh !important; /* Max height for mobile to prevent overflow */
        }}
        .popup-wrapper table {{
            width: 100% !important;
            width: 100% !important; /* Ensure table fills 100% of the wrapper */
            table-layout: auto !important;
        .popup-wrapper table, .popup-wrapper h4 {{
          font-size: 12px !important; /* Smaller font on mobile */
        }}
        .popup-wrapper img {{
            max-width: 100% !important;
            max-width: 100% !important; /* Ensure image scales within the wrapper */
            height: auto !important;
      }}

      /* Desktop-specific adjustments */
      @media (min-width: 501px) {{
        .leaflet-popup-content {{
            /* These values align with the IFrame size set in render_map_two_layers */
            min-width: 650px !important; 
            max-width: 700px !important; /* Max width for the overall popup, slightly more than IFrame */
            width: auto !important; /* Allow internal content to dictate width if smaller than min/max */
        }}
        .popup-wrapper {{
            max-height: 500px !important; /* Max height for desktop to match IFrame */
            overflow-x: hidden; /* No horizontal scroll on desktop normally */
            padding: 10px; /* More padding for desktop */
        }}
      }}
    </style>
    
    <!-- Responsive popup container -->
    <div class='popup-wrapper'>
      <h4 style='font-size:{font_size};'>{row['station_name']}</h4>
      <table style='border-collapse: collapse; border: {border}; font-size:{font_size}; width: 100%; max-width: 100%; table-layout: fixed;'>
      <table style='border-collapse: collapse; border: {border}; font-size:{font_size}; width: 100%; max-width: 100%;'>
      <table style='border-collapse: collapse; font-size:{font_size}; width: 100%; max-width: 100%;'>
        <tr><th style='padding:{padding}; border:{border};'>Metric</th>
    """
    html += ''.join([f"<th style='padding:{padding}; border:{border};'>{d}</th>" for d in selected_dates])
    html += "</tr>"

    if show_daily_flow:
        html += f"<tr><td style='padding:{padding}; border:{border}; font-weight:bold;'>Daily Flow</td>"
        html += ''.join([
            f"<td style='padding:{padding}; border:{border}; background-color:{c};'>{f'{v:.2f}' if pd.notna(v) else 'NA'}</td>"
            for v, c in zip(flows, daily_colors)
        ])
        html += "</tr>"

    if show_calc_flow and any(pd.notna(val) for val in calc_flows):
        html += f"<tr><td style='padding:{padding}; border:{border}; font-weight:bold;'>Calculated Flow</td>"
        html += ''.join([
            f"<td style='padding:{padding}; border:{border}; background-color:{c};'>{f'{v:.2f}' if pd.notna(v) else 'NA'}</td>"
            for v, c in zip(calc_flows, calc_colors)
        ])
        html += "</tr>"

    for label in threshold_labels:
        html += f"<tr><td style='padding:{padding}; border:{border}; font-weight:bold;'>{label}</td>"
        html += ''.join([
            f"<td style='padding:{padding}; border:{border};'>" + (f"{t.get(label):.2f}" if pd.notna(t.get(label)) else "NA") + "</td>"
            for t in threshold_sets
        ])
        html += "</tr>"

    html += "</table><br>"

    # --- Plot rendering ---
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    # Adjusted figsize for better mobile display; width 6.8 is roughly 320px at 72dpi, suitable for mobile
    fig, ax = plt.subplots(figsize=(6.8, 3.5)) 
    ax.plot(plot_dates, flows, 'o-', label='Daily Flow', color='tab:blue', linewidth=2)
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.4, color='lightgrey')
    ax.set_axisbelow(True)
    if any(pd.notna(val) for val in calc_flows):
        ax.plot(plot_dates, calc_flows, 's--', label='Calculated Flow', color='tab:green', linewidth=2)

    threshold_colors = {
        'Cutback1': 'gold', 'Cutback2': 'orange', 'Cutback3': 'purple', 'Cutoff': 'red',
        'IO': 'orange', 'WCO': 'crimson', 'Q80': 'green', 'Q90': 'yellow',
        'Q95': 'orange', 'Minimum flow': 'red', 'IFN': 'red'
    }

    for label in threshold_labels:
        threshold_vals = [t.get(label, float('nan')) for t in threshold_sets]
        if all(pd.isna(threshold_vals)):
            continue
        ax.plot(plot_dates, threshold_vals, linestyle='--', label=label,
                color=threshold_colors.get(label, 'gray'), linewidth=2)

    ax.set_ylabel('Flow')
    ax.legend(fontsize=8)
    ax.set_title('Flow and Thresholds Over Time')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    html += f"<img src='data:image/png;base64,{img_base64}' style='max-width: 100%; height: auto; display: block; margin: 0 auto;'>"
    html += "</div>"

    return html

import hashlib

def get_date_hash(dates):
    """Create a short unique hash string for a list of dates."""
    date_str = ",".join(sorted(dates))
    return hashlib.md5(date_str.encode()).hexdigest()

@st.cache_data(show_spinner=True)
def generate_all_popups(merged_df, selected_dates_tuple):
    selected_dates = list(selected_dates_tuple)  # Convert tuple back to list for processing

    popup_cache_no_diversion = {}
    popup_cache_diversion = {}

    for _, row in merged_df.iterrows():
        wsc = row['WSC']
        try:
            popup_cache_no_diversion[wsc] = make_popup_html_with_plot(row, selected_dates, show_diversion=False)
            popup_cache_diversion[wsc] = make_popup_html_with_plot(row, selected_dates, show_diversion=True)
        except Exception as e:
            st.exception(e)
            popup_cache_no_diversion[wsc] = "<p>Error generating popup</p>"
            popup_cache_diversion[wsc] = "<p>Error generating popup</p>"

    return popup_cache_no_diversion, popup_cache_diversion


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
min_date = datetime.strptime(valid_dates[0], "%Y-%m-%d").date()
max_date = datetime.strptime(valid_dates[-1], "%Y-%m-%d").date()
start_date = st.sidebar.date_input("Start", value=max_date - timedelta(days=7), min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End", value=max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

selected_dates = [d for d in valid_dates if start_date.strftime('%Y-%m-%d') <= d <= end_date.strftime('%Y-%m-%d')]

with st.sidebar.expander("ℹ️ About this App"):
    st.markdown("""
    **🔍 What is this?**  
    This tool visualizes flow data from Alberta water stations and evaluates compliance with flow thresholds used in water policy decisions.

    **📊 Data Sources:**  
    - **Hydrometric data** and  **Diversion thresholds** from Alberta River Basins Water Conservation layer (Rivers.alberta.ca)
    - Alberta has over 400 hydrometric stations operated by both the Alberta provincial government and the federal Water Survey of Canada, which provides near real time flow and water level monitoring data. For the purpose of this app, flow in meters cubed per second is used.
    - **Diversion Tables** from current provincial policy and regulations - use layer toggles on the right to swap between diversion tables and other thresholds for available stations.
    - **Stream size and policy type** from Alberta Environment and Protected Areas and local (Survace Water Allocation Directive) and local jurisdictions (Water Management Plans)

    **📏 Threshold Definitions:**  
    - **WCO (Water Conservation Objective):** Target flow for ecosystem protection - sometimes represented as a percentage of "Natural Flow" (ie 45%), which is a theoretical value depicting what the flow of a system would be if there were no diversions
    - **IO (Instream Objective):** Minimum flow below which withdrawals are restricted  
    - **IFN (Instream Flow Need):** Ecological flow requirement for sensitive systems  
    - **Q80/Q95:** Statistical low flows based on historical comparisons; Q80 means flow is exceeded 80% of the time - often used as a benchmark for the low end of "typical flow". 
    - Q90: The flow value exceeded 90% of the time. This means the river flow is above this level 90% of the time—representing a more extreme low flow than Q80.
    - Q95: The flow exceeded 95% of the time, meaning the river is flowing above this very low level 95% of the time.  This is often considered a critical threshold for ecological health.
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
    for d in sorted(dates, reverse=True):
        daily = extract_daily_data(row['time_series'], d)
        if any(pd.notna(daily.get(k)) for k in ['Daily flow', 'Calculated flow']):
            return d
    return None

@st.cache_data(show_spinner=True)
def render_map_two_layers():
    m = folium.Map(
        location=[merged['LAT'].mean(), merged['LON'].mean()],
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

    for _, row in merged.iterrows():
        coords = [row['LAT'], row['LON']]

        date = get_most_recent_valid_date(row, selected_dates)
        if not date:
            continue

        color = get_color_for_date(row, date)

        # Use diversion popup cache if available
        wsc = row['WSC']
        # fallback to no diversion popup if diversion cache missing
        popup_html_diversion = st.session_state.popup_cache_diversion.get(wsc, "<p>No data</p>")
        popup_html_no_diversion = st.session_state.popup_cache_no_diversion.get(wsc, "<p>No data</p>")

        iframe_no_diversion = IFrame(html=popup_html_no_diversion, width=300, height=400)
        popup_no_diversion = folium.Popup(iframe_no_diversion, max_width='auto')
        iframe_no_diversion = IFrame(html=popup_html_no_diversion, width=650, height=500) # Increased size for desktop
        popup_no_diversion = folium.Popup(iframe_no_diversion, max_width=700) # Slightly larger max_width for the overall popup

        iframe_diversion = IFrame(html=popup_html_diversion, width=300, height=400)
        popup_diversion = folium.Popup(iframe_diversion, max_width='auto')
        iframe_diversion = IFrame(html=popup_html_diversion, width=650, height=500) # Increased size for desktop
        popup_diversion = folium.Popup(iframe_diversion, max_width=700) # Slightly larger max_width for the overall popup

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
    # Load all data and set up popups
    merged = load_data()
    diversion_tables, diversion_labels = load_diversion_tables()

    # Always compute the current hash
    current_dates_hash = get_date_hash(selected_dates)

    if ('popup_cache_no_diversion' not in st.session_state or
        'popup_cache_diversion' not in st.session_state or
        st.session_state.get('cached_dates_hash', '') != current_dates_hash):

        no_diversion_cache, diversion_cache = generate_all_popups(merged, tuple(selected_dates))
        st.session_state.popup_cache_no_diversion = no_diversion_cache
        st.session_state.popup_cache_diversion = diversion_cache
        st.session_state.cached_dates_hash = current_dates_hash

    else:
        cached_dates_hash = st.session_state.get('cached_dates_hash', '')
        if cached_dates_hash != current_dates_hash:
            no_diversion_cache, diversion_cache = generate_all_popups(merged, tuple(selected_dates))
            st.session_state.popup_cache_no_diversion = no_diversion_cache
            st.session_state.popup_cache_diversion = diversion_cache
            st.session_state.cached_dates_hash = current_dates_hash

    # Render and display the map
    m = render_map_two_layers()
    map_html = m.get_root().render()

    # Inject mobile-friendly viewport settings into <head>
    map_html = map_html.replace(
        "<head>",
        "<head><meta name='viewport' content='width=device-width, initial-scale=1, minimum-scale=0.1, maximum-scale=5, user-scalable=yes'>"
        "<head><meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=5.0, minimum-scale=0.1'>"
    )

    # Display map
    st.components.v1.html(map_html, height=1200, scrolling=True)
