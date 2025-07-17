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

st.cache_data.clear()
st.set_page_config(layout="wide")
st.markdown("""
<style>
/* Target Streamlit's main content area and potentially the iframe */
.stApp, .streamlit-container, .stApp > header, .main, .block-container, iframe {
    touch-action: pan-x pan-y pinch-zoom !important;
    -ms-touch-action: pan-x pan-y pinch-zoom !important; /* For older IE/Edge */
}
/* Ensure no overflow issues from main app container */
body {
    overflow: auto !important;
}
</style>
""", unsafe_allow_html=True)
# --- Paths ---
DATA_DIR = "data"
DIVERSION_DIR = os.path.join(DATA_DIR, "DiversionTables")
STREAM_CLASS_FILE = os.path.join(DATA_DIR, "StreamSizeClassification.csv")

geo_data = gpd.read_parquet(os.path.join(DATA_DIR, "AB_WS_R_stations.parquet"))

# --- Load data ---
def make_df_hashable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list columns to tuples, and dicts within time_series lists to frozensets,
    for Streamlit caching compatibility.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if col == 'time_series':
            # This handles the specific 'time_series' column
            df_copy[col] = df_copy[col].apply(
                lambda ts_list: tuple(
                    frozenset(item.items()) if isinstance(item, dict) else item
                    for item in ts_list
                ) if isinstance(ts_list, (list, tuple)) else ts_list
            )
        elif df_copy[col].apply(lambda x: isinstance(x, list)).any():
            # This handles other columns that might be lists
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
def make_df_hashable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list columns to tuples, and dicts within time_series lists to frozensets,
    for Streamlit caching compatibility.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if col == 'time_series':
            # This handles the specific 'time_series' column
            df_copy[col] = df_copy[col].apply(
                lambda ts_list: tuple(
                    frozenset(item.items()) if isinstance(item, dict) else item
                    for item in ts_list
                ) if isinstance(ts_list, (list, tuple)) else ts_list
            )
        elif df_copy[col].apply(lambda x: isinstance(x, list)).any():
            # This handles other columns that might be lists
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
    merged_data = geo_data.merge( # Renamed to merged_data to avoid confusion with global 'merged'
        station_info[['WSC', 'PolicyType', 'StreamSize', 'LAT', 'LON']],
        on='WSC', how='left'
    )

    # Convert geometry to WKT (safe for caching)
    merged_data['geometry_wkt'] = merged_data.geometry.apply(lambda g: g.wkt if g else None)
    merged_data = merged_data.drop(columns=['geometry'])

    # Parse time_series safely AND robustly convert flow values immediately
    # This is the function we've been refining, now correctly integrated
    def safe_parse_and_convert_time_series(val):
        parsed_ts = []
        if isinstance(val, str):
            try:
                parsed_ts = json.loads(val)
            except json.JSONDecodeError:
                return [] # Return empty list if JSON parsing fails
        elif isinstance(val, (list, tuple)): # Already a list/tuple, common when reading Parquet directly
            parsed_ts = list(val) # Convert tuple to list for modification
        else:
            return [] # Handle unexpected types

        converted_ts = []
        for item in parsed_ts:
            if isinstance(item, dict):
                converted_item = item.copy() # Work on a copy
                
                # Safely convert 'Daily flow'
                daily_flow_val = converted_item.get('Daily flow')
                try:
                    # Check if pd.notna before attempting float conversion.
                    # pd.isna handles None and np.nan. If it's a string, it won't be pd.isna.
                    converted_item['Daily flow'] = float(daily_flow_val) if pd.notna(daily_flow_val) else None
                except (ValueError, TypeError):
                    converted_item['Daily flow'] = None # Coerce any non-convertible value (like "NA" string) to None

                # Safely convert 'Calculated flow'
                calc_flow_val = converted_item.get('Calculated flow')
                try:
                    converted_item['Calculated flow'] = float(calc_flow_val) if pd.notna(calc_flow_val) else None
                except (ValueError, TypeError):
                    converted_item['Calculated flow'] = None
                
                converted_ts.append(converted_item)
            else:
                converted_ts.append(item) # Should not happen with well-formed data
        return converted_ts

    merged_data['time_series'] = merged_data['time_series'].apply(safe_parse_and_convert_time_series)

    # Make compatible with streamlit cache
    # Now, make_df_hashable will convert the lists of dictionaries (after float conversion)
    # into tuples of frozensets, making the DataFrame hashable.
    merged_data = make_df_hashable(merged_data)
    print("Columns in merged DataFrame:", merged_data.columns.tolist())

    return merged_data

# Call load_data and assign merged here
merged = load_data() # This will now be hashable!

    return merged_data


# Call load_data and assign merged here
merged = load_data()

# --- ADD THESE DEBUG PRINTS HERE ---
st.write("---") # Add a separator for clarity in Streamlit
st.write("### Debugging Data in Streamlit App")
if not merged.empty:
    latest_date_in_merged = None
    if 'time_series' in merged.columns:
        # Iterate through time_series to find the absolute latest date present
        for _, row_data in merged.iterrows():
            if isinstance(row_data['time_series'], (list, tuple)): # Ensure it's iterable
                for item in row_data['time_series']:
                    if 'date' in item:
                        try:
                            current_date = datetime.strptime(item['date'], '%Y-%m-%d').date()
                            if latest_date_in_merged is None or current_date > latest_date_in_merged:
                                latest_date_in_merged = current_date
                        except ValueError:
                            pass # Handle cases where date format might be off

    st.write(f"**Latest date found across all time_series in `merged` DataFrame:** {latest_date_in_merged}")
    # You can also add a check for a specific station if needed, e.g., '07HC001'
    # Find the row for '07HC001'
    station_07HC001_row = merged[merged['WSC'] == '07HC001']
    if not station_07HC001_row.empty:
        station_07HC001_ts = station_07HC001_row.iloc[0]['time_series']
        latest_date_07HC001 = None
        latest_flow_07HC001 = 'N/A'
        latest_calc_flow_07HC001 = 'N/A'

        if isinstance(station_07HC001_ts, (list, tuple)):
            for item in sorted(station_07HC001_ts, key=lambda x: parse(x['date']) if 'date' in x else datetime.min):
                if 'date' in item:
                    try:
                        current_date = datetime.strptime(item['date'], '%Y-%m-%d').date()
                        if latest_date_07HC001 is None or current_date > latest_date_07HC001:
                            latest_date_07HC001 = current_date
                            latest_flow_07HC001 = item.get('Daily flow', 'N/A')
                            latest_calc_flow_07HC001 = item.get('Calculated flow', 'N/A')
                    except ValueError:
                        pass
        st.write(f"**For station 07HC001:**")
        st.write(f"  Latest Date: {latest_date_07HC001}")
        st.write(f"  Daily Flow: {latest_flow_07HC001}")
        st.write(f"  Calculated Flow: {latest_calc_flow_07HC001}")
    else:
        st.write("Station 07HC001 not found in merged data.")

st.write("---")
# --- END DEBUG PRINTS ---


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
            daily_flow_raw = item.get('Daily flow')
            calc_flow_raw = item.get('Calculated flow')

            # --- ADD THIS DEBUGGING HERE ---
            if date_str in ["2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18"] and item.get('station_no') == '07HC001': # Add station_no check if extract_daily_data gets it, otherwise you'll need to print outside this func.
                 st.sidebar.write(f"DEBUG: Station 07HC001, Date {date_str}")
                 st.sidebar.write(f"  Raw Daily Flow: {daily_flow_raw} (Type: {type(daily_flow_raw)})")
                 st.sidebar.write(f"  Raw Calculated Flow: {calc_flow_raw} (Type: {type(calc_flow_raw)})")
            # --- END DEBUGGING ---

            daily_flow = None
            calc_flow = None

            # First, handle explicit None/NaN
            if pd.isna(daily_flow_raw): # This catches Python None and numpy.nan
                daily_flow = None
            else:
                try:
                    daily_flow = float(daily_flow_raw)
                except (ValueError, TypeError):
                    daily_flow = None # Coerce any non-convertible value (like "NA" string) to None

            if pd.isna(calc_flow_raw):
                calc_flow = None
            else:
                try:
                    calc_flow = float(calc_flow_raw)
                except (ValueError, TypeError):
                    calc_flow = None

            return {
                **item,
                'Daily flow': daily_flow,
                'Calculated flow': calc_flow
            }
    return {} # Return empty dict if date not found

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
            if 'date' in item: # Only check for presence of 'date' key
                try:
                    # Parse and format consistently
                    d = datetime.strptime(item['date'], '%Y-%m-%d').strftime('%Y-%m-%d')
                    dates.add(d)
                except ValueError: # Catch cases where date string is malformed
                    pass
    return sorted(list(dates)) # Convert set to list and sort

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
      /* Base styles for the popup content - apply to all screen sizes first */
      .leaflet-popup-content {{
          padding: 0 !important;
          margin: 0 !important;
          box-sizing: border-box; /* Crucial for consistent sizing */
      }}

      .leaflet-popup-content-wrapper {{
          padding: 4px !important;
          background: #fff; /* Ensure white background */
          border-radius: 12px;
      }}

      .leaflet-popup-content > div {{
          width: 100% !important;
      }}

      .popup-wrapper {{
          width: 100% !important;
          max-width: 100% !important;
          height: auto; /* Let content dictate height */
          min-width: 10px; /* Smallest possible width */
          overflow-x: auto; /* Enable horizontal scrolling for overflow */
          overflow-y: auto; /* Enable vertical scrolling */
          -webkit-overflow-scrolling: touch;
          touch-action: pan-x pan-y;
          box-sizing: border-box;
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
          min-width: 280px; /* Minimum width for image on mobile */
      }}

      /* Mobile-specific adjustments */
      @media (max-width: 500px) {{
        .leaflet-popup-content {{
          max-width: 95vw !important; /* Allow almost full viewport width */
        }}
        .popup-wrapper {{
          max-height: 85vh !important; /* Max height for mobile to prevent overflow */
        }}
        .popup-wrapper table, .popup-wrapper h4 {{
          font-size: 12px !important; /* Smaller font on mobile */
        }}
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
    
    <div class='popup-wrapper'>
      <h4 style='font-size:{font_size};'>{row['station_name']}</h4>
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

def get_most_recent_valid_date_for_map_color(row):
    """
    Finds the most recent date for a station that has a non-null Daily flow or Calculated flow.
    This is used for the map marker color, and should check ALL available data for the station.
    """
    latest_valid_date_str = None # Store as string 'YYYY-MM-DD'
    if isinstance(row['time_series'], (list, tuple)):
        # Sort in reverse chronological order to easily find the latest
        # Ensure we can parse the date string for sorting
        sorted_ts = sorted(row['time_series'], key=lambda x: parse(x['date']) if 'date' in x else datetime.min, reverse=True)
        for item in sorted_ts:
            if 'date' in item:
                daily_flow = item.get('Daily flow')
                calc_flow = item.get('Calculated flow')
                # Check if either flow value is not None/NaN
                if pd.notna(daily_flow) or pd.notna(calc_flow):
                    try:
                        latest_valid_date_str = item['date'] # Return the string directly
                        return latest_valid_date_str
                    except ValueError:
                        continue # Skip malformed dates
    return None # No valid date with flow data found

@st.cache_data(show_spinner=True)
def render_map_two_layers():
    m = folium.Map(
        location=[merged['LAT'].mean(), merged['LON'].mean()],
        zoom_start=6,
        width='100%',
        height='100%',
        zoom_control=True,       # Ensures zoom +/- buttons are there
        scrollWheelZoom=True,    # For desktop scroll wheel
        dragging=True,           # Allows panning
        touchZoom=True,          # VERY IMPORTANT for mobile pinch-zoom
        doubleClickZoom=True     # Allows double-tap/click zoom
      
    )

    # Add responsive popup size script
    from branca.element import Element
    
    # Responsive popup width JS
  #  popup_resize_script = Element("""
   # <script>
    #document.addEventListener("DOMContentLoaded", function() {
     #   const resizePopups = () => {
      #      const popups = document.querySelectorAll('.leaflet-popup-content');
       #     popups.forEach(p => {
        #        if (window.innerWidth < 500) {
         #           p.style.width = '320px';
          #          p.style.maxHeight = '90vh';
           #         p.style.overflow = 'auto';
            #    } else {
             #       p.style.width = '650px';
              #      p.style.maxHeight = '600px';
               #     p.style.overflow = 'auto';
                #}
            #});
       # };
        #const observer = new MutationObserver(resizePopups);
        #observer.observe(document.body, { childList: true, subtree: true });
       # resizePopups();
  #  });
   # </script>
    #""")
    m.get_root().html.add_child(Element("""
        <meta name='viewport' content='width=device-width, initial-scale=1'>
    """))
    
    # Ensure the body (and thus the map) allows touch actions for zooming
    m.get_root().html.add_child(Element("""
        <style>
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

        date_for_map_color = get_most_recent_valid_date_for_map_color(row)
        if not date_for_map_color: # If no valid flow data, skip this station for map coloring
            continue

        color = get_color_for_date(row, date_for_map_color)

        # Use diversion popup cache if available
        wsc = row['WSC']
        # fallback to no diversion popup if diversion cache missing
        popup_html_diversion = st.session_state.popup_cache_diversion.get(wsc, "<p>No data</p>")
        popup_html_no_diversion = st.session_state.popup_cache_no_diversion.get(wsc, "<p>No data</p>")

        iframe_no_diversion = IFrame(html=popup_html_no_diversion, width=650, height=500) # Increased size for desktop
        popup_no_diversion = folium.Popup(iframe_no_diversion, max_width=700) # Slightly larger max_width for the overall popup
        
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
    # Now simply render the map directly to HTML
    st.components.v1.html(m._repr_html_(), height=2000, scrolling=True) # Use m._repr_html_() for direct rendering

    # Inject mobile-friendly viewport settings into <head>
#    map_html = map_html.replace(
#        "<head>",
#        "<head><meta name='viewport' content='width=device-width, initial-scale=1'>"
#    )
    
    # Display map
#
#st.components.v1.html(map_html, height=1200, scrolling=True)
