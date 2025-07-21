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
from datetime import datetime, timedelta, date
from dateutil.parser import parse
import os
import hashlib
from branca.element import Element
from streamlit_folium import folium_static # Re-added: Essential for displaying Folium maps in Streamlit

# --- IMPORTANT: Clear Streamlit's cache at the very start for fresh data on app rerun ---
st.cache_data.clear()

st.set_page_config(layout="wide")

# --- Paths ---
DATA_DIR = "data"
DIVERSION_DIR = os.path.join(DATA_DIR, "DiversionTables")
STREAM_CLASS_FILE = os.path.join(DATA_DIR, "StreamSizeClassification.csv")

# --- Utility function to make objects recursively hashable for Streamlit caching ---
def make_hashable_recursive(obj):
    """
    Recursively converts unhashable objects (lists, dicts, numpy arrays, shapely geometries, NaNs)
    to hashable ones (tuples, frozensets, strings, None).
    """
    if isinstance(obj, list):
        return tuple(make_hashable_recursive(item) for item in obj)
    if isinstance(obj, dict):
        # Sort items for consistent hashing of dictionaries
        return frozenset((k, make_hashable_recursive(v)) for k, v in sorted(obj.items()))

    # Handle pandas/numpy NaN explicitly to ensure consistency
    if obj is None or (isinstance(obj, (float, int)) and pd.isna(obj)):
        return None

    # For shapely geometry objects (e.g., Point, Polygon) in GeoDataFrames
    if hasattr(obj, 'wkt'):
        return obj.wkt # Convert to Well-Known Text string, which is hashable

    # Handle numpy arrays/scalars explicitly (convert to tuple or Python scalar)
    if isinstance(obj, (pd.Series, pd.Index)):
        return tuple(make_hashable_recursive(item) for item in obj.tolist())
    if isinstance(obj, (set, frozenset)):
        return frozenset(make_hashable_recursive(item) for item in obj)
    if hasattr(obj, 'dtype') and hasattr(obj, 'tolist'): # Catches numpy arrays
        if hasattr(obj, 'shape') and obj.shape == (): # If it's a scalar numpy value
            return make_hashable_recursive(obj.item()) # Get the Python scalar
        return tuple(make_hashable_recursive(item) for item in obj.tolist())

    # Ensure any datetime/date objects are handled consistently by converting to ISO format string.
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # If it's still not hashable by now, try converting to a string as a last resort
    try:
        hash(obj) # Try hashing it directly
        return obj # If hashable, return as is
    except TypeError:
        return str(obj) # Fallback: convert to string if unhashable


def make_df_hashable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies make_hashable_recursive to all object columns in a DataFrame,
    and ensures numeric columns with NaNs are handled.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].apply(make_hashable_recursive)
        elif df_copy[col].dtype in ['float64', 'int64'] and df_copy[col].isnull().any():
            df_copy[col] = df_copy[col].apply(lambda x: None if pd.isna(x) else x)
    return df_copy


def hash_dataframe(df: pd.DataFrame):
    """
    Generates a hashable representation of a DataFrame or GeoDataFrame.
    Assumes 'geometry' column (if present) is already in a hashable format (e.g., WKT strings).
    """
    df_for_hash = df.copy() # Work on a copy

    # Ensure all columns are hashable before converting to values for hashing
    df_for_hash = make_df_hashable(df_for_hash)

    # Hash values (which are already made hashable by make_df_hashable)
    # Convert to a tuple of tuples for consistent hashing of the data
    values_hash = tuple(tuple(item) for item in df_for_hash.values)
    
    # Use MD5 hash for index and columns for a more compact and consistent hash
    index_hash = hashlib.md5(str(tuple(str(x) for x in df_for_hash.index)).encode()).hexdigest()
    columns_hash = hashlib.md5(str(tuple(str(x) for x in df_for_hash.columns)).encode()).hexdigest()
    
    return (values_hash, index_hash, columns_hash)

PANDAS_HASH_FUNCS = {
    pd.DataFrame: hash_dataframe,
    gpd.GeoDataFrame: hash_dataframe,
    date: lambda d: d.isoformat(),
    datetime: lambda dt: dt.isoformat(),
}


# --- Data Loading Function ---
@st.cache_data(hash_funcs=PANDAS_HASH_FUNCS)
def load_data():
    # Load spatial GeoData from GeoJSON
    geo_data = gpd.read_file(os.path.join(DATA_DIR, "AB_WS_R_stations.geojson")) # Changed to read GeoJSON
    geo_data = geo_data.rename(columns={'station_no': 'WSC'})

    station_info = pd.read_csv(os.path.join(DATA_DIR, "AB_WS_R_StationList.csv"))

    if 'station_name' in geo_data.columns:
        geo_data = geo_data.rename(columns={'station_name': 'Station Name'})
    
    cols_to_merge_from_station_info = ['WSC', 'PolicyType', 'StreamSize', 'LAT', 'LON']
    if 'Station Name' in station_info.columns:
        cols_to_merge_from_station_info.append('Station Name')

    geo_data = geo_data.merge(
        station_info[cols_to_merge_from_station_info],
        on='WSC', how='left', suffixes=('_geo', '_info')
    )
    
    if 'Station Name_info' in geo_data.columns and 'Station Name_geo' in geo_data.columns:
        geo_data['Station Name'] = geo_data['Station Name_geo'].fillna(geo_data['Station Name_info'])
        geo_data = geo_data.drop(columns=['Station Name_geo', 'Station Name_info'])
    elif 'Station Name_geo' in geo_data.columns:
        geo_data = geo_data.rename(columns={'Station Name_geo': 'Station Name'})
    elif 'Station Name_info' in geo_data.columns:
        geo_data = geo_data.rename(columns={'Station Name_info': 'Station Name'})

    # When reading GeoJSON, 'geometry' will be Shapely objects. Convert to WKT for hashing.
    if 'geometry' in geo_data.columns and isinstance(geo_data, gpd.GeoDataFrame):
        geo_data['geometry'] = geo_data['geometry'].apply(lambda g: g.wkt if g else None)

    # --- Refined safe_parse_and_hash for the time_series column ---
    def safe_parse_and_hash_time_series_entry(val):
        # First, handle explicit None or NaN values that might appear directly
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return tuple() # Return an empty tuple for missing or NaN values

        # If it's a string, assume it's a JSON string and try to parse it
        if isinstance(val, str):
            try:
                parsed_json = json.loads(val)
                return make_hashable_recursive(parsed_json)
            except json.JSONDecodeError:
                # If it's a string but not valid JSON, treat as empty data
                return tuple()
            
        # If it's already a Python list/tuple/dict/frozenset (i.e., already parsed by pyarrow/pandas)
        elif isinstance(val, (list, tuple, dict, frozenset)):
            return make_hashable_recursive(val)
            
        # Fallback for any other unexpected type that isn't handled above
        else:
            try:
                if hasattr(val, 'as_py'): # Common for pyarrow.lib.StructArray or similar
                    py_obj = val.as_py()
                    return make_hashable_recursive(py_obj)
                else:
                    return make_hashable_recursive(val)
            except Exception as e:
                print(f"Error parsing time_series data: {ts_data} - Error: {e}")
                return tuple() # Default to empty tuple for unhandleable types

    geo_data['time_series'] = geo_data['time_series'].apply(safe_parse_and_hash_time_series_entry)

    geo_data = make_df_hashable(geo_data)

    print("Columns in merged DataFrame:", geo_data.columns.tolist())

    return geo_data

merged = load_data()

# --- Load diversion tables ---
@st.cache_data(hash_funcs=PANDAS_HASH_FUNCS)
def load_diversion_tables():
    diversion_tables = {}
    diversion_labels = {}

    for f in os.listdir(DIVERSION_DIR):
        if f.endswith(".parquet"):
            wsc = f.split("_")[0]  
            file_path = os.path.join(DIVERSION_DIR, f)

            df = pd.read_parquet(file_path)
            df.columns = df.columns.str.strip()

            standard_columns = ['Date', 'Cutback1', 'Cutback2']
            if len(df.columns) == 4:
                third_label = df.columns[3]
                df.columns = standard_columns + [third_label]
                diversion_labels[wsc] = third_label
            else:
                diversion_labels[wsc] = "Cutback3"
                df.columns = standard_columns + ['Cutback3']

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()

            def safe_replace_year(d):
                try:
                    if isinstance(d, (pd.Timestamp, datetime)) and pd.notna(d):
                        return d.replace(year=1900)
                except:
                    return pd.NaT
                return pd.NaT

            df['Date'] = df['Date'].apply(safe_replace_year)
            diversion_tables[wsc] = make_df_hashable(df)  
    return diversion_tables, diversion_labels

diversion_tables, diversion_labels = load_diversion_tables()

# --- Helper functions for data extraction and compliance logic ---
def extract_daily_data(time_series_hashed, date_str):
    for item_frozenset in time_series_hashed:
        if isinstance(item_frozenset, frozenset):
            item_dict = dict(item_frozenset)
            if item_dict.get("date") == date_str:
                return item_dict
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

def get_color_for_date(row, date_str):
    daily = extract_daily_data(row['time_series'], date_str)
    flow_daily = daily.get('Daily flow')
    flow_calc = daily.get('Calculated flow')
    flow = max(filter(lambda x: pd.notna(x), [flow_daily, flow_calc]), default=None)

    policy = row['PolicyType']
    if policy == 'SWA':
        return compliance_color_SWA(row.get('StreamSize'), flow, daily.get('Q80'), daily.get('Q95'))
    elif policy == 'WMP':
        return compliance_color_WMP(flow, extract_thresholds(daily))
    return 'gray'

# --- Function to get valid dates from the loaded data ---
@st.cache_data(hash_funcs=PANDAS_HASH_FUNCS)
def get_valid_dates(data: pd.DataFrame):
    all_dates = set()
    df_for_iteration = data.copy()
    
    if 'geometry' in df_for_iteration.columns and not (df_for_iteration['geometry'].empty or isinstance(df_for_iteration['geometry'].iloc[0], str)):
        df_for_iteration = df_for_iteration.drop(columns=['geometry'])
    
    for ts_tuple in df_for_iteration['time_series']:
        if isinstance(ts_tuple, tuple):
            for item_frozenset in ts_tuple:
                if isinstance(item_frozenset, frozenset):
                    item_dict = dict(item_frozenset)
                    if 'date' in item_dict and ('Daily flow' in item_dict or 'Calculated flow' in item_dict):
                        flow_daily = item_dict.get('Daily flow')
                        flow_calc = item_dict.get('Calculated flow')
                        if (flow_daily is not None and pd.notna(flow_daily)) or \
                           (flow_calc is not None and pd.notna(flow_calc)):
                            try:
                                d = parse(item_dict['date']).date()  
                                all_dates.add(d)
                            except (TypeError, ValueError):
                                pass
    
    if not all_dates:
        today = datetime.now().date()
        return sorted([today - timedelta(days=7), today + timedelta(days=7)])

    sorted_dates = sorted(list(all_dates))
    return [d.strftime('%Y-%m-%d') for d in sorted_dates]

valid_dates = get_valid_dates(merged)

# --- Popup HTML Generation Function ---
# This function creates the HTML content for the map popups, including tables and plots.
def make_popup_html_with_plot(row, selected_dates_strs, show_diversion):
    font_size = '16px'
    padding = '6px'
    border = '2px solid black'

    flows, calc_flows = [], []
    threshold_sets = []
    threshold_labels = set()
    show_daily_flow = show_calc_flow = False

    # Convert selected_dates_strs (list of strings) to datetime objects for plotting
    plot_dates_dt = sorted([pd.to_datetime(d) for d in selected_dates_strs])
    plot_dates = [d.strftime('%Y-%m-%d') for d in plot_dates_dt] # Keep as strings for table display

    for d_str in selected_dates_strs:
        daily = extract_daily_data(row['time_series'], d_str)
        df = daily.get('Daily flow', float('nan'))
        cf = daily.get('Calculated flow', float('nan'))
        flows.append(df)
        calc_flows.append(cf)

        if pd.notna(df): show_daily_flow = True
        if pd.notna(cf): show_calc_flow = True

        if show_diversion and row['WSC'] in diversion_tables:
            diversion_df = diversion_tables[row['WSC']]
            target_day_1900 = pd.to_datetime(d_str).replace(year=1900).normalize()
            div_row = diversion_df[diversion_df['Date'] == target_day_1900]
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
    # Use plot_dates_dt (datetime objects) for matplotlib plotting
    # Use plot_dates (strings) for table headers

    daily_colors, calc_colors = [], []
    for d_str in selected_dates_strs: # Iterate over string dates
        daily = extract_daily_data(row['time_series'], d_str)
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

    # Mobile-friendly scrollable popup wrapper CSS
    html = f"""
    <style>
      @media (max-width: 500px) {{
        .leaflet-popup-content {{
          width: auto !important;
          max-width: 95vw !important;
          min-width: 10px !important;
        }}
        .leaflet-popup-content-wrapper {{
          padding: 4px !important;
        }}
        .popup-wrapper {{
          width: 100% !important;
          max-width: 100% !important;
          min-width: 10px !important;
          max-height: 85vh !important;
          overflow-x: auto !important;
          overflow-y: auto !important;
          -webkit-overflow-scrolling: touch;
          touch-action: pan-x pan-y;
          box-sizing: border-box;
          padding: 5px;
        }}
        .popup-wrapper table, .popup-wrapper h4 {{
          font-size: 12px !important;
        }}
        .popup-wrapper table {{
          width: 100% !important;
          table-layout: auto !important;
          min-width: 280px;
        }}
        .popup-wrapper img {{
          min-width: 280px;
        }}
      }}
      
      /* Desktop-specific adjustments */
      @media (min-width: 501px) {{
        .leaflet-popup-content {{
            min-width: 650px !important; 
            max-width: 700px !important;
            width: auto !important;
        }}
        .popup-wrapper {{
            max-height: 500px !important;
            overflow-x: hidden;
            padding: 10px;
        }}
        .popup-wrapper table {{
            width: 100% !important;
            table-layout: auto !important;
        }}
        .popup-wrapper img {{
            max-width: 100% !important;
            height: auto !important;
        }}
      }}

      /* Base styles for the popup content - apply to all screen sizes first */
      .leaflet-popup-content {{
          padding: 0 !important;
          margin: 0 !important;
      }}
      .leaflet-popup-content-wrapper {{
          box-sizing: border-box; /* Crucial for consistent sizing */
          padding: 4px !important;
      }}
      .leaflet-popup-content > div {{
          background: #fff; /* Ensure white background */
          border-radius: 12px;
          width: 100% !important; /* Ensure direct child takes full width */
      }}
      .popup-wrapper {{
          height: auto; /* Let content dictate height */
          box-sizing: border-box;
          padding: 5px; /* Add some internal padding */
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
          word-wrap: break-word;
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
          display: block !important;
          margin: 0 auto !important;
      }}
    </style>
    
    <div class='popup-wrapper'>
      <h4 style='font-size:{font_size};'>{row['Station Name']}</h4>
      <table style='border-collapse: collapse; font-size:{font_size}; width: 100%; max-width: 100%;'>
        <tr><th style='padding:{padding}; border:{border};'>Metric</th>
    """
    html += ''.join([f"<th style='padding:{padding}; border:{border};'>{d}</th>" for d in plot_dates]) # Use string dates for table headers
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
    fig, ax = plt.subplots(figsize=(6.8, 3.5)) # Adjusted figsize for better mobile display
    ax.plot(plot_dates_dt, flows, 'o-', label='Daily Flow', color='tab:blue', linewidth=2) # Use datetime objects for plotting
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.4, color='lightgrey')
    ax.set_axisbelow(True)
    if any(pd.notna(val) for val in calc_flows):
        ax.plot(plot_dates_dt, calc_flows, 's--', label='Calculated Flow', color='tab:green', linewidth=2) # Use datetime objects for plotting

    threshold_colors = {
        'Cutback1': 'gold', 'Cutback2': 'orange', 'Cutback3': 'purple', 'Cutoff': 'red',
        'IO': 'orange', 'WCO': 'crimson', 'Q80': 'green', 'Q90': 'yellow',
        'Q95': 'orange', 'Minimum flow': 'red', 'IFN': 'red'
    }

    for label in threshold_labels:
        threshold_vals = [t.get(label, float('nan')) for t in threshold_sets]
        if all(pd.isna(threshold_vals)):
            continue
        ax.plot(plot_dates_dt, threshold_vals, linestyle='--', label=label, # Use datetime objects for plotting
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

# --- Caching for generated popups ---
def get_date_hash(dates):
    """Create a short unique hash string for a list of dates."""
    date_str = ",".join(sorted(dates))
    return hashlib.md5(date_str.encode()).hexdigest()

@st.cache_data(show_spinner=True, hash_funcs=PANDAS_HASH_FUNCS) # Apply hash_funcs to this cache
def generate_all_popups(merged_df, selected_dates_tuple):
    # Convert tuple back to list of strings for processing within the function
    selected_dates_strs = list(selected_dates_tuple) 

    popup_cache_no_diversion = {}
    popup_cache_diversion = {}

    for _, row in merged_df.iterrows():
        wsc = row['WSC']
        try:
            popup_cache_no_diversion[wsc] = make_popup_html_with_plot(row, selected_dates_strs, show_diversion=False)
            popup_cache_diversion[wsc] = make_popup_html_with_plot(row, selected_dates_strs, show_diversion=True)
        except Exception as e:
            st.exception(e) # Display error in Streamlit if popup generation fails for a station
            popup_cache_no_diversion[wsc] = "<p>Error generating popup</p>"
            popup_cache_diversion[wsc] = "<p>Error generating popup</p>"

    return popup_cache_no_diversion, popup_cache_diversion


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
st.sidebar.write(f"**Calculated Min Date:** {min_date}")
st.sidebar.write(f"**Calculated Max Date:** {max_date}")
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

# --- Map Rendering Logic ---
def get_most_recent_valid_date(row, dates):
    # Find the most recent date with flow data for a given station within the selected range
    for d in sorted(dates, reverse=True):
        daily = extract_daily_data(row['time_series'], d)
        if any(pd.notna(daily.get(k)) for k in ['Daily flow', 'Calculated flow']):
            return d
    return None

@st.cache_data(show_spinner=True, hash_funcs=PANDAS_HASH_FUNCS) # Apply hash_funcs to this cache
def render_map_two_layers():
    m = folium.Map(
        location=[merged['LAT'].mean(), merged['LON'].mean()],
        zoom_start=6,
        width='100%',
        height='1200px'
    )

    # Add responsive popup size script (ensures popups adapt to screen size)
    popup_resize_script = Element("""
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        const resizePopups = () => {
            const popups = document.querySelectorAll('.leaflet-popup-content');
            popups.forEach(p => {
                p.style.maxHeight = window.innerWidth < 500 ? '90vh' : '600px';
                p.style.overflow = 'auto';
            });
        };
        const observer = new MutationObserver(resizePopups);
        observer.observe(document.body, { childList: true, subtree: true });
        resizePopups(); // Initial call
    });
    </script>
    """)
    m.get_root().html.add_child(popup_resize_script)

    # Enable touch actions for map interactions on mobile
    m.get_root().html.add_child(folium.Element("""
        <style>
            body {
                touch-action: pan-x pan-y pinch-zoom !important;
            }
        </style>
    """))
    Fullscreen().add_to(m) # Add fullscreen button

    # FeatureGroups for two map layers (All Stations and Diversion Stations)
    fg_all = folium.FeatureGroup(name='All Stations')
    fg_diversion = folium.FeatureGroup(name='Diversion Stations')

    for _, row in merged.iterrows():
        coords = [row['LAT'], row['LON']]

        date = get_most_recent_valid_date(row, selected_dates)
        if not date:
            continue # Skip stations with no valid flow data in the selected range

        color = get_color_for_date(row, date)

        wsc = row['WSC']
        # Retrieve pre-generated popup HTML from session state caches
        popup_html_no_diversion = st.session_state.popup_cache_no_diversion.get(wsc, "<p>No data</p>")
        popup_html_diversion = st.session_state.popup_cache_diversion.get(wsc, "<p>No data</p>")

        # Define IFrame and Popup with general dimensions, letting internal CSS handle responsiveness
        iframe_width = 700 
        iframe_height = 550 

        iframe_no_diversion = IFrame(html=popup_html_no_diversion, width=iframe_width, height=iframe_height)
        popup_no_diversion = folium.Popup(iframe_no_diversion, max_width=iframe_width + 50) 

        iframe_diversion = IFrame(html=popup_html_diversion, width=iframe_width, height=iframe_height)
        popup_diversion = folium.Popup(iframe_diversion, max_width=iframe_width + 50)


        # Marker for ALL stations (shows no diversion popup by default)
        folium.CircleMarker(
            location=coords,
            radius=7,
            color='black',
            weight=3,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_no_diversion,
            tooltip=row['Station Name'] 
        ).add_to(fg_all)

        # Marker for diversion stations only (shows diversion popup)
        if wsc in diversion_tables:
            folium.CircleMarker(
                location=coords,
                radius=7,
                color='blue', # Blue border indicates diversion table available
                weight=3,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_diversion,
                tooltip=row['Station Name']
            ).add_to(fg_diversion)

    # Add both layers to map and enable layer control
    fg_all.add_to(m)
    fg_diversion.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m

# --- Main Application Display ---
st.title("Alberta Flow Threshold Viewer")

with st.spinner("🚧 App is loading... Grab a coffee while we fire it up ☕"):
    # Always compute the current date hash based on selected_dates
    current_dates_hash = get_date_hash(selected_dates)

    # Check session state for cached popups using the date hash.
    # If caches don't exist, or the date range has changed (hash mismatch), re-generate popups.
    if ('popup_cache_no_diversion' not in st.session_state or
        'popup_cache_diversion' not in st.session_state or
        st.session_state.get('cached_dates_hash', '') != current_dates_hash):

        no_diversion_cache, diversion_cache = generate_all_popups(merged, tuple(selected_dates))
        st.session_state.popup_cache_no_diversion = no_diversion_cache
        st.session_state.popup_cache_diversion = diversion_cache
        st.session_state.cached_dates_hash = current_dates_hash
    
    # Render and display the map
    m = render_map_two_layers()
    map_html = m.get_root().render()

    # Inject mobile-friendly viewport settings into <head> for proper scaling
    map_html = map_html.replace(
        "<head>",
        "<head><meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=5.0, minimum-scale=0.1'>"
    )

    # Display map using Streamlit's components.v1.html
    st.components.v1.html(map_html, height=1200, scrolling=True)
