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

# --- IMPORTANT: Clear Streamlit's cache at the very start for fresh data on app rerun ---
# This helps ensure new data or code changes are picked up, especially after file updates.
st.cache_data.clear()

st.set_page_config(layout="wide")

# --- Paths ---
DATA_DIR = "data"
DIVERSION_DIR = os.path.join(DATA_DIR, "DiversionTables")
STREAM_CLASS_FILE = os.path.join(DATA_DIR, "StreamSizeClassification.csv")

# --- Utility function to make objects recursively hashable for Streamlit caching ---
# This function is crucial for handling complex nested data structures (like lists of dicts)
# within your DataFrame columns, ensuring they are compatible with st.cache_data.
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
    if pd.isna(obj):
        return None # Convert all NaN variations to hashable None
    
    # Handle None explicitly (already hashable, but good for consistency)
    if obj is None:
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
        # If it's a scalar numpy value (e.g., np.int64, np.float64)
        if hasattr(obj, 'shape') and obj.shape == ():
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
        # For numeric columns, ensure no non-hashable numpy types or NaNs remain if they somehow got in
        elif df_copy[col].dtype in ['float64', 'int64'] and df_copy[col].isnull().any():
            # Convert NaNs to None for consistent hashing of numeric columns
            df_copy[col] = df_copy[col].apply(lambda x: None if pd.isna(x) else x)
    return df_copy

# --- Function to make a DataFrame or GeoDataFrame itself hashable for st.cache_data ---
# This tells Streamlit's caching mechanism how to create a consistent hash for DataFrames,
# which are generally unhashable by default.
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

# --- Define the hash_funcs dictionary for Streamlit's cache ---
# This dictionary maps data types to their custom hashing functions.
PANDAS_HASH_FUNCS = {
    pd.DataFrame: hash_dataframe,
    gpd.GeoDataFrame: hash_dataframe,
    date: lambda d: d.isoformat(), # Hash date objects by their ISO format string
    datetime: lambda dt: dt.isoformat(), # Hash datetime objects by their ISO format string
}

# --- Data Loading Function ---
# This function loads and preprocesses your station data.
@st.cache_data(hash_funcs=PANDAS_HASH_FUNCS) # Apply hash_funcs to this cache
def load_data():
    # Load spatial GeoData from parquet
    geo_data = gpd.read_parquet(os.path.join(DATA_DIR, "AB_WS_R_stations.parquet"))
    geo_data = geo_data.rename(columns={'station_no': 'WSC'})

    # Load station attributes from CSV
    station_info = pd.read_csv(os.path.join(DATA_DIR, "AB_WS_R_StationList.csv"))

    # --- FIX: Ensure 'Station Name' column is correctly handled and present ---
    # This addresses the KeyError: 'Station Name' by ensuring the column
    # exists and has the expected casing after all data transformations.
    if 'station_name' in geo_data.columns:
        geo_data = geo_data.rename(columns={'station_name': 'Station Name'})
    
    # Prepare columns to merge from station_info
    cols_to_merge_from_station_info = ['WSC', 'PolicyType', 'StreamSize', 'LAT', 'LON']
    if 'Station Name' in station_info.columns:
        cols_to_merge_from_station_info.append('Station Name')

    # Merge in additional attributes. Use suffixes to resolve potential column name conflicts
    geo_data = geo_data.merge(
        station_info[cols_to_merge_from_station_info],
        on='WSC', how='left', suffixes=('_geo', '_info') 
    )
    
    # Resolve any potential duplicate 'Station Name' columns after merge
    if 'Station Name_info' in geo_data.columns and 'Station Name_geo' in geo_data.columns:
        # Prioritize the 'Station Name' from the original geo_data, fill NaNs with info data
        geo_data['Station Name'] = geo_data['Station Name_geo'].fillna(geo_data['Station Name_info'])
        geo_data = geo_data.drop(columns=['Station Name_geo', 'Station Name_info'])
    elif 'Station Name_geo' in geo_data.columns:
        geo_data = geo_data.rename(columns={'Station Name_geo': 'Station Name'})
    elif 'Station Name_info' in geo_data.columns:
        geo_data = geo_data.rename(columns={'Station Name_info': 'Station Name'})
    # If no 'Station Name' found from either source, consider setting a default or raising an error.
    # For now, we assume at least one source will provide it.

    # Convert geometry to WKT (Well-Known Text) for hashability if it's a GeoDataFrame.
    # make_hashable_recursive also handles this, but explicit conversion here ensures consistency.
    if 'geometry' in geo_data.columns and isinstance(geo_data, gpd.GeoDataFrame):
        geo_data['geometry'] = geo_data['geometry'].apply(lambda g: g.wkt if g else None)

    # Parse 'time_series' JSON strings and make its content fully hashable.
    # This is crucial for the UnhashableParamError fix when time_series contains nested structures.
    def safe_parse_and_hash(val):
        if isinstance(val, str):
            try:
                parsed_json = json.loads(val)
                # Apply the recursive hashable conversion right after parsing
                return make_hashable_recursive(parsed_json)
            except json.JSONDecodeError:
                return tuple() # Return an empty tuple if JSON parsing fails
        # If it's already not a string or is NaN, make it a hashable empty tuple for consistency
        return tuple() if pd.isna(val) else make_hashable_recursive(val)

    geo_data['time_series'] = geo_data['time_series'].apply(safe_parse_and_hash)

    # Apply make_df_hashable to the entire DataFrame to ensure all columns (especially 'object' types)
    # and their contents are fully hashable, which is essential for st.cache_data.
    geo_data = make_df_hashable(geo_data)

    print("Columns in merged DataFrame:", geo_data.columns.tolist())

    return geo_data

# Call load_data to get the processed DataFrame
merged = load_data()

# --- Load diversion tables ---
# This function loads and preprocesses diversion tables.
@st.cache_data(hash_funcs=PANDAS_HASH_FUNCS) # Apply hash_funcs to this cache
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
                diversion_labels[wsc] = "Cutback3" # Default label if not explicitly found
                df.columns = standard_columns + ['Cutback3']

            # Normalize and fix date format (year replaced by 1900)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()

            def safe_replace_year(d):
                try:
                    if isinstance(d, (pd.Timestamp, datetime)) and pd.notna(d):
                        return d.replace(year=1900) # Replace year to 1900 for consistent seasonal comparison
                except:
                    return pd.NaT # Return Not a Time if parsing fails
                return pd.NaT

            df['Date'] = df['Date'].apply(safe_replace_year)

            # Ensure the diversion table itself is hashable before storing in dictionary
            # This is important as DataFrames are stored in a dict which might be cached indirectly.
            diversion_tables[wsc] = make_df_hashable(df) 

    return diversion_tables, diversion_labels

diversion_tables, diversion_labels = load_diversion_tables()

# --- Helper functions for data extraction and compliance logic ---
def extract_daily_data(time_series_hashed, date_str):
    # time_series_hashed is now expected to be a tuple of frozensets (thanks to make_hashable_recursive)
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

def get_color_for_date(row, date_str): # Expects date_str
    daily = extract_daily_data(row['time_series'], date_str)
    flow_daily = daily.get('Daily flow')
    flow_calc = daily.get('Calculated flow')
    # Prioritize Daily flow, then Calculated flow, otherwise None
    flow = max(filter(lambda x: pd.notna(x), [flow_daily, flow_calc]), default=None)

    policy = row['PolicyType']
    if policy == 'SWA':
        return compliance_color_SWA(row.get('StreamSize'), flow, daily.get('Q80'), daily.get('Q95'))
    elif policy == 'WMP':
        return compliance_color_WMP(flow, extract_thresholds(daily))
    return 'gray'

# --- Function to get valid dates from the loaded data ---
# This determines the range of dates available for the date picker in the sidebar.
@st.cache_data(hash_funcs=PANDAS_HASH_FUNCS) # Apply hash_funcs to this cache
def get_valid_dates(data: pd.DataFrame):
    all_dates = set()
    df_for_iteration = data.copy()
    
    # Ensure 'geometry' is dropped if it exists and is not WKT to avoid issues
    # (though make_hashable_recursive should have converted it to WKT already)
    if 'geometry' in df_for_iteration.columns and not (df_for_iteration['geometry'].empty or isinstance(df_for_iteration['geometry'].iloc[0], str)):
        df_for_iteration = df_for_iteration.drop(columns=['geometry'])
    
    for ts_tuple in df_for_iteration['time_series']:
        if isinstance(ts_tuple, tuple): # Confirm it's the expected hashable tuple
            for item_frozenset in ts_tuple:
                if isinstance(item_frozenset, frozenset): # Confirm it's a frozenset
                    item_dict = dict(item_frozenset)
                    if 'date' in item_dict and ('Daily flow' in item_dict or 'Calculated flow' in item_dict):
                        # Ensure the flow value itself is not None/NaN before considering the date valid
                        flow_daily = item_dict.get('Daily flow')
                        flow_calc = item_dict.get('Calculated flow')
                        if (flow_daily is not None and pd.notna(flow_daily)) or \
                           (flow_calc is not None and pd.notna(flow_calc)):
                            try:
                                # Ensure the date is parsed into a datetime.date object for consistency
                                d = parse(item_dict['date']).date() 
                                all_dates.add(d)
                            except (TypeError, ValueError):
                                pass
    
    if not all_dates:
        # If no valid dates are found, provide a sensible default range around today's date
        today = datetime.now().date()
        return sorted([today - timedelta(days=7), today + timedelta(days=7)])

    sorted_dates = sorted(list(all_dates))
    # Convert dates back to string format for consistency with `selected_dates` further down
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
