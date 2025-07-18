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
import hashlib
from branca.element import Element # Import Element for JS injection

st.cache_data.clear()
st.set_page_config(layout="wide")

# --- Paths ---
DATA_DIR = "data"
DIVERSION_DIR = os.path.join(DATA_DIR, "DiversionTables")
STREAM_CLASS_FILE = os.path.join(DATA_DIR, "StreamSizeClassification.csv")


# --- Utility function to make objects recursively hashable for Streamlit caching ---
def make_hashable_recursive(obj):
    """
    Recursively converts unhashable objects (lists, dicts) to hashable ones (tuples, frozensets).
    """
    if isinstance(obj, list):
        return tuple(make_hashable_recursive(item) for item in obj)
    if isinstance(obj, dict):
        return frozenset((k, make_hashable_recursive(v)) for k, v in sorted(obj.items()))
    if pd.isna(obj):
        return None
    # For shapely geometry objects, convert to WKT or a simple representation
    if hasattr(obj, '__geo_interface__'): # Catches shapely geometry objects
        try:
            return obj.wkt # Well-Known Text representation is a string and hashable
        except:
            return str(obj) # Fallback to string representation
    return obj

def make_df_hashable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies make_hashable_recursive to all object columns in a DataFrame.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            if not df_copy[col].empty and \
               df_copy[col].apply(lambda x: isinstance(x, (list, dict)) or hasattr(x, '__geo_interface__') or pd.isna(x)).any():
                df_copy[col] = df_copy[col].apply(make_hashable_recursive)
            elif not df_copy[col].empty and df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].apply(lambda x: None if pd.isna(x) else x)
    return df_copy


# --- Function to make a DataFrame or GeoDataFrame itself hashable for st.cache_data ---
def hash_dataframe(df: pd.DataFrame):
    """
    Generates a hashable representation of a DataFrame or GeoDataFrame.
    Handles the 'geometry' column specifically for GeoDataFrames.
    """
    # Create a copy to avoid modifying the original DataFrame,
    # and drop geometry column if present as it's typically unhashable
    df_for_hash = df.drop(columns=['geometry'], errors='ignore')

    # Ensure all remaining columns are hashable before converting to values for hashing
    df_for_hash = make_df_hashable(df_for_hash) 

    # Hash values (which are already made hashable by make_df_hashable)
    values_hash = tuple(tuple(row) for row in df_for_hash.values)

    # Hash index more robustly
    index_values_as_str = tuple(str(x) for x in df_for_hash.index)
    index_hash = hash(index_values_as_str)

    # Hash columns
    columns_hash = hash(tuple(df_for_hash.columns))
    return (values_hash, index_hash, columns_hash)

# --- Define the hash_funcs dictionary ---
PANDAS_HASH_FUNCS = {
    pd.DataFrame: hash_dataframe,
    gpd.GeoDataFrame: hash_dataframe,
    # These must be the *class types*, not instances
    date: lambda d: d.isoformat(), # Use 'date' (the class) as the key
    datetime: lambda dt: dt.isoformat(), # Use 'datetime' (the class) as the key
}

@st.cache_data
def load_data():
    PARQUET_FILE_PATH = os.path.join(DATA_DIR, "AB_WS_R_stations.parquet")
    try:
        geo_data_df = gpd.read_parquet(PARQUET_FILE_PATH)
        if 'station_no' in geo_data_df.columns and 'WSC' not in geo_data_df.columns:
            geo_data_df = geo_data_df.rename(columns={'station_no': 'WSC'})
        elif 'WSC' not in geo_data_df.columns:
            st.error(f"ERROR: 'WSC' or 'station_no' column not found in {PARQUET_FILE_PATH}.")
            st.stop()
    except FileNotFoundError:
        st.error(f"Error: Parquet file not found at {PARQUET_FILE_PATH}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading Parquet file: {e}")
        return pd.DataFrame()

    station_info = pd.read_csv(os.path.join(DATA_DIR, "AB_WS_R_StationList.csv"))
    station_info.columns = station_info.columns.str.strip()

    if 'WSC' in station_info.columns:
        pass
    elif 'station_no' in station_info.columns:
        station_info = station_info.rename(columns={'station_no': 'WSC'})
    else:
        st.error(f"ERROR: Neither 'WSC' nor 'station_no' column found in AB_WS_R_StationList.csv.")
        st.stop()

    geo_data_df = geo_data_df.merge(
        station_info[['WSC', 'PolicyType', 'StreamSize', 'LAT', 'LON']],
        on='WSC', how='left'
    )

    def parse_time_series_string(val):
        if pd.isna(val) or not isinstance(val, str):
            return []
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return []
    
    geo_data_df['time_series'] = geo_data_df['time_series'].apply(parse_time_series_string)

    geo_data_df = make_df_hashable(geo_data_df)

    return geo_data_df


merged = load_data()


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


# --- Helper functions ---
def extract_daily_data(time_series, date_str):
    for item_frozenset in time_series:
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
    flow = max(filter(pd.notna, [flow_daily, flow_calc]), default=None)

    policy = row['PolicyType']
    if policy == 'SWA':
        return compliance_color_SWA(row.get('StreamSize'), flow, daily.get('Q80'), daily.get('Q95'))
    elif policy == 'WMP':
        return compliance_color_WMP(flow, extract_thresholds(daily))
    return 'gray'

@st.cache_data(hash_funcs=PANDAS_HASH_FUNCS)
def get_valid_dates(data: pd.DataFrame):
    all_dates = set()
    df_for_iteration = data.drop(columns=['geometry'], errors='ignore')
    
    for ts_tuple in df_for_iteration['time_series']:
        if isinstance(ts_tuple, tuple):
            for item_frozenset in ts_tuple:
                if isinstance(item_frozenset, frozenset):
                    item_dict = dict(item_frozenset)
                    if 'date' in item_dict and ('Daily flow' in item_dict or 'Calculated flow' in item_dict):
                        if item_dict.get('Daily flow') is not None or item_dict.get('Calculated flow') is not None:
                            try:
                                # Ensure the date is parsed into a datetime.date object for consistency
                                d = parse(item_dict['date']).date() 
                                all_dates.add(d)
                            except (TypeError, ValueError):
                                pass
    
    if not all_dates:
        # If no valid dates, provide a sensible default range around today's date
        today = datetime.now().date()
        return sorted([today - timedelta(days=7), today + timedelta(days=7)])

    sorted_dates = sorted(list(all_dates))
    return sorted_dates # Already datetime.date objects

# --- Main App Logic ---
valid_dates = get_valid_dates(merged) # This returns datetime.date objects

# Ensure valid_dates is not empty before proceeding
if not valid_dates:
    st.error("No valid dates found in the data. Please check your data source.")
    st.stop() # Stop the app if no dates are available

# Determine default end date (today or latest valid date)
default_end_date = min(datetime.now().date(), max(valid_dates))
# Determine default start date (8 days before default_end_date, or min valid date)
default_start_date = max(min(valid_dates), default_end_date - timedelta(days=8))


with st.sidebar:
    st.header("Select Date Range")

    # Start Date input
    selected_start_date = st.date_input(
        "Start Date",
        value=default_start_date,
        min_value=min(valid_dates),
        max_value=max(valid_dates)
    )

    # End Date input
    selected_end_date = st.date_input(
        "End Date",
        value=default_end_date,
        min_value=min(valid_dates),
        max_value=max(valid_dates)
    )

# Ensure start date is not after end date
if selected_start_date > selected_end_date:
    st.sidebar.warning("Start date cannot be after end date. Adjusting start date.")
    selected_start_date = selected_end_date # Automatically adjust if invalid

st.write(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")

# Filter the list of valid_dates to only include those in the selected range
# This list contains datetime.date objects
selected_dates_for_processing = [d for d in valid_dates if selected_start_date <= d <= selected_end_date]

# For caching, convert this list of datetime.date objects to a tuple
selected_dates_tuple_for_cache = tuple(selected_dates_for_processing)

# Function to generate HTML for popup content
def make_popup_html_with_plot(row, selected_dates_list_for_popup, show_diversion):
    font_size = '16px'
    padding = '6px'
    border = '2px solid black'

    flows, calc_flows = [], []
    threshold_sets = []
    threshold_labels = set()
    show_daily_flow = show_calc_flow = False

    # Convert datetime.date objects to string format for lookup
    selected_date_strs = [d.strftime('%Y-%m-%d') for d in selected_dates_list_for_popup]

    for d_str in selected_date_strs:
        daily = extract_daily_data(row['time_series'], d_str)
        df = daily.get('Daily flow', float('nan'))
        cf = daily.get('Calculated flow', float('nan'))
        flows.append(df)
        calc_flows.append(cf)

        if pd.notna(df): show_daily_flow = True
        if pd.notna(cf): show_calc_flow = True

        current_thresholds = {}
        if show_diversion and row['WSC'] in diversion_tables:
            diversion_df = diversion_tables[row['WSC']]
            target_day = pd.to_datetime(d_str).replace(year=1900).normalize()
            div_row = diversion_df[diversion_df['Date'] == target_day]
            if not div_row.empty:
                div = div_row.iloc[0]
                third_label = diversion_labels.get(row['WSC'], 'Cutback3')
                current_thresholds = {
                    'Cutback1': div.get('Cutback1', float('nan')),
                    'Cutback2': div.get('Cutback2', float('nan')),
                    third_label: div.get(third_label, float('nan'))
                }
        elif row['PolicyType'] == 'WMP':
            current_thresholds = extract_thresholds(daily)
        elif row['PolicyType'] == 'SWA':
            current_thresholds = {k: daily.get(k, float('nan')) for k in ['Q80', 'Q90', 'Q95']}
        
        threshold_sets.append(current_thresholds)
        threshold_labels.update(current_thresholds.keys())

    threshold_labels = sorted(list(threshold_labels))
    plot_dates = pd.to_datetime(selected_date_strs)

    daily_colors, calc_colors = [], []
    for d_str in selected_date_strs:
        daily = extract_daily_data(row['time_series'], d_str)
        flow_daily = daily.get('Daily flow')
        flow_calc = daily.get('Calculated flow')
        
        # Determine color based on policy type for daily flow
        if row['PolicyType'] == 'SWA':
            daily_colors.append(compliance_color_SWA(row.get('StreamSize'), flow_daily, daily.get('Q80'), daily.get('Q95')))
        elif row['PolicyType'] == 'WMP':
            daily_colors.append(compliance_color_WMP(flow_daily, extract_thresholds(daily)))
        else:
            daily_colors.append('gray') # Default color if no policy or data

        # Determine color based on policy type for calculated flow
        if row['PolicyType'] == 'SWA':
            calc_colors.append(compliance_color_SWA(row.get('StreamSize'), flow_calc, daily.get('Q80'), daily.get('Q95')))
        elif row['PolicyType'] == 'WMP':
            calc_colors.append(compliance_color_WMP(flow_calc, extract_thresholds(daily)))
        else:
            calc_colors.append('gray') # Default color if no policy or data


    # Mobile-friendly scrollable popup wrapper
    # Adjusted CSS for more robust responsiveness and overflow handling
    html = f"""
    <style>
    /* Base styles for the popup content */
    .leaflet-popup-content-wrapper {{
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 12px;
    }}
    .leaflet-popup-content {{
        padding: 0 !important;
        margin: 0 !important;
        background: #fff; /* Ensure white background for content */
        border-radius: 12px; /* Match wrapper border-radius */
    }}

    /* The actual scrollable content wrapper inside the iframe */
    .popup-wrapper {{
        padding: 10px; /* Internal padding for content */
        box-sizing: border-box; /* Include padding in element's total width and height */
        overflow-x: auto; /* Enable horizontal scrolling if content overflows */
        overflow-y: auto; /* Enable vertical scrolling */
        -webkit-overflow-scrolling: touch; /* Smooth scrolling on iOS */
        touch-action: pan-x pan-y; /* Enable touch scrolling */
    }}

    /* Shared table styles */
    .popup-wrapper table {{
        border-collapse: collapse;
        border: {border};
        width: 100%; /* Table takes full width of its container */
        table-layout: auto; /* Allow columns to size based on content */
        word-wrap: break-word;
        margin-bottom: 15px; /* Space below table */
    }}
    .popup-wrapper th, .popup-wrapper td {{
        padding: {padding};
        border: {border};
        text-align: center;
        vertical-align: middle;
    }}
    .popup-wrapper th {{
        background-color: #f2f2f2;
    }}

    /* Shared image styles */
    .popup-wrapper img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0 auto;
    }}

    /* Mobile-specific adjustments */
    @media (max-width: 500px) {{
        .leaflet-popup-content {{
            width: 95vw !important; /* Almost full viewport width */
            min-width: 280px !important; /* Minimum practical width */
        }}
        .popup-wrapper {{
            max-height: 85vh !important; /* Max height for mobile to prevent overflow */
            font-size: 12px !important; /* Smaller font on mobile */
            padding: 5px; /* Less padding on mobile */
        }}
        .popup-wrapper h4 {{
            font-size: 14px !important; /* Slightly smaller title on mobile */
        }}
        .popup-wrapper table {{
            min-width: 280px; /* Ensure table has a minimum width on small screens */
            font-size: 12px !important;
        }}
        .popup-wrapper img {{
            min-width: 280px; /* Ensure image is at least this wide on mobile */
        }}
    }}
    
    /* Desktop-specific adjustments */
    @media (min-width: 501px) {{
        .leaflet-popup-content {{
            /* These values are for the overall popup, the iframe inside will size to its content */
            min-width: 650px !important;
            max-width: 700px !important;
            width: auto !important; /* Allow content to dictate width if smaller than min/max */
        }}
        .popup-wrapper {{
            max-height: 500px !important; /* Max height for desktop popups */
            font-size: {font_size}; /* Base font size for desktop */
            padding: 10px; /* More padding for desktop */
        }}
        .popup-wrapper h4 {{
            font-size: {font_size};
        }}
    }}
    </style>
    
    <div class='popup-wrapper'>
      <h4 style='font-size:{font_size};'>{row['station_name']}</h4>
      <table style=''>
        <tr><th>Metric</th>
    """
    html += ''.join([f"<th>{d_str}</th>" for d_str in selected_date_strs])
    html += "</tr>"

    if show_daily_flow:
        html += f"<tr><td>Daily Flow</td>"
        html += ''.join([
            f"<td style='background-color:{c};'>{f'{v:.2f}' if pd.notna(v) else 'NA'}</td>"
            for v, c in zip(flows, daily_colors)
        ])
        html += "</tr>"

    if show_calc_flow and any(pd.notna(val) for val in calc_flows):
        html += f"<tr><td>Calculated Flow</td>"
        html += ''.join([
            f"<td style='background-color:{c};'>{f'{v:.2f}' if pd.notna(v) else 'NA'}</td>"
            for v, c in zip(calc_flows, calc_colors)
        ])
        html += "</tr>"

    for label in threshold_labels:
        html += f"<tr><td>{label}</td>"
        html += ''.join([
            f"<td>" + (f"{t.get(label):.2f}" if pd.notna(t.get(label)) else "NA") + "</td>"
            for t in threshold_sets
        ])
        html += "</tr>"

    html += "</table>"

    # --- Plot rendering ---
    fig, ax = plt.subplots(figsize=(6.8, 3.5)) # Adjusted figsize for better mobile display
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

    html += f"<img src='data:image/png;base64,{img_base64}'>"
    html += "</div>" # Close popup-wrapper div

    return html

def get_date_hash(dates_list):
    """Create a short unique hash string for a list of datetime.date objects."""
    date_strs = [d.strftime('%Y-%m-%d') for d in sorted(dates_list)]
    return hashlib.md5(",".join(date_strs).encode()).hexdigest()

@st.cache_data(show_spinner=True)
def generate_all_popups(merged_df, selected_dates_tuple): # Expects tuple of datetime.date objects
    selected_dates_list = list(selected_dates_tuple) # Convert tuple back to list for processing

    popup_cache_no_diversion = {}
    popup_cache_diversion = {}

    for _, row in merged_df.iterrows():
        wsc = row['WSC']
        try:
            # Pass the list of datetime.date objects
            popup_cache_no_diversion[wsc] = make_popup_html_with_plot(row, selected_dates_list, show_diversion=False)
            popup_cache_diversion[wsc] = make_popup_html_with_plot(row, selected_dates_list, show_diversion=True)
        except Exception as e:
            st.exception(e)
            popup_cache_no_diversion[wsc] = "<p>Error generating popup</p>"
            popup_cache_diversion[wsc] = "<p>Error generating popup</p>"

    return popup_cache_no_diversion, popup_cache_diversion

def get_most_recent_valid_date(row, selected_dates_list): # Expects a list of datetime.date objects
    # Iterate over the provided selected_dates_list (datetime.date objects) in reverse
    for d_obj in sorted(selected_dates_list, reverse=True):
        d_str = d_obj.strftime('%Y-%m-%d') # Convert to string for lookup
        daily = extract_daily_data(row['time_series'], d_str)
        if any(pd.notna(daily.get(k)) for k in ['Daily flow', 'Calculated flow']):
            return d_str # Return the date as a string for get_color_for_date
    return None

# --- Map Rendering Function ---
def render_map_two_layers(selected_dates_list_for_map_coloring): # Pass the list of datetime.date objects
    m = folium.Map(
        location=[merged['LAT'].mean(), merged['LON'].mean()],
        zoom_start=6,
        width='100%',
        height='1200px'
    )

    # Add responsive popup size script - TARGETING THE IFRAME CONTENT
    # This script will run INSIDE the iframe, reacting to its parent (the popup) size.
    # The outer popup container itself is handled by folium.Popup max_width.
    popup_resize_script = Element("""
    <script>
    // This script runs within the iframe's context
    // It adjusts the popup-wrapper inside the iframe
    function adjustIframeContentSize() {
        const popupWrapper = document.querySelector('.popup-wrapper');
        if (!popupWrapper) return;

        // Get the computed size of the iframe's parent (the Folium popup content div)
        // This is tricky from inside the iframe. A simpler approach is to use media queries
        // within the CSS itself, and let the iframe adjust to content, or set explicit iframe sizes.
        // For dynamic sizing, it's often better to let the CSS inside the HTML handle responsiveness.
        // However, if we must, we can get the parent window's width (assuming same origin or relaxed sandbox)
        // For simplicity and robustness, rely on the CSS media queries in make_popup_html_with_plot
        // and fixed iframe sizes for desktop, responsive content for mobile.
        
        // This JS snippet for dynamic resizing of the popup *itself* (the iframe) is generally not needed
        // if the iframe's width/height is fixed or controlled by the parent map.
        // The CSS within make_popup_html_with_plot is more effective for internal content.
        
        // Let's remove this JS as it's trying to do what CSS media queries should do.
        // If it's truly for the parent popup, it needs to be injected differently.
    }
    // No longer adding this specific JS for dynamic iframe resizing based on screen size.
    // The CSS in make_popup_html_with_plot handles responsiveness within the popup.
    </script>
    """)
    # m.get_root().html.add_child(popup_resize_script) # Removed this line

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

    # Define common IFrame dimensions for desktop
    DESKTOP_IFRAME_WIDTH = 650
    DESKTOP_IFRAME_HEIGHT = 500

    for _, row in merged.iterrows():
        coords = [row['LAT'], row['LON']]

        # Get the most recent valid date (as a string) within the *selected range*
        # This date will be used for the marker's color on the map.
        date_for_marker_color = get_most_recent_valid_date(row, selected_dates_list_for_map_coloring)
        
        if not date_for_marker_color: # If no valid data in the selected range, skip marker
            continue

        color = get_color_for_date(row, date_for_marker_color) # get_color_for_date expects string date

        wsc = row['WSC']
        
        # Popups are pre-generated using `selected_dates_tuple_for_cache`, which contains all relevant dates
        popup_html_diversion = st.session_state.popup_cache_diversion.get(wsc, "<p>No data</p>")
        popup_html_no_diversion = st.session_state.popup_cache_no_diversion.get(wsc, "<p>No data</p>")

        # For desktop, set fixed IFrame sizes. For mobile, CSS inside the HTML content handles width.
        # Max_width on the Folium.Popup itself controls the *outer* popup container.
        # It's crucial for the `IFrame` to allow its content to scroll if it overflows.
        iframe_no_diversion = IFrame(html=popup_html_no_diversion, width=DESKTOP_IFRAME_WIDTH, height=DESKTOP_IFRAME_HEIGHT)
        popup_no_diversion = folium.Popup(iframe_no_diversion, max_width=DESKTOP_IFRAME_WIDTH + 50) # Allow slightly more for popup wrapper

        iframe_diversion = IFrame(html=popup_html_diversion, width=DESKTOP_IFRAME_WIDTH, height=DESKTOP_IFRAME_HEIGHT)
        popup_diversion = folium.Popup(iframe_diversion, max_width=DESKTOP_IFRAME_WIDTH + 50) # Allow slightly more for popup wrapper

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
                color='blue', # Blue border for diversion stations
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

    # Always compute the current hash based on the list of datetime.date objects
    current_dates_hash = get_date_hash(selected_dates_for_processing)

    # Check if popups need to be regenerated
    if ('popup_cache_no_diversion' not in st.session_state or
        'popup_cache_diversion' not in st.session_state or
        st.session_state.get('cached_dates_hash', '') != current_dates_hash):

        # Regenerate and store in session_state
        no_diversion_cache, diversion_cache = generate_all_popups(merged, selected_dates_tuple_for_cache)
        st.session_state.popup_cache_no_diversion = no_diversion_cache
        st.session_state.popup_cache_diversion = diversion_cache
        st.session_state.cached_dates_hash = current_dates_hash

    # Render and display the map, passing the list of datetime.date objects
    m = render_map_two_layers(selected_dates_for_processing) 
    map_html = m.get_root().render()

    # Inject mobile-friendly viewport settings into <head>
    # Make sure this replacement happens only once in the head tag
    if "<head>" in map_html:
        map_html = map_html.replace(
            "<head>",
            "<head><meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=5.0, minimum-scale=0.1'>"
        )

    # Display map
    st.components.v1.html(map_html, height=1200, scrolling=True)
