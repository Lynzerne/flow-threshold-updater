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

     
                    

            # Safer date replacement with try-except to catch errors
            def safe_replace_year(d):
                try:
                    if isinstance(d, (pd.Timestamp, datetime)) and pd.notna(d):
                        return d.replace(year=1900)
                    else:
                        return pd.NaT
                except Exception as e:
                    
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
show_diversion = st.sidebar.checkbox("Show Diversion Tables", value=False)

# --- Popup HTML builder ---
def make_popup_html_with_plot(row, selected_dates, show_diversion):
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    font_size = '13px'
    padding = '6px'
    border = '2px solid black'

    flows = []
    calc_flows = []
    threshold_sets = []
    threshold_labels = set()
    show_daily_flow = False
    show_calc_flow = False

    # Ensure selected_dates are sorted chronologically
    selected_dates = sorted(selected_dates, key=pd.to_datetime)

    for d in selected_dates:
        daily = extract_daily_data(row['time_series'], d)
        df = daily.get('Daily flow', float('nan'))
        cf = daily.get('Calculated flow', float('nan'))
        flows.append(df)
        calc_flows.append(cf)

        if pd.notna(df):
            show_daily_flow = True
        if pd.notna(cf):
            show_calc_flow = True

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

    daily_colors = []
    calc_colors = []

    for d in selected_dates:
        daily = extract_daily_data(row['time_series'], d)
        flow_daily = daily.get('Daily flow')
        flow_calc = daily.get('Calculated flow')
        thresholds = extract_thresholds(daily) if row['PolicyType'] == 'WMP' else {}
        q80 = daily.get('Q80')
        q95 = daily.get('Q95')

        if pd.notna(flow_daily):
            color = compliance_color_SWA(row.get('StreamSize'), flow_daily, q80, q95) if row['PolicyType'] == 'SWA' else compliance_color_WMP(flow_daily, thresholds)
        else:
            color = 'gray'
        daily_colors.append(color)

        if pd.notna(flow_calc):
            color = compliance_color_SWA(row.get('StreamSize'), flow_calc, q80, q95) if row['PolicyType'] == 'SWA' else compliance_color_WMP(flow_calc, thresholds)
        else:
            color = 'gray'
        calc_colors.append(color)

    # Wrapper div for popup
    html = "<div style='max-width: 100%;'>"
    html += f"<h4 style='font-size:{font_size};'>{row['station_name']}</h4>"
    html += f"<table style='border-collapse: collapse; border: {border}; font-size:{font_size};'>"
    html += f"<tr><th style='padding:{padding}; border:{border};'>Metric</th>" + \
            ''.join([f"<th style='padding:{padding}; border:{border};'>{d}</th>" for d in selected_dates]) + "</tr>"

    if show_daily_flow:
        html += f"<tr><td style='padding:{padding}; border:{border};'>Daily Flow</td>"
        for i, flow in enumerate(flows):
            code = {'green': '#90ee90', 'yellow': '#fffacd', 'red': '#ffcccb', 'gray': '#d3d3d3'}[daily_colors[i]]
            flow_str = f"{flow:.2f}" if not pd.isna(flow) else "N/A"
            html += f"<td style='padding:{padding}; background:{code}; border:{border}; text-align:center;'>{flow_str}</td>"
        html += "</tr>"

    if show_calc_flow:
        html += f"<tr><td style='padding:{padding}; border:{border};'>Calculated Flow</td>"
        for i, flow in enumerate(calc_flows):
            code = {'green': '#90ee90', 'yellow': '#fffacd', 'red': '#ffcccb', 'gray': '#d3d3d3'}[calc_colors[i]]
            flow_str = f"{flow:.2f}" if not pd.isna(flow) else "N/A"
            html += f"<td style='padding:{padding}; background:{code}; border:{border}; text-align:center;'>{flow_str}</td>"
        html += "</tr>"

    for label in threshold_labels:
        html += f"<tr><td style='padding:{padding}; border:{border};'>{label}</td>"
        for thresholds in threshold_sets:
            val = thresholds.get(label, float('nan'))
            cell = f"{val:.2f}" if not pd.isna(val) else "N/A"
            html += f"<td style='padding:{padding}; border:{border}; text-align:center;'>{cell}</td>"
        html += "</tr>"

    html += "</table>"

    # Plot section
    figsize = (7.5, 3.5)
    fig, ax = plt.subplots(figsize=figsize)

    if show_daily_flow:
        ax.plot(plot_dates, flows, marker='d', color='blue', label='Daily Flow')
    if show_calc_flow:
        ax.plot(plot_dates, calc_flows, marker='d', linestyle='-', color='black', label='Calculated Flow')

    color_map = {
        'Q80': 'green', 'Q90': 'gold', 'Q95': 'red',
        'WCO': 'red', 'IO': 'orange', 'IFN': 'pink',
        'Minimum flow': 'purple', 'Industrial IO': 'brown', 'Non-industrial IO': 'blue',
        'Cutback1': 'yellow', 'Cutback2': 'orange', 'Cutback3': 'purple', 'Cutoff': 'red'
    }

    for label in threshold_labels:
        y = [ts.get(label, float('nan')) for ts in threshold_sets]
        if any(not pd.isna(val) for val in y):
            ax.plot(plot_dates, y, marker='o', linestyle='--', color=color_map.get(label, 'gray'), label=label)

    ax.set_title('Flow vs Threshold', fontsize=14)
    ax.set_ylabel('Flow', fontsize=12)
    fig.autofmt_xdate()
    if threshold_labels or show_daily_flow or show_calc_flow:
        ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    img_data = base64.b64encode(buf.getvalue()).decode()

    html += f"<br><img src='data:image/png;base64,{img_data}' style='width:100%; max-height:300px; display:block; margin: 0 auto;'/>"
    html += "</div>"

    return html

# --- Map rendering ---
def get_most_recent_valid_date(row, dates):
    for d in sorted(dates, reverse=True):
        daily = extract_daily_data(row['time_series'], d)
        if any(pd.notna(daily.get(k)) for k in ['Daily flow', 'Calculated flow']):
            return d
    return None

def render_map():
    # Use LAT and LON from AB_WS_R_StationList.csv columns for map center
  def render_map():
    m = folium.Map(
        location=[merged['LAT'].mean(), merged['LON'].mean()],
        zoom_start=6,
        width='100%',
        height='100%'
    )
    Fullscreen().add_to(m)

    for _, row in merged.iterrows():
        wsc = row['WSC']
        is_diversion = wsc in diversion_tables

        # Filtering logic
        if show_diversion and not is_diversion:
            continue
        if not show_diversion and is_diversion:
            continue

        coords = [row['LAT'], row['LON']]
        date = get_most_recent_valid_date(row, selected_dates)
        if not date:
            continue

        color = get_color_for_date(row, date)
        popup_html = make_popup_html_with_plot(row, selected_dates, show_diversion=show_diversion)
        iframe = IFrame(html=popup_html, width=700, height=500)
        popup = folium.Popup(iframe)

        folium.CircleMarker(
            location=coords,
            radius=7,
            color='blue' if is_diversion else 'black',  # Blue outline for diversion stations
            weight=3 if is_diversion else 2,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup,
            tooltip=row['station_name']
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

