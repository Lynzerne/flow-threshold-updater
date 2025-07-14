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

# --- Paths ---
DATA_DIR = "data"
DIVERSION_DIR = os.path.join(DATA_DIR, "DiversionTables")
STREAM_CLASS_FILE = os.path.join(DATA_DIR, "StreamSizeClassification.csv")
CSV_FILE = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")], reverse=True)[0]

# --- Load data ---
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
            file_path = os.path.join(DIVERSION_DIR, f)

            # Read columns B to E: [Date, Cutback1, Cutback2, Cutback3 or Cutoff]
            df = pd.read_excel(file_path, usecols="B:E")
            df.columns = df.columns.str.strip()

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
@st.cache_data
def generate_popup_cache(merged_df, selected_dates):
    popup_cache = {}
    for _, row in merged_df.iterrows():
        wsc = row['WSC']
        popup_cache[wsc] = {}
        for mode in [True, False]:
            try:
                popup_cache[wsc][mode] = make_popup_html_with_plot(row, selected_dates, show_diversion=mode)
            except Exception as e:
                st.exception(e)
                popup_cache[wsc][mode] = "<p>Error generating popup</p>"
    return popup_cache

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

popup_cache = generate_popup_cache(merged, selected_dates)

# --- Popup HTML builder ---
def make_popup_html_with_plot(row, selected_dates, show_diversion):
    font_size = '15px'
    padding = '6px'
    border = '2px solid black'

    flows = []
    calc_flows = []
    threshold_sets = []
    threshold_labels = set()
    show_daily_flow = False
    show_calc_flow = False

    
    # Ensure selected_dates sorted chronologically
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

    html = "<div style='max-width: 100%;'>"
    html += f"<h4 style='font-size:{font_size};'>{row['station_name']}</h4>"
    html += f"<table style='border-collapse: collapse; border: {border}; font-size:{font_size};'>"
    html += f"<tr><th style='padding:{padding}; border:{border};'>Metric</th>" + \
            ''.join([f"<th style='padding:{padding}; border:{border};'>{d}</th>" for d in selected_dates]) + "</tr>"

    if show_daily_flow:
        html += "<tr><td style='padding:{0}; border:{1}; font-weight:bold;'>Daily Flow</td>".format(padding, border)
        for val, color in zip(flows, daily_colors):
            val_str = f"{val:.2f}" if pd.notna(val) else "NA"
            html += f"<td style='padding:{padding}; border:{border}; background-color:{color};'>{val_str}</td>"
        html += "</tr>"

    if show_calc_flow and any(pd.notna(val) for val in calc_flows):
        html += "<tr><td style='padding:{0}; border:{1}; font-weight:bold;'>Calculated Flow</td>".format(padding, border)
        for val, color in zip(calc_flows, calc_colors):
            val_str = f"{val:.2f}" if pd.notna(val) else "NA"
            html += f"<td style='padding:{padding}; border:{border}; background-color:{color};'>{val_str}</td>"
        html += "</tr>"

    for label in threshold_labels:
        html += f"<tr><td style='padding:{padding}; border:{border}; font-weight:bold;'>{label}</td>"
        for t_set in threshold_sets:
            val = t_set.get(label, float('nan'))
            val_str = f"{val:.2f}" if pd.notna(val) else "NA"
            html += f"<td style='padding:{padding}; border:{border};'>{val_str}</td>"
        html += "</tr>"

    html += "</table><br>"


    # --- Plot flow series with thresholds ---
    fig, ax = plt.subplots(figsize=(8, 3))  # Correct placement

    ax.plot(plot_dates, flows, 'o-', label='Daily Flow', color='tab:blue', linewidth=2)

    if any(pd.notna(val) for val in calc_flows):
        ax.plot(plot_dates, calc_flows, 's--', label='Calculated Flow', color='tab:green', linewidth=2)

    threshold_colors = {
        'Cutback1': 'gold',
        'Cutback2': 'orange',
        'Cutback3': 'purple',
        'Cutoff': 'red',
        'IO': 'orange', 
        'WCO': 'crimson', 
        'Q80': 'green',
        'Q90': 'yellow',
        'Q95': 'orange',
        'Minimum flow': 'red',
        'IFN': 'red',
    }

    for label in threshold_labels:
        threshold_vals = [t.get(label, float('nan')) for t in threshold_sets]
        if all(pd.isna(threshold_vals)):
            continue
        color = threshold_colors.get(label, 'gray')
        ax.plot(plot_dates, threshold_vals, linestyle='--', label=label, color=color, linewidth=2)

    ax.set_ylabel('Flow')
    ax.legend(fontsize=8)
    ax.set_title('Flow and Thresholds Over Time')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    html += f"<img src='data:image/png;base64,{img_base64}' style='max-width:100%; height:auto;'>"

    return html

# --- Map rendering ---

def get_most_recent_valid_date(row, dates):
    for d in sorted(dates, reverse=True):
        daily = extract_daily_data(row['time_series'], d)
        if any(pd.notna(daily.get(k)) for k in ['Daily flow', 'Calculated flow']):
            return d
    return None

def render_map():
    m = folium.Map(
        location=[merged['LAT'].mean(), merged['LON'].mean()],
        zoom_start=6,
        width='100%',
        height='1200px'
    )
    Fullscreen().add_to(m)

    for _, row in merged.iterrows():
        coords = [row['LAT'], row['LON']]

        if show_diversion and row['WSC'] not in diversion_tables:
            continue

        date = get_most_recent_valid_date(row, selected_dates)
        if not date:
            continue

        color = get_color_for_date(row, date)
        popup_html = popup_cache.get(row['WSC'], {}).get(show_diversion, "<p>No data</p>")
        iframe = IFrame(html=popup_html, width=700, height=500)
        popup = folium.Popup(iframe)

        border_color = 'blue' if show_diversion and row['WSC'] in diversion_tables else 'black'

        folium.CircleMarker(
            location=coords,
            radius=7,
            color=border_color,
            weight=3,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup,
            tooltip=row['station_name']
        ).add_to(m)

    return m

# --- Display ---
st.title("Alberta Flow Threshold Viewer")


if not selected_dates:
    st.warning("No data available for the selected date range.")
else:
    m = render_map()
    map_html = m.get_root().render()
    st.components.v1.html(map_html, height=800, scrolling=True)
