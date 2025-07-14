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

# Removed: st.cache_data.clear()  # Clearing cache every run causes reloads

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

            df = pd.read_excel(file_path, usecols="B:E")
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

def make_popup_html_with_plot(row, selected_dates, show_diversion):
    font_size = '15px'
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

    html = f"<div style='max-width: 100%;'><h4 style='font-size:{font_size};'>{row['station_name']}</h4>"
    html += f"<table style='border-collapse: collapse; border: {border}; font-size:{font_size};'>"
    html += "<tr><th style='padding:{0}; border:{1};'>Metric</th>{2}</tr>".format(
        padding, border,
        ''.join([f"<th style='padding:{padding}; border:{border};'>{d}</th>" for d in selected_dates])
    )

    if show_daily_flow:
        html += "<tr><td style='padding:{0}; border:{1}; font-weight:bold;'>Daily Flow</td>".format(padding, border)
        html += ''.join([
            f"<td style='padding:{padding}; border:{border}; background-color:{c};'>{f'{v:.2f}' if pd.notna(v) else 'NA'}</td>"
            for v, c in zip(flows, daily_colors)
        ])
        html += "</tr>"

    if show_calc_flow and any(pd.notna(val) for val in calc_flows):
        html += "<tr><td style='padding:{0}; border:{1}; font-weight:bold;'>Calculated Flow</td>".format(padding, border)
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

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(plot_dates, flows, 'o-', label='Daily Flow', color='tab:blue', linewidth=2)
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

    html += f"<img src='data:image/png;base64,{img_base64}' style='max-width:100%; height:auto;'>"
    html += "</div>"

    return html

def get_date_hash(dates):
    date_str = ",".join(sorted(dates))
    return hashlib.md5(date_str.encode()).hexdigest()

@st.cache_data(show_spinner=True)
def generate_popup_cache(merged_df, selected_dates, show_diversion):
    popup_cache = {}
    for _, row in merged_df.iterrows():
        wsc = row['WSC']
        try:
            popup_cache[wsc] = make_popup_html_with_plot(row, selected_dates, show_diversion)
        except Exception as e:
            popup_cache[wsc] = "<p>Error generating popup</p>"
    return popup_cache

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

        if 'selected_dates' not in st.session_state:
            selected_dates = valid_dates[-3:]  # default last 3 days
        else:
            selected_dates = st.session_state.selected_dates

        most_recent_date = get_most_recent_valid_date(row, selected_dates)
        color = get_color_for_date(row, most_recent_date) if most_recent_date else 'gray'

        wsc = row['WSC']
        html = st.session_state.popup_cache.get(wsc, "<p>No data</p>")

        iframe = IFrame(html, width=400, height=400)
        popup = folium.Popup(iframe, max_width=500)
        folium.CircleMarker(
            location=coords,
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup
        ).add_to(m)

    return m

# --- Sidebar ---
st.sidebar.header("Date Range")

min_date = datetime.strptime(valid_dates[0], "%Y-%m-%d").date()
max_date = datetime.strptime(valid_dates[-1], "%Y-%m-%d").date()

default_start = max_date - timedelta(days=7)
default_end = max_date

default_dates = [d.strftime('%Y-%m-%d') for d in pd.date_range(default_start, default_end)]
default_dates = [d for d in default_dates if d in valid_dates]

if len(default_dates) > 3:
    default_dates = default_dates[:3]

if not default_dates:
    default_dates = valid_dates[-3:] if len(valid_dates) >= 3 else valid_dates

selected_dates = st.sidebar.multiselect(
    "Select Dates (max 3)", valid_dates,
    default=default_dates,
    max_selections=3
)

if not selected_dates:
    st.sidebar.error("Please select at least one date.")
    st.stop()

# Assume show_diversion was set earlier from sidebar radio
st.session_state.selected_dates = selected_dates
st.session_state.show_diversion = show_diversion

# Generate popup cache only if needed
popup_cache_key = (tuple(selected_dates), show_diversion)
if 'popup_cache_key' not in st.session_state or st.session_state.popup_cache_key != popup_cache_key:
    with st.spinner("Generating popups..."):
        st.session_state.popup_cache = generate_popup_cache(merged, selected_dates, show_diversion)
        st.session_state.popup_cache_key = popup_cache_key

# Render map
folium_map = render_map()

st.markdown("<h2>Stations Map</h2>", unsafe_allow_html=True)
st_data = st_folium(folium_map, width=1000, height=600)


