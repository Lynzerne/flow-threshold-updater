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

    # Plot with fixed image encoding
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(plot_dates, flows, 'o-', label='Daily Flow', color='tab:blue', linewidth=2)
    if any(pd.notna(val) for val in calc_flows):
        ax.plot(plot_dates, calc_flows, 's--', label='Calculated Flow', color='tab:green', linewidth=2)

    threshold_colors = {
        'Cutback1': 'gold', 'Cutback2': 'orange', 'Cutback3': 'purple', 'Cutoff': 'red',
        'IO': 'brown', 'WCO': 'black', 'Minimum flow': 'gray', 'Industrial IO': 'cyan',
        'Non-industrial IO': 'pink', 'IFN': 'darkgreen', 'Q80': 'green', 'Q90': 'darkblue', 'Q95': 'darkred'
    }

    for label in threshold_labels:
        for thresholds in threshold_sets:
            val = thresholds.get(label)
            if pd.notna(val):
                ax.axhline(y=val, color=threshold_colors.get(label, 'black'), linestyle='--', alpha=0.7)
                break

    ax.set_xlabel('Date')
    ax.set_ylabel('Flow')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)
    fig.autofmt_xdate()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode()
    html += f"<img src='data:image/png;base64,{encoded}' style='max-width: 100%; height: auto;'><br></div>"

    return html

# --- Generate popup caches for both diversion modes ---
def generate_all_popups(merged, selected_dates):
    cache_no_diversion = {}
    cache_diversion = {}

    for idx, row in merged.iterrows():
        # No diversion
        cache_no_diversion[row['WSC']] = make_popup_html_with_plot(row, selected_dates, False)
        # With diversion
        cache_diversion[row['WSC']] = make_popup_html_with_plot(row, selected_dates, True)

    return cache_no_diversion, cache_diversion

# --- Map rendering ---
def render_map(popup_cache, show_diversion):
    m = folium.Map(location=[55.000, -114.000], zoom_start=7)
    Fullscreen(position='topright').add_to(m)

    for idx, row in merged.iterrows():
        coord = (row['latitude'], row['longitude'])
        color = get_color_for_date(row, selected_dates[0])  # Color for first selected date

        popup_html = popup_cache.get(row['WSC'], "<b>No data</b>")
        iframe = IFrame(html=popup_html, width=350, height=350)
        popup = folium.Popup(iframe, max_width=350)

        folium.CircleMarker(
            location=coord,
            radius=7,
            popup=popup,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=1,
        ).add_to(m)

    return m

# --- Streamlit UI ---
st.title("Flow Compliance Map")

# Date range picker
min_date = datetime.strptime(valid_dates[0], '%Y-%m-%d')
max_date = datetime.strptime(valid_dates[-1], '%Y-%m-%d')
selected_dates = st.slider(
    "Select date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

date_range = pd.date_range(selected_dates[0], selected_dates[1])
selected_dates = [d.strftime('%Y-%m-%d') for d in date_range]

# Diversion toggle
show_diversion = st.checkbox("Show Diversion")

# Cache popup htmls on first run or date change
def get_date_range_hash(dates):
    m = hashlib.md5()
    for d in dates:
        m.update(d.encode())
    return m.hexdigest()

date_hash = get_date_range_hash(selected_dates)

if 'popup_cache_no_diversion' not in st.session_state or \
   'popup_cache_diversion' not in st.session_state or \
   st.session_state.get('date_hash', '') != date_hash:

    with st.spinner("Generating popup caches (takes a moment)..."):
        no_div_cache, div_cache = generate_all_popups(merged, selected_dates)
        st.session_state.popup_cache_no_diversion = no_div_cache
        st.session_state.popup_cache_diversion = div_cache
        st.session_state.date_hash = date_hash

# Select correct popup cache
popup_cache = st.session_state.popup_cache_diversion if show_diversion else st.session_state.popup_cache_no_diversion

# Render map
m = render_map(popup_cache, show_diversion)
map_html = m.get_root().render()

st.components.v1.html(map_html, height=800, scrolling=True)
