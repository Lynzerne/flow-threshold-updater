import os
import json
import re
import time
import requests
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta, date
from urllib.parse import urljoin

from fetch_erv_licences import fetch_waterlicence_authorization_raw

try:
    import pyarrow
except ImportError:
    raise ImportError("pyarrow is required to read/write Parquet files. Please install it via pip.")

pd.options.io.parquet.engine = "pyarrow"

# ============================================================
# CONFIG
# ============================================================
station_list_csv = "data/AB_WS_R_StationList.csv"
output_parquet = "data/WS_R_master_daily.parquet"
base_url_template = "https://rivers.alberta.ca/apps/Basins/data/figures/river/abrivers/stationdata/WS_R_{}_table.json"

# Water licence files
licence_csv_path = "data/WaterLicence_Authorization.csv"
doc_links_csv_path = "data/WaterLicence_DocumentLinks.csv"

# Date Window
today = datetime.today().date()
lookback_days = 7  # kept for readability

# ============================================================
# HELPERS - DOCUMENT LINK RESOLUTION
# ============================================================
ERV_BASE = "https://geospatial.alberta.ca"
ERV_WATER_ACT_URL = "https://geospatial.alberta.ca/erv-water-act/?page=Water-Act&authorizationNumber={auth}"

# Looks for direct DRAS document links in page HTML
DRAS_DOC_PATTERN = re.compile(
    r'(/services/DRASDocuments/Document/Get\?documentType=WL&authorizationNumber=(\d+)&id=(\d+))',
    re.IGNORECASE
)

def build_erv_link(auth_number):
    auth_str = str(auth_number).strip()
    return ERV_WATER_ACT_URL.format(auth=auth_str)

def extract_direct_pdf_from_html(html, expected_auth=None):
    """
    Tries to find a direct WL document link in HTML.
    Returns dict with pdf_url, authorizationNumber, document_id if found.
    """
    matches = DRAS_DOC_PATTERN.findall(html)
    if not matches:
        return None

    # Prefer matching authorization number if available
    for full_path, auth_num, doc_id in matches:
        if expected_auth is None or str(auth_num) == str(expected_auth):
            return {
                "PdfHyperlink": urljoin(ERV_BASE, full_path),
                "AuthorizationNumber": str(auth_num),
                "DocumentID": str(doc_id),
                "PdfFound": True,
                "PdfSource": "Direct DRAS document"
            }

    # Fallback to first match
    full_path, auth_num, doc_id = matches[0]
    return {
        "PdfHyperlink": urljoin(ERV_BASE, full_path),
        "AuthorizationNumber": str(auth_num),
        "DocumentID": str(doc_id),
        "PdfFound": True,
        "PdfSource": "Direct DRAS document (first match fallback)"
    }

def resolve_pdf_link_for_auth(auth_number, session, timeout=30):
    """
    Attempts to resolve a direct PDF link for a water licence authorization.
    Falls back to ERV landing page if direct PDF is not found.
    """
    auth_str = str(auth_number).strip()
    erv_link = build_erv_link(auth_str)

    try:
        resp = session.get(erv_link, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return {
            "AuthorizationNumber": auth_str,
            "PdfHyperlink": erv_link,
            "DocumentID": None,
            "PdfFound": False,
            "PdfSource": f"ERV fallback (request failed: {str(e)[:150]})",
            "LastChecked": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }

    doc_info = extract_direct_pdf_from_html(html, expected_auth=auth_str)

    if doc_info:
        doc_info["LastChecked"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return doc_info

    return {
        "AuthorizationNumber": auth_str,
        "PdfHyperlink": erv_link,
        "DocumentID": None,
        "PdfFound": False,
        "PdfSource": "ERV fallback (direct PDF not found in HTML)",
        "LastChecked": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }

def get_authorization_column(df):
    """
    Tries common authorization-number column names.
    """
    auth_col_candidates = [
        "AuthorizationNumber",
        "AUTHORIZATION_NUMBER",
        "Authorization Number",
        "authorizationNumber"
    ]
    for c in auth_col_candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find authorization number column. Checked: {auth_col_candidates}"
    )

def update_document_link_lookup(
    licence_csv_path,
    doc_links_csv_path,
    sleep_seconds=0.5
):
    """
    Reads fetched licence CSV, resolves direct document links where possible,
    and updates a separate lookup CSV.
    """
    if not os.path.exists(licence_csv_path):
        print(f"Licence CSV not found: {licence_csv_path}")
        return

    licences_df = pd.read_csv(licence_csv_path, dtype=str)
    auth_col = get_authorization_column(licences_df)

    licences_df[auth_col] = licences_df[auth_col].astype(str).str.strip()
    licences_df = licences_df[
        licences_df[auth_col].notna() &
        (licences_df[auth_col] != "") &
        (licences_df[auth_col].str.lower() != "nan")
    ].copy()

    if licences_df.empty:
        print("No valid authorization numbers found in licence CSV.")
        return

    unique_auths = sorted(licences_df[auth_col].dropna().unique().tolist())
    print(f"Found {len(unique_auths)} unique authorization numbers in licence CSV.")

    # Load existing lookup if present
    if os.path.exists(doc_links_csv_path):
        links_df = pd.read_csv(doc_links_csv_path, dtype=str)
        print(f"Loaded existing document link lookup: {doc_links_csv_path}")
    else:
        links_df = pd.DataFrame(columns=[
            "AuthorizationNumber",
            "PdfHyperlink",
            "DocumentID",
            "PdfFound",
            "PdfSource",
            "LastChecked"
        ])
        print("No existing document link lookup found; starting fresh.")

    if not links_df.empty:
        links_df["AuthorizationNumber"] = links_df["AuthorizationNumber"].astype(str).str.strip()
        already_done = set(
            links_df.loc[
                links_df["AuthorizationNumber"].notna() &
                (links_df["AuthorizationNumber"] != "") &
                (links_df["AuthorizationNumber"].str.lower() != "nan"),
                "AuthorizationNumber"
            ].tolist()
        )
    else:
        already_done = set()

    to_resolve = [a for a in unique_auths if a not in already_done]

    print(f"Already resolved: {len(already_done)}")
    print(f"Need to resolve: {len(to_resolve)}")

    if not to_resolve:
        print("No new authorizations to resolve.")
        return

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; WaterLicenceLinkResolver/1.0)"
    })

    new_results = []
    for i, auth in enumerate(to_resolve, start=1):
        print(f"[{i}/{len(to_resolve)}] Resolving authorization {auth}...")
        result = resolve_pdf_link_for_auth(auth, session=session)
        new_results.append(result)
        time.sleep(sleep_seconds)

    new_results_df = pd.DataFrame(new_results)

    if links_df.empty:
        updated_links_df = new_results_df
    else:
        updated_links_df = pd.concat([links_df, new_results_df], ignore_index=True)
        updated_links_df.drop_duplicates(subset=["AuthorizationNumber"], keep="last", inplace=True)

    updated_links_df.to_csv(doc_links_csv_path, index=False)
    print(f"Updated document link lookup saved: {doc_links_csv_path}")

def enrich_licence_csv_with_doc_links(
    licence_csv_path,
    doc_links_csv_path,
    output_csv_path=None
):
    """
    Merges document-link lookup back onto the licence CSV.
    If output_csv_path is None, overwrites the licence CSV.
    """
    if output_csv_path is None:
        output_csv_path = licence_csv_path

    if not os.path.exists(licence_csv_path):
        print(f"Licence CSV not found: {licence_csv_path}")
        return

    if not os.path.exists(doc_links_csv_path):
        print(f"Document link lookup not found: {doc_links_csv_path}")
        return

    licences_df = pd.read_csv(licence_csv_path, dtype=str)
    links_df = pd.read_csv(doc_links_csv_path, dtype=str)

    auth_col = get_authorization_column(licences_df)

    licences_df[auth_col] = licences_df[auth_col].astype(str).str.strip()
    links_df["AuthorizationNumber"] = links_df["AuthorizationNumber"].astype(str).str.strip()

    merged_df = licences_df.merge(
        links_df,
        how="left",
        left_on=auth_col,
        right_on="AuthorizationNumber"
    )

    merged_df.to_csv(output_csv_path, index=False)
    print(f"Licence CSV enriched with document links: {output_csv_path}")

# ============================================================
# PART 1 - FETCH / UPDATE RIVER STATION DATA
# ============================================================
stns = pd.read_csv(station_list_csv)
required_cols = ['WSC', 'LAT', 'LON']
for c in required_cols:
    if c not in stns.columns:
        raise ValueError(f"Missing column '{c}' in station list CSV")

if os.path.exists(output_parquet):
    master_df = pd.read_parquet(output_parquet, engine="pyarrow")
    master_df['Date'] = pd.to_datetime(master_df['Date'], errors='coerce').dt.date
    print(f"Loaded existing master dataset with {len(master_df)} rows.")
else:
    master_df = pd.DataFrame()
    print("No existing master dataset found; starting fresh.")

all_data = []

for _, row in stns.iterrows():
    station_id = row['WSC']
    url = base_url_template.format(station_id)

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Failed to fetch data for {station_id}: {e}")
        continue

    site_name = data.get('station_name', station_id)

    columns = ['Date']
    for md in data.get('ts_metadata', []):
        columns.append(md.get('ts_label', md.get('parameter_name', 'unknown')))

    # Deduplicate columns
    label_counts = Counter()
    deduped_columns = []
    for col in columns:
        count = label_counts[col]
        label_counts[col] += 1
        if count == 0:
            deduped_columns.append(col)
        else:
            deduped_columns.append(f"{col}_{count}")

    rows = []
    for entry in data.get('data', []):
        rows.append(entry.get('values', []))

    if not rows:
        print(f"No data rows for {station_id}, skipping.")
        continue

    df = pd.DataFrame(rows, columns=deduped_columns)
    if 'Date' not in df.columns:
        print(f"No 'Date' column in data for {station_id}, skipping.")
        continue

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    df = df[df['Date'] <= today]
    df = df.dropna(subset=['Date'])

    if df.empty:
        print(f"No recent (today or past) data for {station_id}, skipping.")
        continue

    df['station_no'] = station_id
    df['station_name'] = site_name
    df['lon'] = row['LON']
    df['lat'] = row['LAT']

    all_data.append(df)

# --- Merge/Update Master Dataset ---
if all_data:
    new_data_df = pd.concat(all_data, ignore_index=True)

    merge_keys = ['station_no', 'Date']
    for col in merge_keys:
        if col not in new_data_df.columns:
            new_data_df[col] = pd.NA

    metadata_cols = ['station_no', 'station_name', 'Date', 'lon', 'lat']

    all_possible_columns = list(set(master_df.columns.tolist() + new_data_df.columns.tolist()))
    ts_cols = [col for col in all_possible_columns if col not in metadata_cols]

    if not master_df.empty:
        updated_rows_list = []

        master_dict = master_df.set_index(merge_keys).to_dict('index')

        master_df_keys = master_df[merge_keys].copy()
        master_df_keys['Date'] = pd.to_datetime(master_df_keys['Date'], errors='coerce').dt.date
        master_df_keys = master_df_keys.dropna(subset=['Date'])

        new_data_df_keys = new_data_df[merge_keys].copy()
        new_data_df_keys['Date'] = pd.to_datetime(new_data_df_keys['Date'], errors='coerce').dt.date
        new_data_df_keys = new_data_df_keys.dropna(subset=['Date'])

        all_unique_keys_df = pd.DataFrame(columns=merge_keys)
        if not master_df_keys.empty:
            all_unique_keys_df = pd.concat([all_unique_keys_df, master_df_keys], ignore_index=True)
        if not new_data_df_keys.empty:
            all_unique_keys_df = pd.concat([all_unique_keys_df, new_data_df_keys], ignore_index=True)
        all_unique_keys_df.drop_duplicates(inplace=True)

        for _, row_key in all_unique_keys_df.iterrows():
            stn = row_key['station_no']
            dt = row_key['Date']

            existing_master_row_data = master_dict.get((stn, dt))

            new_row_df = new_data_df[
                (new_data_df['station_no'] == stn) &
                (new_data_df['Date'] == dt)
            ]
            new_scrape_row_data = new_row_df.iloc[0].to_dict() if not new_row_df.empty else None

            current_row_data = {col: None for col in all_possible_columns}
            current_row_data['is_revised'] = False

            if existing_master_row_data is not None:
                current_row_data.update(existing_master_row_data)

            if new_scrape_row_data is not None:
                current_row_data.update(new_scrape_row_data)

            current_row_data['station_no'] = stn
            current_row_data['Date'] = dt

            is_row_revised = False

            # Update time-series columns if old was null and new is populated
            for col in ts_cols:
                old_val = current_row_data.get(col)
                new_val = new_scrape_row_data.get(col) if new_scrape_row_data is not None else None

                if pd.isna(old_val) and pd.notna(new_val):
                    current_row_data[col] = new_val
                    is_row_revised = True

            if new_scrape_row_data is not None:
                current_row_data['station_name'] = new_scrape_row_data.get('station_name', current_row_data.get('station_name'))
                current_row_data['lon'] = new_scrape_row_data.get('lon', current_row_data.get('lon'))
                current_row_data['lat'] = new_scrape_row_data.get('lat', current_row_data.get('lat'))

            current_row_data['is_revised'] = is_row_revised
            updated_rows_list.append(current_row_data)

        updated_df = pd.DataFrame(updated_rows_list)
        updated_df['Date'] = pd.to_datetime(updated_df['Date'], errors='coerce').dt.date

        for col in ts_cols:
            if col in updated_df.columns:
                updated_df[col] = pd.to_numeric(updated_df[col], errors='coerce')
            else:
                updated_df[col] = pd.NA

        for col in ['station_no', 'station_name', 'lon', 'lat', 'is_revised']:
            if col not in updated_df.columns:
                updated_df[col] = pd.NA

        updated_keys_df = updated_df[merge_keys].drop_duplicates()
        master_df = master_df[
            ~master_df.set_index(merge_keys).index.isin(updated_keys_df.set_index(merge_keys).index)
        ]
        master_df = pd.concat([master_df, updated_df], ignore_index=True)

    else:
        new_data_df['is_revised'] = False
        master_df = new_data_df
        master_df['Date'] = pd.to_datetime(master_df['Date'], errors='coerce').dt.date

    master_df.sort_values(merge_keys, inplace=True)
    master_df.to_parquet(output_parquet, index=False, engine="pyarrow")
    print(f"Master dataset saved to {output_parquet}")

else:
    print("No new data collected from any stations.")

# ============================================================
# PART 2 - STITCH / ROLLING GEOJSON
# ============================================================
iday = date.today().strftime('%Y-%m-%d')
station_list_csv = "data/AB_WS_R_StationList.csv"
master_parquet_path = "data/WS_R_master_daily.parquet"
master_geojson_path = "data/AB_WS_R_stations.geojson"

stns = pd.read_csv(station_list_csv)
stns['WSC'] = stns['WSC'].astype(str).str.strip()

if os.path.exists(master_geojson_path):
    print(f"📄 Loading existing GeoJSON: {master_geojson_path}")
    with open(master_geojson_path, 'r') as f:
        geojson = json.load(f)
else:
    print("⚙️ GeoJSON skeleton not found. Building it...")
    geojson = {"type": "FeatureCollection", "features": []}
    for _, row in stns.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row['LON'], row['LAT']]},
            "properties": {
                "station_no": row['WSC'],
                "station_name": None,
                "time_series": []
            }
        }
        geojson['features'].append(feature)
    print("✅ GeoJSON skeleton created.")

master_df = pd.read_parquet(master_parquet_path, engine="pyarrow")
master_df['station_no'] = master_df['station_no'].astype(str).str.strip()

TS_KEEP = [
    "Daily flow",
    "Calculated flow",
    "Q80",
    "Q90",
    "Q95",
    "WCO",
    "IO",
    "Minimum flow",
    "Industrial IO",
    "Non-industrial IO",
    "IFN",
    "is_revised",
]

metadata_cols = {"station_no", "station_name", "Date", "lon", "lat"}
ts_cols = [c for c in TS_KEEP if c in master_df.columns]

feature_map = {feat['properties']['station_no']: feat for feat in geojson['features']}

for stn_id, group_df in master_df.groupby('station_no'):
    if stn_id not in feature_map:
        continue

    feat = feature_map[stn_id]

    static_keys = {'station_no', 'station_name', 'lat', 'lon', 'time_series'}
    feat['properties'] = {k: v for k, v in feat['properties'].items() if k in static_keys}

    if 'Date' in group_df.columns:
        group_df = group_df.copy()
        group_df['Date'] = pd.to_datetime(group_df['Date'], errors='coerce')
        group_df = group_df.dropna(subset=['Date']).sort_values('Date')

    if group_df.empty:
        continue

    latest_record = group_df.iloc[-1]

    feat['properties']['station_no'] = stn_id
    feat['properties']['station_name'] = str(latest_record.get('station_name', stn_id))
    feat['properties']['lat'] = float(latest_record.get('lat', feat['properties'].get('lat', 0)))
    feat['properties']['lon'] = float(latest_record.get('lon', feat['properties'].get('lon', 0)))

    timeseries = []
    for _, row in group_df.iterrows():
        d = row.get('Date', None)
        if pd.isna(d):
            continue

        d_str = pd.to_datetime(d).strftime('%Y-%m-%d')
        ts_entry = {'date': d_str}

        for col in ts_cols:
            val = row.get(col)
            if pd.notnull(val) and str(val).strip() != '':
                if col == 'is_revised':
                    ts_entry[col] = bool(val)
                else:
                    try:
                        ts_entry[col] = float(val)
                    except (ValueError, TypeError):
                        pass

        timeseries.append(ts_entry)

    feat['properties']['time_series'] = timeseries

with open(master_geojson_path, 'w') as f:
    json.dump(geojson, f, indent=2)

print(f"🔁 Rolling master GeoJSON updated: {master_geojson_path}")

# ============================================================
# PART 3 - FETCH AER WATER LICENCE DATA
# ============================================================
print("Starting ERV licence download...")

parent_query = (
    "AUTHORIZATION_STATUS='AUTAC' AND "
    "AUTHORIZATION_REGULATOR='AER' AND "
    "ALLOCATION_RIVER_SUB_BASIN_CO LIKE '%05CC%'"
)

fetch_waterlicence_authorization_raw(
    parent_query,
    out_dir="data",
    out_csv_name="WaterLicence_Authorization.csv"
)

print("ERV licence download complete.")

# ============================================================
# PART 4 - RESOLVE / UPDATE DOCUMENT LINKS
# ============================================================
print("Starting document-link enrichment...")

update_document_link_lookup(
    licence_csv_path=licence_csv_path,
    doc_links_csv_path=doc_links_csv_path,
    sleep_seconds=0.5
)

# Overwrite WaterLicence_Authorization.csv with the merged/enriched version
enrich_licence_csv_with_doc_links(
    licence_csv_path=licence_csv_path,
    doc_links_csv_path=doc_links_csv_path,
    output_csv_path=licence_csv_path
)

print("Document-link enrichment complete.")
