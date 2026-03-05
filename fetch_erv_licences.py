# fetch_erv_licences.py

import time
import json
import zipfile
from pathlib import Path

import requests

GP_BASE = "https://geospatial.alberta.ca/charon/rest/services/dras/dras_erv_extraction/GPServer/extract"


def fetch_waterlicence_authorization_raw(
    parent_query: str,
    out_dir: str = "data",
    out_csv_name: str = "WaterLicence_Authorization.csv",
    timeout_sec: int = 240,
    poll_every_sec: float = 1.0,
):
    """
    Downloads the ERV Water Act Public extract (CSV) for WATERACT-AUTHORIZATION using the same backend call as the ERV UI.
    Saves a single raw CSV to: <out_dir>/<out_csv_name>
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_dir / out_csv_name
    tmp_zip_path = out_dir / "_tmp_erv_extract.zip"

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://geospatial.alberta.ca/",
        "Origin": "https://geospatial.alberta.ca",
    })

    # --- Build the same commandArgs the ERV app sends ---
    command_args = {
        "ervId": "WaterActPublic",
        "formId": "WATERACT-AUTHORIZATION",
        "extractType": "CSV",
        "childQuery": None,
        "parentQuery": parent_query,
        "geometry": None,
        "token": None,
    }

    # --- 1) Submit the extract job ---
    submit = session.get(
        f"{GP_BASE}/submitJob",
        params={
            "commandName": "Extract",
            "commandArgs": json.dumps(command_args, separators=(",", ":")),
            "f": "json",
        },
        timeout=60,
    )
    submit.raise_for_status()
    submit_json = submit.json()

    job_id = submit_json.get("jobId") or submit_json.get("jobID")
    if not job_id:
        raise RuntimeError(f"No jobId returned from ERV submitJob. Response: {submit_json}")

    print(f"✅ ERV extract job submitted. jobId={job_id}")

    # --- 2) Poll job status until it finishes ---
    job_url = f"{GP_BASE}/jobs/{job_id}"
    deadline = time.time() + timeout_sec
    last_status = None

    while time.time() < deadline:
        jr = session.get(job_url, params={"f": "json"}, timeout=60)
        jr.raise_for_status()
        job_json = jr.json()

        status = job_json.get("jobStatus")
        if status != last_status:
            print(f"⏳ ERV job status: {status}")
            last_status = status

        if status == "esriJobSucceeded":
            break
        if status == "esriJobFailed":
            raise RuntimeError(f"ERV job failed: {job_json}")

        time.sleep(poll_every_sec)
    else:
        raise TimeoutError("ERV job timed out waiting for completion.")

    # --- 3) Get the output file URL ---
    # Try common result parameter names first
    file_url = None
    for param in ["output", "result", "file", "Output", "Result", "File"]:
        rr = session.get(f"{GP_BASE}/jobs/{job_id}/results/{param}", params={"f": "json"}, timeout=60)
        if rr.status_code != 200:
            continue
        data = rr.json()
        val = data.get("value")
        if isinstance(val, dict) and isinstance(val.get("url"), str):
            file_url = val["url"]
            break
        if isinstance(val, str) and val.startswith("http"):
            file_url = val
            break

    # Fallback: endpoint you saw in Network
    if not file_url:
        rr = session.get(f"{GP_BASE}/file", params={"returnType": "data", "f": "json"}, timeout=60)
        rr.raise_for_status()
        meta = rr.json()
        file_url = meta.get("url") or meta.get("href")

    if not file_url:
        raise RuntimeError("Could not locate ERV download URL from job results.")

    print("⬇️ Downloading ERV extract...")

    # --- 4) Download the zip (ERV usually delivers a zip containing the CSV) ---
    with session.get(file_url, stream=True, timeout=180) as dl:
        dl.raise_for_status()
        with open(tmp_zip_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    # --- 5) Extract CSV from zip and save as the exact filename your dashboard expects ---
    extracted_csv = None
    with zipfile.ZipFile(tmp_zip_path, "r") as z:
        # Find the first CSV inside
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("ERV extract zip contained no CSV files.")

        # Extract the first CSV to temp location
        extracted_name = csv_names[0]
        z.extract(extracted_name, path=out_dir)
        extracted_csv = out_dir / extracted_name

    # Overwrite / update the stable "raw" file your dashboard uses
    out_csv_path.write_bytes(extracted_csv.read_bytes())
    assert out_csv_path.exists(), f"Expected output missing: {out_csv_path}"

    # Clean up temporary files
    try:
        extracted_csv.unlink(missing_ok=True)
    except Exception:
        pass
    tmp_zip_path.unlink(missing_ok=True)

    print(f"✅ Saved raw licence CSV to: {out_csv_path}")
    return str(out_csv_path)


if __name__ == "__main__":
    # Your current ERV filter (05CC)
    parent_query = (
        "AUTHORIZATION_STATUS='AUTAC' AND "
        "AUTHORIZATION_REGULATOR='AER' AND "
        "ALLOCATION_RIVER_SUB_BASIN_CO LIKE '%05CC%'"
    )

    fetch_waterlicence_authorization_raw(parent_query)
