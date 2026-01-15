import json
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentinelhub import SentinelHubRequest, DataCollection, MimeType

from utils import (
    DATA_PROCESSED, DATA_DATASET, IMG_SIZE, SLEEP,
    get_sh_config, has_credentials, make_bbox, analyze_patch_scl, 
    load_api_keys, make_config_for_key
)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

MAX_CLOUD_PCT = 0.30
MIN_VALID_PCT = 0.95
MAX_WATER_PCT = 0.30

N_LIMIT = None

# ═══════════════════════════════════════════════════════════════
# EVALSCRIPT
# ═══════════════════════════════════════════════════════════════

EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{bands: ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"]}],
        output: {bands: 8, sampleType: "FLOAT32"}
    };
}
function evaluatePixel(s) {
    return [s.B02, s.B03, s.B04, s.B08, s.B11, s.B12, s.SCL, s.dataMask];
}
"""

# ═══════════════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════════════

def fetch_patch(config, lat, lon, s2_dt_str):
    dt = pd.to_datetime(s2_dt_str)
    t0 = dt.isoformat().replace("+00:00", "Z")
    t1 = (dt + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(t0, t1)
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=make_bbox(lat, lon),
        size=(IMG_SIZE, IMG_SIZE),
        config=config
    )
    
    img = request.get_data()[0]
    return {
        "bands": img[:, :, :6],
        "scl": img[:, :, 6].astype(np.uint8),
        "mask": img[:, :, 7].astype(np.uint8)
    }


def _download_one(row_dict, label, save_dir, config, existing_cache):
    """Download + QC for a single row. Returns (result, reason)."""
    latitude = row_dict["latitude"]
    longitude = row_dict["longitude"]
    s2_dt = row_dict["s2_dt"]
    event_id = str(row_dict.get("event_id", f"{label}_unk"))
    name = f"{label}_{event_id}"

    # Fast cache check using pre-built set
    if name in existing_cache:
        return "ok", "cached"

    try:
        data = fetch_patch(config, latitude, longitude, s2_dt)
        analysis = analyze_patch_scl(data["bands"], data["scl"], data["mask"])

        if analysis["valid_pct"] < MIN_VALID_PCT:
            return "valid", "low_valid"
        if analysis["cloud_pct"] > MAX_CLOUD_PCT:
            return "cloud", "cloudy"
        if analysis["water_pct"] > MAX_WATER_PCT:
            return "water", "water"

        delta_val = row_dict.get("delta", row_dict.get("delta_h", None))
        metadata = {
            "event_id": event_id,
            "label": label,
            "s2_dt": str(row_dict.get("s2_dt", "")),
            "firms_dt": str(row_dict.get("firms_dt", "")) if "firms_dt" in row_dict else "",
            "delta_h": float(delta_val) if delta_val is not None else None,
            "item_id": row_dict.get("item_id"),
            "tile": row_dict.get("tile"),
            "cloud": float(row_dict.get("cloud", np.nan)) if "cloud" in row_dict else None,
            "frp": float(row_dict.get("frp", 0)),
            "confidence": row_dict.get("confidence"),
            "event_n": row_dict.get("event_n"),
            "anchor": row_dict.get("anchor"),
            "days_after": row_dict.get("days_after"),
            "fire_dt": str(row_dict.get("fire_dt", "")) if "fire_dt" in row_dict else "",
            "latitude": float(latitude),
            "longitude": float(longitude),
            **analysis
        }

        save_patch(data["bands"], data["scl"], data["mask"], metadata, save_dir, name)
        return "ok", "ok"

    except Exception as e:
        return "error", str(e)[:100]


def save_patch(bands, scl, mask, metadata, save_dir, name):
    np.save(save_dir / f"{name}.npy", bands)
    np.save(save_dir / f"{name}_scl.npy", scl)
    np.save(save_dir / f"{name}_mask.npy", mask)
    
    with open(save_dir / f"{name}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    rgb = bands[:, :, [2, 1, 0]]
    rgb_uint8 = np.clip(rgb * 255 * 3.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / f"{name}.jpg"), rgb_uint8)


def download_batch(df, label, config):
    save_dir = DATA_DATASET / label
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if "lat" in df.columns:
        df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
    
    if N_LIMIT:
        df = df.head(N_LIMIT)
    
    # Build cache set once (much faster than checking .exists() for each)
    existing = {p.stem for p in save_dir.glob("*.json")} if save_dir.exists() else set()
    print(f"\nDownloading {label}: {len(df)} total, {len(existing)} cached")
    
    stats = {"ok": 0, "cloud": 0, "water": 0, "valid": 0, "error": 0}
    lock = Lock()
    
    rows = [row.to_dict() for _, row in df.iterrows()]
    
    keys = load_api_keys()
    configs = [make_config_for_key(k) for k in keys] if keys else [config]
    n_workers = len(configs)
    
    def submit_one(i, row_dict):
        cfg = configs[i % n_workers]
        res, _ = _download_one(row_dict, label, save_dir, cfg, existing)
        with lock:
            stats[res] = stats.get(res, 0) + 1
        return res
    
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(submit_one, i, row_dict) for i, row_dict in enumerate(rows)]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=label):
            pass
    
    print(f"  OK: {stats['ok']}, Cloud: {stats['cloud']}, Water: {stats['water']}, Valid: {stats['valid']}, Error: {stats['error']}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 50)
    print("Download Dataset")
    print("=" * 50)
    
    if not has_credentials():
        print("No credentials")
        return
    
    config = get_sh_config()
    DATA_DATASET.mkdir(parents=True, exist_ok=True)
    
    # Download all classes (filtering happens in prepare_splits.py)
    for name in ["fire", "no_fire", "burn_scar"]:
        csv = DATA_PROCESSED / f"{name}_download.csv"
        if csv.exists():
            download_batch(pd.read_csv(csv), name, config)
        else:
            print(f"Not found: {csv}")
    
    print(f"\nOutput: {DATA_DATASET}")


if __name__ == "__main__":
    main()
