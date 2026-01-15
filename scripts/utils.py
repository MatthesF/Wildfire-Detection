import datetime as dt
import math
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_DATASET = ROOT / "data" / "dataset"

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

R_EARTH = 6371.0  # km
METERS_PER_DEGREE = 111320  # meters per degree latitude (at equator)
IMG_SIZE = 224
RESOLUTION = 10
PATCH_M = IMG_SIZE * RESOLUTION
SLEEP = 0.21

# ═══════════════════════════════════════════════════════════════
# SENTINEL HUB CONFIG
# ═══════════════════════════════════════════════════════════════

_keys = []


def load_api_keys():
    """Load all available API keys from environment."""
    global _keys
    if _keys:
        return _keys
    
    keys = []
    # Try primary key
    client_id = os.environ.get("SH_CLIENT_ID")
    client_secret = os.environ.get("SH_CLIENT_SECRET")
    if client_id and client_secret:
        keys.append({"id": client_id, "secret": client_secret})
    
    # Try numbered keys (SH_CLIENT_ID_2, SH_CLIENT_ID_3, etc.)
    i = 2
    while True:
        client_id = os.environ.get(f"SH_CLIENT_ID_{i}")
        client_secret = os.environ.get(f"SH_CLIENT_SECRET_{i}")
        if not client_id or not client_secret:
            break
        keys.append({"id": client_id, "secret": client_secret})
        i += 1
    
    _keys = keys
    return keys


def get_sh_config():
    """Get SentinelHub config with first API key."""
    from sentinelhub import SHConfig
    
    keys = load_api_keys()
    if not keys:
        return None
    
    cfg = SHConfig(use_defaults=True)
    cfg.sh_client_id = keys[0]["id"]
    cfg.sh_client_secret = keys[0]["secret"]
    return cfg


def has_credentials():
    return len(load_api_keys()) > 0


def make_config_for_key(key):
    """Create SHConfig for a specific API key dict."""
    from sentinelhub import SHConfig
    cfg = SHConfig(use_defaults=True)
    cfg.sh_client_id = key["id"]
    cfg.sh_client_secret = key["secret"]
    return cfg


def make_bbox(lat, lon, size_m=PATCH_M):
    from sentinelhub import CRS, BBox
    half = size_m / 2
    d_lat = half / METERS_PER_DEGREE
    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    d_lon = half / (METERS_PER_DEGREE * cos_lat)
    return BBox([lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat], crs=CRS.WGS84)


# ═══════════════════════════════════════════════════════════════
# GEOMETRY
# ═══════════════════════════════════════════════════════════════

def offset_point(lat, lon, km, bearing):
    lat1 = math.radians(lat)
    d = km / R_EARTH
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(bearing))
    lon2 = math.radians(lon) + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2)
    )
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


# ═══════════════════════════════════════════════════════════════
# S2 CATALOG SEARCH
# ═══════════════════════════════════════════════════════════════

def search_s2_catalog(catalog, lat, lon, time_range, max_cloud=30.0, prefer_nearest_to=None):
    """Search S2 catalog. Returns dict or None."""
    from sentinelhub import DataCollection
    
    t0 = time_range[0].isoformat().replace("+00:00", "Z")
    t1 = time_range[1].isoformat().replace("+00:00", "Z")
    
    items = list(catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=make_bbox(lat, lon),
        time=(t0, t1),
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover", "properties.s2:mgrs_tile"]},
        limit=50
    ))
    
    if not items:
        return None
    
    def scene_dt(it):
        return dt.datetime.fromisoformat(it["properties"]["datetime"].replace("Z", "+00:00"))
    
    def cloud(it):
        return it["properties"].get("eo:cloud_cover", 999)
    
    # Sort by preference
    if prefer_nearest_to:
        def delta_h(it):
            return abs((scene_dt(it) - prefer_nearest_to).total_seconds() / 3600)
        best = min(items, key=lambda it: (delta_h(it), cloud(it)))
    else:
        best = min(items, key=lambda it: cloud(it))
    
    if max_cloud is not None and cloud(best) > max_cloud:
        return None
    
    delta_h = abs((scene_dt(best) - prefer_nearest_to).total_seconds() / 3600) if prefer_nearest_to else None
    
    return {
        "s2_dt": scene_dt(best),
        "cloud": cloud(best),
        "item_id": best.get("id"),
        "tile": best["properties"].get("s2:mgrs_tile"),
        "delta_h": delta_h
    }


def _worker_search(worker_id, key, rows_to_process, search_fn, pbar, lock):
    """Worker function that processes a subset of rows with a specific API key."""
    from sentinelhub import SentinelHubCatalog, SHConfig
    
    # Create config for this specific API key
    cfg = SHConfig(use_defaults=True)
    cfg.sh_client_id = key["id"]
    cfg.sh_client_secret = key["secret"]
    catalog = SentinelHubCatalog(cfg)
    
    results = []
    for row in rows_to_process:
        eid = str(row["event_id"])
        error_msg = None
        
        try:
            r = search_fn(catalog, row)
        except Exception as e:
            r = None
            error_msg = str(e)[:100]  # Truncate long errors
        
        if r:
            results.append({"event_id": eid, "lat": row["lat"], "lon": row["lon"], **r, "ok": True})
        else:
            results.append({"event_id": eid, "lat": row["lat"], "lon": row["lon"], "ok": False, "error": error_msg})
        
        # Update progress bar
        with lock:
            pbar.update(1)
        
        # Sleep only for this worker's key (rate limit per key)
        time.sleep(SLEEP)
    
    return results


def s2_match_batch_parallel(points, cache_path, desc, search_fn, budget=None):
    """Parallel S2 matching with multiple API keys."""
    points = points.copy()
    points["event_id"] = points["event_id"].astype(str)  # Enforce string type
    if "dt" in points.columns:
        points["dt"] = pd.to_datetime(points["dt"], utc=True)
    
    # Load cache (enforce string type for merge consistency)
    prev = pd.read_csv(cache_path) if cache_path.exists() else None
    if prev is not None:
        prev["event_id"] = prev["event_id"].astype(str)
    done = set(prev["event_id"]) if prev is not None else set()
    
    # What to process
    todo = [eid for eid in points["event_id"] if eid not in done]
    if budget and len(todo) > budget:
        todo = todo[:budget]
    
    keys = load_api_keys()
    n_workers = len(keys)
    print(f"  {len(done)} cached, {len(todo)} to fetch | Parallel workers: {n_workers}")
    
    if len(todo) == 0:
        matches = prev if prev is not None else pd.DataFrame()
        n_ok = (matches["ok"] == True).sum() if len(matches) else 0
        print(f"  Matched: {n_ok}/{len(matches)}")
        return matches, 0
    
    # Filter to only process what we need
    todo_set = set(todo)
    to_process = points[points["event_id"].isin(todo_set)]
    
    # Split work among workers (round-robin)
    rows_list = [row for _, row in to_process.iterrows()]
    worker_assignments = [[] for _ in range(n_workers)]
    for i, row in enumerate(rows_list):
        worker_assignments[i % n_workers].append(row)
    
    # Run workers in parallel - collect results incrementally
    all_results = []
    ok_count = 0
    lock = Lock()
    failed_workers = 0
    
    with tqdm(total=len(rows_list), desc=desc) as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for worker_id, (key, rows) in enumerate(zip(keys, worker_assignments)):
                if len(rows) > 0:
                    future = executor.submit(_worker_search, worker_id, key, rows, search_fn, pbar, lock)
                    futures[future] = worker_id
            
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                    processed = len(all_results)
                    ok_count += sum(1 for r in results if r.get("ok"))
                    if processed:
                        pbar.set_postfix_str(f"ok={ok_count}/{processed} ({ok_count/processed:.0%})")
                except Exception as e:
                    failed_workers += 1
                    print(f"\nWorker {futures[future]} failed: {e}")
    
    if failed_workers:
        print(f"  {failed_workers} worker(s) failed, partial results collected")
    
    # Merge with cache
    new_df = pd.DataFrame(all_results)
    if prev is not None and len(new_df):
        matches = pd.concat([prev, new_df], ignore_index=True)
    elif prev is not None:
        matches = prev
    else:
        matches = new_df
    
    matches = matches.drop_duplicates(subset=["event_id"], keep="last")
    matches.to_csv(cache_path, index=False)
    
    n_ok = (matches["ok"] == True).sum() if len(matches) else 0
    print(f"  Matched: {n_ok}/{len(matches)}")
    return matches, len(all_results)


def s2_match_batch(points, cache_path, desc, search_fn, budget=None):
    """Match points to S2 scenes with caching. Uses parallel if multiple keys."""
    keys = load_api_keys()
    
    # Use parallel processing if multiple keys available
    if len(keys) > 1:
        return s2_match_batch_parallel(points, cache_path, desc, search_fn, budget)
    
    # Single key: use sequential processing
    from sentinelhub import SentinelHubCatalog
    
    points = points.copy()
    points["event_id"] = points["event_id"].astype(str)  # Enforce string type
    if "dt" in points.columns:
        points["dt"] = pd.to_datetime(points["dt"], utc=True)
    
    # Load cache (enforce string type for merge consistency)
    prev = pd.read_csv(cache_path) if cache_path.exists() else None
    if prev is not None:
        prev["event_id"] = prev["event_id"].astype(str)
    done = set(prev["event_id"]) if prev is not None else set()
    
    # What to process
    todo = [eid for eid in points["event_id"] if eid not in done]
    if budget and len(todo) > budget:
        todo = todo[:budget]
    
    print(f"  {len(done)} cached, {len(todo)} to fetch")
    
    # Filter to only process what we need
    todo_set = set(todo)
    to_process = points[points["event_id"].isin(todo_set)]
    
    rows = []
    catalog = SentinelHubCatalog(get_sh_config())
    
    ok_count = 0
    processed = 0
    pbar = tqdm(to_process.iterrows(), total=len(to_process), desc=desc, disable=(len(to_process)==0))
    for _, row in pbar:
        eid = str(row["event_id"])
        error_msg = None
        
        try:
            r = search_fn(catalog, row)
        except Exception as e:
            r = None
            error_msg = str(e)[:100]
        
        if r:
            rows.append({"event_id": eid, "lat": row["lat"], "lon": row["lon"], **r, "ok": True})
        else:
            rows.append({"event_id": eid, "lat": row["lat"], "lon": row["lon"], "ok": False, "error": error_msg})
        
        time.sleep(SLEEP)
        
        processed += 1
        ok_count += 1 if r else 0
        if processed:
            pbar.set_postfix_str(f"ok={ok_count}/{processed} ({ok_count/processed:.0%})")
    
    # Merge with cache
    new_df = pd.DataFrame(rows)
    if prev is not None and len(new_df):
        matches = pd.concat([prev, new_df], ignore_index=True)
    elif prev is not None:
        matches = prev
    else:
        matches = new_df
    
    matches = matches.drop_duplicates(subset=["event_id"], keep="last")
    matches.to_csv(cache_path, index=False)
    
    n_ok = (matches["ok"] == True).sum() if len(matches) else 0
    print(f"  Matched: {n_ok}/{len(matches)}")
    return matches, len(rows)


# ═══════════════════════════════════════════════════════════════
# SCL-BASED ANALYSIS
# ═══════════════════════════════════════════════════════════════

SCL_CLOUD = {3, 8, 9, 10}  # Shadow, cloud med/high/cirrus
SCL_WATER = {6}
SCL_SNOW = {11}


def analyze_patch_scl(bands, scl, mask):
    """Analyze patch quality using SCL. Returns dict with pct and B12/NBR stats."""
    valid = mask == 1
    valid_count = valid.sum()
    
    if valid_count == 0:
        return {"valid_pct": 0, "cloud_pct": 1, "water_pct": 0, "b12_max": 0, "nbr_mean": 0, "burn_pct": 0}
    
    cloud = np.isin(scl, list(SCL_CLOUD))
    water = np.isin(scl, list(SCL_WATER))
    snow = np.isin(scl, list(SCL_SNOW))
    clear = valid & ~cloud & ~water & ~snow
    clear_count = clear.sum()
    
    result = {
        "valid_pct": round(valid_count / mask.size, 4),
        "cloud_pct": round((cloud & valid).sum() / valid_count, 4),
        "water_pct": round((water & valid).sum() / valid_count, 4),
        "snow_pct": round((snow & valid).sum() / valid_count, 4),
        "clear_pct": round(clear_count / valid_count, 4) if valid_count else 0
    }
    
    if clear_count > 0:
        b08 = bands[:, :, 3][clear]  # NIR
        b12 = bands[:, :, 5][clear]  # SWIR2
        
        result["b12_max"] = round(float(b12.max()), 4)
        result["b12_mean"] = round(float(b12.mean()), 4)
        
        # NBR for burn scar detection
        eps = 1e-6
        nbr = (b08 - b12) / (b08 + b12 + eps)
        result["nbr_mean"] = round(float(nbr.mean()), 4)
        result["nbr_min"] = round(float(nbr.min()), 4)
        result["burn_pct"] = round(float((nbr < -0.1).sum() / clear_count), 4)
    else:
        result.update({"b12_max": 0, "b12_mean": 0, "nbr_mean": 0, "nbr_min": 0, "burn_pct": 0})
    
    return result
