
import datetime as dt
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from tqdm import tqdm

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# CONFIG - Edit these values to adjust the pipeline
# ═══════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data" / "processed"

# Input FIRMS archives - auto-detect all fire_archive*.csv files in data/raw/
INPUTS = sorted(DATA_RAW.glob("fire_archive*.csv"))

# FIRMS filtering
KEEP_TYPES = {0}           # 0 = vegetation fire (exclude volcanoes, industrial)
KEEP_CONF = {"n", "h"}     # n = nominal, h = high confidence
FRP_MIN = 5.0              # Minimum fire radiative power in MW
DAY_ONLY = True            # Only daytime detections (Sentinel-2 is optical)

# Static hotspot removal (industrial sites, gas flares)
GRID_DEG = 0.005           # Grid cell size (~500m)
STATIC_MAX_DAYS = 10       # Remove locations active >N different days

# Event clustering (group nearby detections)
EVENT_RADIUS_KM = 1.0      # Max distance between detections in same event
EVENT_MIN_PTS = 2          # Minimum detections to form an event

# Sentinel-2 matching
DO_S2 = True               # Enable/disable S2 matching
MAX_DELTA_H = 12.0         # Max hours between FIRMS and S2 image
MAX_CLOUD = 30.0           # Max cloud cover percentage
SEARCH_DAYS = 7            # Search window ±days
N_LIMIT = None             # Limit number of events (for testing)
SLEEP = 0.21               # Seconds between API calls (rate limit: 300/min)

# NO_FIRE generation
DO_NO_FIRE = True          # Generate negative samples
OFFSET_KM = (10.0, 30.0)   # Distance range from fire location
EXCLUDE_KM = 10.0          # No FIRMS detection within this radius
EXCLUDE_DAYS = 3           # No FIRMS detection within ±days
SEED = 42                  # Random seed for reproducibility

# Constants
R_EARTH = 6371.0           # Earth radius in km
PATCH_M = 224 * 10         # Patch size: 224 pixels × 10m resolution

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def parse_hhmm(series):
    """
    Parse FIRMS time column (HHMM format) safely.
    Handles invalid values by clamping to valid ranges.
    """
    s = series.astype(str).str.zfill(4)
    hh = pd.to_numeric(s.str[:2], errors="coerce").fillna(0).clip(0, 23).astype(int)
    mm = pd.to_numeric(s.str[2:], errors="coerce").fillna(0).clip(0, 59).astype(int)
    return hh, mm


def bbox(lat, lon):
    """Create bounding box for patch centered at lat/lon."""
    from sentinelhub import CRS, BBox
    d = PATCH_M / 2 / 111320  # degrees latitude
    cos_lat = max(math.cos(math.radians(lat)), 0.01)  # avoid division by zero near poles
    dx = d / cos_lat  # degrees longitude (adjusted for latitude)
    return BBox([lon - dx, lat - d, lon + dx, lat + d], crs=CRS.WGS84)


def offset_point(lat, lon, km, bearing):
    """
    Calculate new point at distance km and bearing (radians) from lat/lon.
    Uses spherical Earth approximation.
    """
    lat1 = math.radians(lat)
    d = km / R_EARTH
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(bearing))
    lon2 = math.radians(lon) + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2)
    )
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


# ═══════════════════════════════════════════════════════════════
# FIRMS CLEANUP
# ═══════════════════════════════════════════════════════════════

def firms_cleanup():
    """
    Load and clean FIRMS data, cluster into fire events.
    
    Returns:
        firms_all: All raw FIRMS data (for NO_FIRE exclusion)
        candidates: All detections belonging to events
        reps: One representative point per event (earliest detection)
    """
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    
    # Find and load all input files
    input_files = list(INPUTS) if not isinstance(INPUTS, list) else INPUTS
    existing = [p for p in input_files if p.exists()]
    
    if not existing:
        raise FileNotFoundError(f"No fire_archive*.csv files found in {DATA_RAW}")
    
    print(f"Loading {len(existing)} archive files:")
    for p in existing:
        print(f"  - {p.name}")
    
    dfs = [pd.read_csv(p) for p in existing]
    df = pd.concat(dfs, ignore_index=True)
    
    # Parse timestamps
    df["confidence"] = df["confidence"].astype(str).str.lower()
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    hh, mm = parse_hhmm(df["acq_time"])
    df["firms_dt"] = (df["acq_date"] + pd.to_timedelta(hh, "h") + pd.to_timedelta(mm, "m")).dt.tz_localize("UTC")
    
    # Remove invalid rows and duplicates
    df = df.dropna(subset=["latitude", "longitude", "acq_date", "firms_dt", "frp"])
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite", "frp"])
    firms_all = df.copy()
    
    # Apply quality filters
    df = df[df["type"].isin(KEEP_TYPES)]
    df = df[df["confidence"].isin(KEEP_CONF)]
    df = df[df["frp"] >= FRP_MIN]
    if DAY_ONLY and "daynight" in df.columns:
        df = df[df["daynight"].str.upper() == "D"]
    
    # Remove static hotspots (locations active on many different days)
    df = df.copy()
    df["grid"] = (np.floor(df["latitude"] / GRID_DEG).astype(int).astype(str) + "_" +
                  np.floor(df["longitude"] / GRID_DEG).astype(int).astype(str))
    df["day"] = df["acq_date"].dt.date
    grid_days = df.groupby("grid")["day"].nunique()
    static_grids = set(grid_days[grid_days > STATIC_MAX_DAYS].index)
    df = df[~df["grid"].isin(static_grids)].reset_index(drop=True)
    
    # Cluster detections into events using DBSCAN (per day)
    df["event_id"] = None
    days = df.groupby("day").indices
    
    for day, idx in tqdm(days.items(), desc="Clustering", total=len(days)):
        idx = list(idx)
        if len(idx) < EVENT_MIN_PTS:
            continue
        coords = np.radians(df.loc[idx, ["latitude", "longitude"]].values)
        labels = DBSCAN(
            eps=EVENT_RADIUS_KM / R_EARTH, 
            min_samples=EVENT_MIN_PTS, 
            metric="haversine"
        ).fit_predict(coords)
        for i, label in zip(idx, labels):
            if label >= 0:
                df.loc[i, "event_id"] = f"{day}_{label}"
    
    # Keep only clustered points (drop noise/singletons)
    df = df[df["event_id"].notna()]
    
    # Build output tables
    event_sizes = df.groupby("event_id").size().rename("event_n")
    
    # All candidates (every detection in an event)
    candidates = df.merge(event_sizes, left_on="event_id", right_index=True)
    candidates = candidates[["event_id", "latitude", "longitude", "firms_dt", "frp", "confidence", "type", "event_n"]]
    candidates.to_csv(DATA_OUT / "fire_candidates.csv", index=False)
    
    # Representatives (earliest detection per event - better chance of S2 being after)
    reps = df.sort_values(["event_id", "firms_dt"]).groupby("event_id").first().reset_index()
    reps = reps.rename(columns={"latitude": "lat", "longitude": "lon", "firms_dt": "dt"})
    reps = reps.merge(event_sizes, left_on="event_id", right_index=True)
    reps = reps[["event_id", "lat", "lon", "dt", "frp", "event_n"]]
    reps.to_csv(DATA_OUT / "fire_reps.csv", index=False)
    
    return firms_all, candidates, reps


# ═══════════════════════════════════════════════════════════════
# SENTINEL-2 MATCHING
# ═══════════════════════════════════════════════════════════════

def find_s2_scene(catalog, lat, lon, target_dt):
    """
    Find nearest Sentinel-2 scene within ±SEARCH_DAYS of target time.
    Returns scene metadata or None if no suitable scene found.
    """
    from sentinelhub import DataCollection
    
    t0 = (target_dt - dt.timedelta(days=SEARCH_DAYS)).isoformat().replace("+00:00", "Z")
    t1 = (target_dt + dt.timedelta(days=SEARCH_DAYS)).isoformat().replace("+00:00", "Z")
    
    items = list(catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox(lat, lon),
        time=(t0, t1),
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover", "properties.s2:mgrs_tile"]},
        limit=50
    ))
    
    if not items:
        return None
    
    def scene_dt(item):
        return dt.datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00"))
    
    def delta_hours(item):
        return (scene_dt(item) - target_dt).total_seconds() / 3600
    
    def cloud(item):
        return item["properties"].get("eo:cloud_cover", 999)
    
    # Find nearest scene (lowest |delta|, then lowest cloud)
    best = min(items, key=lambda it: (abs(delta_hours(it)), cloud(it)))
    
    if abs(delta_hours(best)) > MAX_DELTA_H:
        return None
    
    delta = delta_hours(best)
    return {
        "s2_dt": scene_dt(best),
        "delta_h": abs(delta),
        "delta_signed": delta,
        "cloud": cloud(best),
        "item_id": best.get("id"),
        "tile": best["properties"].get("s2:mgrs_tile")
    }


def s2_match(points, cache_name, desc):
    """
    Match points to Sentinel-2 scenes.
    Results are cached to allow resuming interrupted runs.
    """
    from sentinelhub import SHConfig, SentinelHubCatalog
    
    cfg = SHConfig(use_defaults=True)
    cfg.sh_client_id = os.environ.get("SH_CLIENT_ID")
    cfg.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
    catalog = SentinelHubCatalog(cfg)
    
    points = points.copy()
    points["dt"] = pd.to_datetime(points["dt"], utc=True)
    if N_LIMIT:
        points = points.head(N_LIMIT)
    
    # Load cache (resume from previous run)
    cache = DATA_OUT / cache_name
    done = set()
    prev = None
    if cache.exists():
        prev = pd.read_csv(cache)
        done = set(prev["event_id"].astype(str))
    
    # Process each point
    rows = []
    for _, row in tqdm(points.iterrows(), total=len(points), desc=desc):
        event_id = str(row["event_id"])
        if event_id in done:
            continue
        
        try:
            result = find_s2_scene(catalog, row["lat"], row["lon"], row["dt"].to_pydatetime())
        except Exception:
            result = None
        
        if result:
            rows.append({
                "event_id": event_id,
                "lat": row["lat"],
                "lon": row["lon"],
                "s2_dt": result["s2_dt"].isoformat(),
                "delta_h": result["delta_h"],
                "delta_signed": result["delta_signed"],
                "cloud": result["cloud"],
                "item_id": result["item_id"],
                "tile": result["tile"],
                "ok": result["cloud"] <= MAX_CLOUD
            })
        else:
            rows.append({"event_id": event_id, "lat": row["lat"], "lon": row["lon"], "ok": False})
        
        time.sleep(SLEEP)
    
    # Merge with cached results
    new_df = pd.DataFrame(rows)
    if prev is not None and len(new_df):
        matches = pd.concat([prev, new_df], ignore_index=True)
    elif prev is not None:
        matches = prev
    else:
        matches = new_df
    
    matches = matches.drop_duplicates(subset=["event_id"], keep="last")
    matches.to_csv(cache, index=False)
    
    n_ok = (matches["ok"] == True).sum() if len(matches) else 0
    print(f"  Matched: {n_ok}/{len(matches)}")
    return matches


def build_fire_download(candidates, matches):
    """
    Build final fire download list.
    Picks the FIRMS detection closest in time to S2 for each event.
    Later filtering with B12 > 0.5 removes images without visible fire.
    """
    ok_matches = matches[matches["ok"] == True].copy()
    if len(ok_matches) == 0:
        print("  No S2 matches found")
        return pd.DataFrame()
    
    ok_matches["s2_dt"] = pd.to_datetime(ok_matches["s2_dt"], utc=True)
    
    cand = candidates.copy()
    cand["firms_dt"] = pd.to_datetime(cand["firms_dt"], utc=True)
    
    # Join candidates with their event's S2 match
    joined = cand.merge(ok_matches[["event_id", "s2_dt", "cloud", "item_id", "tile"]], on="event_id")
    joined["delta_signed"] = (joined["s2_dt"] - joined["firms_dt"]).dt.total_seconds() / 3600
    joined["delta"] = joined["delta_signed"].abs()
    
    # Pick closest FIRMS detection to S2 for each event
    best = (joined
            .sort_values(["event_id", "delta"])
            .groupby("event_id").first().reset_index())
    
    # Filter by time and cloud
    best["ok"] = (best["delta"] <= MAX_DELTA_H) & (best["cloud"] <= MAX_CLOUD)
    
    ok = best[best["ok"]]
    ok.to_csv(DATA_OUT / "fire_download.csv", index=False)
    
    return ok


# ═══════════════════════════════════════════════════════════════
# NO_FIRE GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_no_fire(firms_all, anchors, target_n):
    """
    Generate NO_FIRE points by offsetting from fire locations.
    Ensures no FIRMS detection exists nearby in space and time.
    Uses ALL FIRMS data (not just filtered) for robust exclusion.
    """
    coords_df = firms_all[["latitude", "longitude", "firms_dt"]].dropna()
    tree = BallTree(np.radians(coords_df[["latitude", "longitude"]].values), metric="haversine")
    r_rad = EXCLUDE_KM / R_EARTH
    time_window = pd.Timedelta(days=EXCLUDE_DAYS)
    
    rng = np.random.default_rng(SEED)
    results = []
    
    for _, anchor in tqdm(anchors.iterrows(), total=len(anchors), desc="NO_FIRE"):
        # Try random offsets until we find a valid location
        for attempt in range(100):
            dist = rng.uniform(*OFFSET_KM)
            bearing = rng.uniform(0, 2 * math.pi)
            lat2, lon2 = offset_point(anchor["lat"], anchor["lon"], dist, bearing)
            
            # Check no FIRMS detections nearby in space AND time
            idx = tree.query_radius(np.radians([[lat2, lon2]]), r=r_rad)[0]
            if len(idx):
                times = pd.to_datetime(coords_df.iloc[idx]["firms_dt"])
                if ((times - anchor["dt"]).abs() <= time_window).any():
                    continue
            
            results.append({
                "id": f"NF_{anchor['id']}_{attempt}",
                "lat": lat2,
                "lon": lon2,
                "dt": anchor["dt"],
                "anchor": anchor["id"]
            })
            break
    
    return pd.DataFrame(results[:target_n])


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Build FIRE + NO_FIRE Candidates")
    print("=" * 60)
    
    has_creds = bool(os.environ.get("SH_CLIENT_ID") and os.environ.get("SH_CLIENT_SECRET"))
    
    # Step 1: FIRMS cleanup and clustering
    firms_all, candidates, reps = firms_cleanup()
    print(f"FIRMS: {len(firms_all):,} raw -> {len(candidates):,} candidates -> {len(reps):,} events")
    
    fire_ok = None
    
    # Step 2: Match fire events to Sentinel-2
    if DO_S2 and has_creds:
        matches = s2_match(reps, cache_name="s2_fire_cache.csv", desc="S2 FIRE")
        fire_ok = build_fire_download(candidates, matches)
        print(f"FIRE download: {len(fire_ok)} ready")
    elif DO_S2:
        print("S2 matching skipped (missing SH_CLIENT_ID/SH_CLIENT_SECRET)")
    
    # Step 3: Generate NO_FIRE points
    if DO_NO_FIRE:
        # Use fire download points as anchors (or reps if no S2 match)
        if fire_ok is not None and len(fire_ok):
            anchors = fire_ok.rename(columns={"event_id": "id", "latitude": "lat", "longitude": "lon", "s2_dt": "dt"})
        else:
            anchors = reps.rename(columns={"event_id": "id"})
        anchors = anchors[["id", "lat", "lon", "dt"]]
        
        no_fire = generate_no_fire(firms_all, anchors, len(anchors))
        
        # Match NO_FIRE to S2
        if DO_S2 and has_creds and len(no_fire):
            nf_input = no_fire.rename(columns={"id": "event_id"})
            nf_matches = s2_match(nf_input, cache_name="s2_nofire_cache.csv", desc="S2 NO_FIRE")
            nf_ok = nf_matches[nf_matches["ok"] == True]
            nf_ok.to_csv(DATA_OUT / "no_fire_download.csv", index=False)
            print(f"NO_FIRE download: {len(nf_ok)} ready")
        else:
            no_fire.to_csv(DATA_OUT / "no_fire_download.csv", index=False)
            print(f"NO_FIRE: {len(no_fire)} generated")
    
    print("=" * 60)
    print(f"Input: {DATA_RAW}")
    print(f"Output: {DATA_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
