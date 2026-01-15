import datetime as dt
import math

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from tqdm import tqdm

from utils import (
    DATA_RAW, DATA_PROCESSED, R_EARTH,
    has_credentials, search_s2_catalog, s2_match_batch
)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

INPUTS = sorted(DATA_RAW.glob("fire_archive*.csv"))

# FIRMS filters
KEEP_TYPES = {0}
KEEP_CONF = {"n", "h"}
FRP_MIN = 5.0
DAY_ONLY = True

# Static hotspot removal
GRID_DEG = 0.005
STATIC_MAX_DAYS = 10

# DBSCAN
EVENT_RADIUS_KM = 1.0
EVENT_MIN_PTS = 2

# S2 matching
SKIP_FIRE_S2 = False
MAX_DELTA_H = 12.0   # Max time diff FIRMS→S2 (hours)
MAX_CLOUD = 30.0
SEARCH_DAYS = 1      # Search ±1 day (covers 12h filter, saves API calls)

# NO_FIRE
OFFSET_KM = (10.0, 30.0)
EXCLUDE_KM = 10.0
EXCLUDE_DAYS = 3
NO_FIRE_RATIO = 1.0  # 1.0 = same count as FIRE

# BURN_SCAR
DO_BURN_SCAR = True
BURN_SCAR_DAYS = (3, 30)
BURN_SCAR_RATIO = 1.0  # 1.0 = same count as FIRE

# Budget
MAX_REQUESTS = 120000  # Set e.g. 50000 to limit total API requests
SEED = 42

# ═══════════════════════════════════════════════════════════════
# FIRMS
# ═══════════════════════════════════════════════════════════════

def parse_hhmm(series):
    s = series.astype(str).str.zfill(4)
    hh = pd.to_numeric(s.str[:2], errors="coerce").fillna(0).clip(0, 23).astype(int)
    mm = pd.to_numeric(s.str[2:], errors="coerce").fillna(0).clip(0, 59).astype(int)
    return hh, mm


def load_firms():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    existing = [p for p in INPUTS if p.exists()]
    if not existing:
        raise FileNotFoundError(f"No fire_archive*.csv in {DATA_RAW}")
    
    print(f"Loading {len(existing)} archives")
    df = None
    for p in existing:
        chunk = pd.read_csv(p)
        df = chunk if df is None else pd.concat([df, chunk], ignore_index=True)
    
    # Parse time
    df["confidence"] = df["confidence"].astype(str).str.lower()
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    hh, mm = parse_hhmm(df["acq_time"])
    df["firms_dt"] = (df["acq_date"] + pd.to_timedelta(hh, "h") + pd.to_timedelta(mm, "m")).dt.tz_localize("UTC")
    
    df = df.dropna(subset=["latitude", "longitude", "acq_date", "firms_dt", "frp"])
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite", "frp"])
    firms_all = df.copy()
    
    # Filter
    df = df[df["type"].isin(KEEP_TYPES)]
    df = df[df["confidence"].isin(KEEP_CONF)]
    df = df[df["frp"] >= FRP_MIN]
    if DAY_ONLY and "daynight" in df.columns:
        df = df[df["daynight"].str.upper() == "D"]
    
    # Static hotspots
    df = df.copy()
    df["grid"] = (np.floor(df["latitude"] / GRID_DEG).astype(int).astype(str) + "_" +
                  np.floor(df["longitude"] / GRID_DEG).astype(int).astype(str))
    df["day"] = df["acq_date"].dt.date
    grid_days = df.groupby("grid")["day"].nunique()
    static = set(grid_days[grid_days > STATIC_MAX_DAYS].index)
    df = df[~df["grid"].isin(static)].reset_index(drop=True)
    
    # Cluster
    df["event_id"] = None
    for day, idx in tqdm(df.groupby("day").indices.items(), desc="Clustering"):
        idx = list(idx)
        if len(idx) < EVENT_MIN_PTS:
            continue
        coords = np.radians(df.loc[idx, ["latitude", "longitude"]].values)
        labels = DBSCAN(eps=EVENT_RADIUS_KM / R_EARTH, min_samples=EVENT_MIN_PTS, metric="haversine").fit_predict(coords)
        for i, label in zip(idx, labels):
            if label >= 0:
                df.loc[i, "event_id"] = f"{day}_{label}"
    
    df = df[df["event_id"].notna()]
    
    # Reps
    event_n = df.groupby("event_id").size().rename("event_n")
    reps = df.sort_values(["event_id", "firms_dt"]).groupby("event_id").first().reset_index()
    reps = reps.rename(columns={"latitude": "lat", "longitude": "lon", "firms_dt": "dt"})
    reps = reps.merge(event_n, left_on="event_id", right_index=True)
    reps = reps[["event_id", "lat", "lon", "dt", "frp", "event_n"]]
    reps = reps.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    candidates = df.merge(event_n, left_on="event_id", right_index=True)
    candidates = candidates[["event_id", "latitude", "longitude", "firms_dt", "frp", "confidence", "type", "event_n"]]
    candidates.to_csv(DATA_PROCESSED / "fire_candidates.csv", index=False)
    reps.to_csv(DATA_PROCESSED / "fire_reps.csv", index=False)
    
    return firms_all, candidates, reps


# ═══════════════════════════════════════════════════════════════
# S2 MATCHING
# ═══════════════════════════════════════════════════════════════

def search_fire(catalog, row):
    """Search S2 scene nearest to FIRMS detection."""
    target = row["dt"].to_pydatetime()
    t0 = target - dt.timedelta(days=SEARCH_DAYS)
    t1 = target + dt.timedelta(days=SEARCH_DAYS)
    r = search_s2_catalog(catalog, row["lat"], row["lon"], (t0, t1), 
                          max_cloud=None, prefer_nearest_to=target)
    if r and r["delta_h"] <= MAX_DELTA_H and r["cloud"] <= MAX_CLOUD:
        r["ok"] = True
        return r
    return None


def search_nofire(catalog, row):
    """Search ANY S2 scene with low cloud."""
    target = row["dt"].to_pydatetime()
    t0 = target - dt.timedelta(days=SEARCH_DAYS)
    t1 = target + dt.timedelta(days=SEARCH_DAYS)
    return search_s2_catalog(catalog, row["lat"], row["lon"], (t0, t1), 
                             max_cloud=MAX_CLOUD, prefer_nearest_to=None)


def search_burnscar(catalog, row):
    """Search S2 scene 3-10 days AFTER fire."""
    fire_dt = pd.to_datetime(row["firms_dt"], utc=True).to_pydatetime()
    t0 = fire_dt + dt.timedelta(days=BURN_SCAR_DAYS[0])
    t1 = fire_dt + dt.timedelta(days=BURN_SCAR_DAYS[1])
    r = search_s2_catalog(catalog, row["lat"], row["lon"], (t0, t1),
                          max_cloud=MAX_CLOUD, prefer_nearest_to=None)
    if r:
        r["fire_dt"] = fire_dt.isoformat()
        r["days_after"] = (r["s2_dt"] - fire_dt).days
    return r


def build_fire_download(candidates, matches):
    ok = matches[matches["ok"] == True].copy()
    if len(ok) == 0:
        return pd.DataFrame()
    
    ok["s2_dt"] = pd.to_datetime(ok["s2_dt"], utc=True)
    cand = candidates.copy()
    cand["firms_dt"] = pd.to_datetime(cand["firms_dt"], utc=True)
    
    joined = cand.merge(ok[["event_id", "s2_dt", "cloud", "item_id", "tile"]], on="event_id")
    joined["delta"] = (joined["s2_dt"] - joined["firms_dt"]).dt.total_seconds().abs() / 3600
    
    best = joined.sort_values(["event_id", "delta"]).groupby("event_id").first().reset_index()
    best["ok"] = (best["delta"] <= MAX_DELTA_H) & (best["cloud"] <= MAX_CLOUD)
    
    result = best[best["ok"]]
    result.to_csv(DATA_PROCESSED / "fire_download.csv", index=False)
    return result

# ═══════════════════════════════════════════════════════════════
# NO_FIRE
# ═══════════════════════════════════════════════════════════════

def generate_no_fire(firms_all, anchors, target_n):
    """Generate NO_FIRE points by offsetting from anchors, excluding areas with ANY FIRMS detection."""
    from utils import offset_point
    
    # Use ALL raw FIRMS detections for exclusion (not just filtered ones)
    # This ensures we don't accidentally create NO_FIRE on low-confidence fires
    coords_df = firms_all[["latitude", "longitude", "firms_dt"]].dropna()
    tree = BallTree(np.radians(coords_df[["latitude", "longitude"]].values), metric="haversine")
    r_rad = EXCLUDE_KM / R_EARTH
    time_window = pd.Timedelta(days=EXCLUDE_DAYS)
    
    rng = np.random.default_rng(SEED)
    results = []
    failed_anchors = 0
    
    for _, anchor in tqdm(anchors.iterrows(), total=len(anchors), desc="NO_FIRE"):
        # Stop early if we have enough
        if len(results) >= target_n:
            break
        
        found = False
        for attempt in range(100):
            dist = rng.uniform(*OFFSET_KM)
            bearing = rng.uniform(0, 2 * math.pi)
            lat2, lon2 = offset_point(anchor["lat"], anchor["lon"], dist, bearing)
            
            # Check if any FIRMS detection is nearby within time window
            idx = tree.query_radius(np.radians([[lat2, lon2]]), r=r_rad)[0]
            if len(idx):
                times = pd.to_datetime(coords_df.iloc[idx]["firms_dt"])
                if ((times - anchor["dt"]).abs() <= time_window).any():
                    continue  # Too close to a fire, try again
            
            results.append({
                "event_id": f"NF_{anchor['id']}",
                "lat": lat2, "lon": lon2, "dt": anchor["dt"], "anchor": anchor["id"]
            })
            found = True
            break
        
        if not found:
            failed_anchors += 1
    
    if len(results) < target_n:
        print(f"NO_FIRE: only generated {len(results)}/{target_n} (failed: {failed_anchors})")
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 50)
    print(f"Build Candidates | Budget: {MAX_REQUESTS or 'unlimited'}")
    print("=" * 50)
    
    firms_all, candidates, reps = load_firms()
    print(f"FIRMS: {len(firms_all):,} -> {len(candidates):,} -> {len(reps):,} events")
    
    requests_used = 0
    
    # FIRE
    if SKIP_FIRE_S2:
        fire_csv = DATA_PROCESSED / "fire_download.csv"
        if fire_csv.exists():
            fire_ok = pd.read_csv(fire_csv)
            fire_ok["s2_dt"] = pd.to_datetime(fire_ok["s2_dt"], utc=True)
            fire_ok = fire_ok.sample(frac=1, random_state=SEED).reset_index(drop=True)
            print(f"FIRE: {len(fire_ok)} (cached)")
        else:
            print("No fire_download.csv")
            return
    elif has_credentials():
        fire_budget = MAX_REQUESTS // 3 if MAX_REQUESTS else None
        matches, n = s2_match_batch(reps, DATA_PROCESSED / "s2_fire_cache.csv", "S2 FIRE", search_fire, fire_budget)
        requests_used += n
        fire_ok = build_fire_download(candidates, matches)
        fire_pass = len(fire_ok)
        fire_total = len(reps)
        fire_pct = (fire_pass / fire_total) if fire_total else 0
        print(f"FIRE: {fire_pass}/{fire_total} ({fire_pct:.1%})")
    else:
        print("No credentials")
        return
    
    if len(fire_ok) == 0:
        return
    
    # Calculate targets (separate classes, not combined)
    n_fire = len(fire_ok)
    n_nofire = int(n_fire * NO_FIRE_RATIO)
    n_burn = int(n_fire * BURN_SCAR_RATIO) if DO_BURN_SCAR else 0
    
    remaining = MAX_REQUESTS - requests_used if MAX_REQUESTS else None
    total_needed = n_nofire + n_burn
    
    print(f"\nTarget dataset:")
    print(f"  FIRE:       {n_fire}")
    print(f"  NO_FIRE:    {n_nofire}")
    print(f"  BURN_SCAR:  {n_burn}")
    print(f"  Total API needed: {total_needed} | Budget: {remaining if remaining else 'unlimited'}\n")
    
    # Budget allocation
    if remaining and total_needed > remaining:
        ratio_nf = n_nofire / total_needed
        budget_nofire = int(remaining * ratio_nf)
        budget_burn = remaining - budget_nofire
        print(f"Budget split: {budget_nofire} no-fire, {budget_burn} burn-scar")
    else:
        budget_nofire = n_nofire
        budget_burn = n_burn
    
    # NO_FIRE
    anchors = fire_ok.rename(columns={"event_id": "id", "latitude": "lat", "longitude": "lon", "s2_dt": "dt"})
    anchors = anchors[["id", "lat", "lon", "dt"]]
    
    target_nf = min(n_nofire, budget_nofire)
    no_fire = generate_no_fire(firms_all, anchors.head(target_nf), target_nf)
    
    if has_credentials():
        nf_matches, n = s2_match_batch(no_fire, DATA_PROCESSED / "s2_nofire_cache.csv", "S2 NO_FIRE", search_nofire, budget_nofire)
        requests_used += n
        nf_ok = nf_matches[nf_matches["ok"] == True]
        nf_ok.to_csv(DATA_PROCESSED / "no_fire_download.csv", index=False)
        print(f"NO_FIRE: {len(nf_ok)}/{target_nf} ({len(nf_ok)/target_nf:.0%})")
    
    # BURN_SCAR
    if DO_BURN_SCAR and has_credentials():
        target_bs = min(n_burn, budget_burn)
        if target_bs > 0:
            bs_input = fire_ok.head(target_bs).rename(columns={"latitude": "lat", "longitude": "lon"})
            bs_matches, n = s2_match_batch(bs_input, DATA_PROCESSED / "s2_burnscar_cache.csv", "BURN_SCAR", search_burnscar, budget_burn)
            requests_used += n
            bs_ok = bs_matches[bs_matches["ok"] == True]
            bs_ok["event_id"] = "BS_" + bs_ok["event_id"].astype(str)
            bs_ok.to_csv(DATA_PROCESSED / "burn_scar_download.csv", index=False)
            print(f"BURN_SCAR: {len(bs_ok)}/{target_bs} ({len(bs_ok)/target_bs:.0%})")
    
    print("=" * 50)
    print(f"Done. Requests: {requests_used}" + (f" / {MAX_REQUESTS}" if MAX_REQUESTS else ""))


if __name__ == "__main__":
    main()
