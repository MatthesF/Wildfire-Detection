import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
DATA_DATASET = ROOT / "data" / "dataset"
DATA_SPLITS = ROOT / "data" / "splits"

MIN_FIRE_B12 = 1.0
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
BALANCE_TRAIN = True  # Only balance train, keep val/test natural
GEO_GRID_DEG = 0.5    # ~50km grid cells for grouping
SEED = 42

# ═══════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def geo_grid_key(lat, lon, grid_deg=GEO_GRID_DEG):
    """Create spatial group key from lat/lon grid."""
    if pd.isna(lat) or pd.isna(lon):
        return None
    lat_bin = int(np.floor(lat / grid_deg))
    lon_bin = int(np.floor(lon / grid_deg))
    return f"G_{lat_bin}_{lon_bin}"


def load_samples(label_dir):
    """Load all samples from a label directory."""
    if not label_dir.exists():
        return []
    
    samples = []
    for json_path in sorted(label_dir.glob("*.json")):
        with open(json_path) as f:
            meta = json.load(f)
        
        event_id = meta.get("event_id")
        if event_id is None:
            continue  # Skip invalid
        
        lat = meta.get("latitude")
        lon = meta.get("longitude")
        group = geo_grid_key(lat, lon)
        
        # Fallback: use event_id if no valid geo
        if group is None:
            group = f"E_{event_id}"
        
        samples.append({
            "name": json_path.stem,
            "event_id": str(event_id),
            "b12": meta.get("b12_max", 0) or 0,
            "group": group,
            "label": label_dir.name
        })
    return samples


def copy_patch(name, label, dst_dir):
    """Copy all files for a single patch."""
    src_dir = DATA_DATASET / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    for ext in [".npy", "_scl.npy", "_mask.npy", ".json", ".jpg"]:
        src = src_dir / f"{name}{ext}"
        if src.exists():
            shutil.copy2(src, dst_dir / f"{name}{ext}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"Prepare Splits | B12 >= {MIN_FIRE_B12} | Grid = {GEO_GRID_DEG}°")
    print("=" * 60)
    
    # 1. Load all data
    fire = pd.DataFrame(load_samples(DATA_DATASET / "fire"))
    nofire = pd.DataFrame(load_samples(DATA_DATASET / "no_fire"))
    burn = pd.DataFrame(load_samples(DATA_DATASET / "burn_scar"))
    
    print(f"\nRaw: fire={len(fire)}, no_fire={len(nofire)}, burn_scar={len(burn)}")
    
    # 2. Filter fire by B12
    fire_filtered = fire[fire["b12"] >= MIN_FIRE_B12].copy()
    fire_kept_ids = set(fire_filtered["event_id"])
    print(f"Fire after B12 >= {MIN_FIRE_B12}: {len(fire_filtered)}/{len(fire)}")
    
    # 3. Filter burn_scar: only if source fire is in filtered set
    burn["fire_id"] = burn["event_id"].str.replace("BS_", "", regex=False)
    burn_filtered = burn[burn["fire_id"].isin(fire_kept_ids)].drop(columns=["fire_id"])
    print(f"Burn_scar after fire filter: {len(burn_filtered)}/{len(burn)}")
    
    # 4. Combine (no balancing yet!)
    df = pd.concat([fire_filtered, nofire, burn_filtered], ignore_index=True)
    print(f"\nCombined: {len(df)} | By label: {df['label'].value_counts().to_dict()}")
    print(f"Unique geo-groups: {df['group'].nunique()}")
    
    # 5. Group split (train vs rest)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO+TEST_RATIO, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(df, groups=df["group"]))
    
    # Split rest into val/test
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=SEED)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group"]))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    
    # 6. Balance ONLY train (keep val/test natural)
    if BALANCE_TRAIN:
        before = train_df["label"].value_counts().to_dict()
        min_count = train_df["label"].value_counts().min()
        balanced_dfs = []
        for label in train_df["label"].unique():
            label_df = train_df[train_df["label"] == label]
            sampled = label_df.sample(n=min_count, random_state=SEED)
            balanced_dfs.append(sampled)
        train_df = pd.concat(balanced_dfs, ignore_index=True)
        after = train_df["label"].value_counts().to_dict()
        print(f"\nTrain balanced: {before} -> {after}")
    
    splits = {"train": train_df, "val": val_df, "test": test_df}
    
    # 7. Verify no group leakage
    groups = {k: set(v["group"]) for k, v in splits.items()}
    train_val = groups["train"] & groups["val"]
    train_test = groups["train"] & groups["test"]
    val_test = groups["val"] & groups["test"]
    
    if train_val or train_test or val_test:
        print(f"\nWARNING: Group overlap! train&val={len(train_val)}, train&test={len(train_test)}, val&test={len(val_test)}")
    else:
        print(f"\nNo group leakage (OK)")
    
    # 8. Copy files
    print("\nCopying files...")
    if DATA_SPLITS.exists():
        shutil.rmtree(DATA_SPLITS)
    
    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            copy_patch(row["name"], row["label"], DATA_SPLITS / split_name / row["label"])
    
    # 9. Summary
    print("\n" + "=" * 60)
    print("FINAL SPLITS")
    print("=" * 60)
    
    total = sum(len(s) for s in splits.values())
    for name in ["train", "val", "test"]:
        s = splits[name]
        n = len(s)
        by_label = s["label"].value_counts().to_dict()
        n_groups = s["group"].nunique()
        print(f"{name.upper():5} | {n:5} ({n/total*100:5.1f}%) | {n_groups:4} groups | {by_label}")
    
    print("=" * 60)
    print(f"Output: {DATA_SPLITS}")


if __name__ == "__main__":
    main()
