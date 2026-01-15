from __future__ import annotations
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

def plot_fires(df: pd.DataFrame) -> None:
    if "lat" in df.columns:
        df = df.rename(columns={"lat": "latitude"})

    if "lon" in df.columns:
        df = df.rename(columns={"lon": "longitude"})

    # antag df med latitude, longitude
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lat, lon in zip(df["latitude"], df["longitude"])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(
        ax=ax,
        markersize=4,
        color="red",
        alpha=0.5
    )

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    ax.set_axis_off()
    ax.set_title("FIRMS brand-hotspots (overblik)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import pandas as pd

    fire_cand_path = "data/processed/fire_candidates.csv"
    fire_reps_path = "data/processed/fire_reps.csv"
    df = pd.read_csv(fire_reps_path)
    plot_fires(df)