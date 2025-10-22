from argparse import ArgumentParser
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from pyproj import Proj
from tqdm import tqdm

locations_dict = {
    "la_paz": {
        "lat_min": -21.5,
        "lat_max": -11.5,
        "lon_min": -73.0,
        "lon_max": -63.0,
    },
    "manaus": {
        "lat_min": -8.0,
        "lat_max": 2.0,
        "lon_min": -65.0,
        "lon_max": -55.0,
    },
    "toronto": {
        "lat_min": 39.06,
        "lat_max": 48.34,
        "lon_min": -85.77,
        "lon_max": -73.03,
    },
    "miami": {
        "lat_min": 21.16,
        "lat_max": 30.44,
        "lon_min": -85.31,
        "lon_max": -75.08,
    },
}


def process_satellite(
    file_path: str,
    bands: list[str],
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    include_dataset_name: bool = False,
) -> pd.DataFrame:
    """Returns processed satellite data for desired region and bands.

    Conversion of coordinates to latitudes and longitudes based on [1].

    Args:
        file_path: path to netcdf file with satellite data
        bands: bands to be extracted from data
        lat_bounds: minimum and maximum latitudes to consider
        lon_bounds: minimum and maximum longitudes to consider

    References:
        [1] https://github.com/blaylockbk/pyBKB_v3/blob/master/BB_GOES/mapping_GOES16_TrueColor.ipynb
        [2] https://proj4.org/operations/projections/geos.html
    """
    # Read satellite data
    dataset = xr.open_dataset(file_path)

    # Retrieve datetimes of file creation and start and end of scan (UTC)
    scan_start = datetime.strptime(dataset.time_coverage_start, "%Y-%m-%dT%H:%M:%S.%fZ")
    scan_end = datetime.strptime(dataset.time_coverage_end, "%Y-%m-%dT%H:%M:%S.%fZ")
    creation = datetime.strptime(dataset.date_created, "%Y-%m-%dT%H:%M:%S.%fZ")

    if hasattr(dataset, "lon") and hasattr(dataset, "lat"):
        lons, lats = dataset["lon"], dataset["lat"]
    else:
        # Load satellite height, longitude and sweep
        sat_h = dataset["goes_imager_projection"].perspective_point_height
        sat_lon = dataset["goes_imager_projection"].longitude_of_projection_origin
        sat_sweep = dataset["goes_imager_projection"].sweep_angle_axis

        # Calculate projection x and y coordinates as the scanning angle (in radians)
        #   multiplied by the satellite height (cf. [2])
        x = dataset["x"][:] * sat_h
        y = dataset["y"][:] * sat_h

        # Create a pyproj geostationary map object
        p = Proj(proj="geos", h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

        # Perform cartographic transformation, that is, convert
        #  image projection coordinates (x and y) to latitude and longitude values
        XX, YY = np.meshgrid(x, y)
        lons, lats = p(XX, YY, inverse=True)

    # Load data from bands of interest and append to latitude and longitude
    data = [lats, lons] + [dataset[band].data for band in bands]

    # Construct dataframe
    df = pd.DataFrame(
        np.array(data).reshape(len(data), -1).T,
        columns=["lat", "lon", *bands],
    )

    # Remove nan and infinite values
    df = df.replace(np.inf, np.nan)
    df = df.dropna(how="any", axis=0)

    # Discard observations for latitudes and longitudes outside bounds
    if lat_bounds:
        df = df[(df["lat"] > lat_bounds[0]) & (df["lat"] < lat_bounds[1])]
    if lon_bounds:
        df = df[(df["lon"] > lon_bounds[0]) & (df["lon"] < lon_bounds[1])]

    # Include datetimes of file creation and start and end of scan (UTC)
    df["start"] = scan_start  # - timedelta(hours=3)
    df["end"] = scan_end  # - timedelta(hours=3)
    df["creation"] = creation  # - timedelta(hours=3)

    if include_dataset_name:
        df["name"] = "_".join(dataset.dataset_name.split("_")[:2])

    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--product", type=str, default="ABI-L2-RRQPEF")
    parser.add_argument("--lat_min", type=float, default=-26.0)
    parser.add_argument("--lat_max", type=float, default=-19.0)
    parser.add_argument("--lon_min", type=float, default=-47.0)
    parser.add_argument("--lon_max", type=float, default=-40.0)
    parser.add_argument("--location", type=str, default="manaus")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    if args.location in locations_dict:
        lat_bounds = (
            locations_dict[args.location]["lat_min"],
            locations_dict[args.location]["lat_max"],
        )
        lon_bounds = (
            locations_dict[args.location]["lon_min"],
            locations_dict[args.location]["lon_max"],
        )
        output_path = Path(f"data/processed/satellite/{args.location}/{args.product}")
    else:
        lat_bounds = args.lat_min, args.lat_max
        lon_bounds = args.lon_min, args.lon_max
        output_path = Path(f"data/processed/satellite/{args.product}")

    match args.product:
        case "ABI-L2-MCMIPF":  # Cloud and Moisture Imagery
            bands = ["CMI_C08", "CMI_C09", "CMI_C10", "CMI_C11"]
            include_dataset_name = False
        # Rainfall Rate (Quantitative Precipitation Estimate)
        case "ABI-L2-RRQPEF":
            bands = ["RRQPE"]
            include_dataset_name = False
        case "ABI-L2-DMWF":
            bands = [
                "wind_direction",
                "wind_speed",
                "temperature",
                "pressure",
            ]
            include_dataset_name = True
        case "ABI-L2-ACHAF":
            bands = ["HT"]
            include_dataset_name = False
        case "ABI-L2-LSTF":
            bands = ["LST", "DQF", "PQI"]
            include_dataset_name = False
        case _:
            raise ValueError("Unsupported product selected.")

    def load_entire_day(ts: pd.Timestamp) -> pd.DataFrame:
        year = ts.year
        day = ts.dayofyear
        return pd.concat(
            Parallel(n_jobs=args.n_jobs)(
                delayed(process_satellite)(file, bands, lat_bounds, lon_bounds, include_dataset_name)
                for file in tqdm(glob(f"data/raw/satellite/{args.product}/{year}/{day:03d}/*/*.nc"))
            )
        )

    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 1, 1)

    df_current = load_entire_day(pd.Timestamp(start_date))

    output_path.mkdir(exist_ok=True, parents=True)

    for date in tqdm(
        pd.date_range(start_date, end_date),
        desc="Saving files",
    ):
        output_file = f"{output_path}/{date.date()}.feather"
        output_file = Path(output_file)
        if output_file.exists():
            print(f"File {output_file} already exists. Skipping.")
            continue
        next_date = date + timedelta(days=1)
        df_next = load_entire_day(next_date)

        df = pd.concat([df_current, df_next])
        df = df[df["creation"].dt.date == date.date()]
        df = df.reset_index(drop=True)
        df.to_feather(output_file)

        df_current = df_next.copy()
        del df_next
