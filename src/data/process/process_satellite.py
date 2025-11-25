import gc
import yaml
from argparse import ArgumentParser
import datetime
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from pyproj import Proj
from tqdm import tqdm

locations_dict = {
    "rio_de_janeiro": {
        "lat_min": -26.0,
        "lat_max": -19.0,
        "lon_min": -47.0,
        "lon_max": -40.0
    },
    "la_paz": {
        "lat_min": -20.5,
        "lat_max": -12.5,
        "lon_min": -72.0,
        "lon_max": -64.0,
    },
    "manaus": {
        "lat_min": -7.0,
        "lat_max": 1.0,
        "lon_min": -64.0,
        "lon_max": -56.0,
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


def process_file(
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
    creation = datetime.datetime.strptime(
        dataset.date_created, "%Y-%m-%dT%H:%M:%S.%fZ")

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
        np.array(data).astype(dtype=np.float32).reshape(len(data), -1).T,
        columns=["lat", "lon", *bands],
    )

    # Remove nan and infinite values
    df.replace(np.inf, np.nan, inplace=True)
    df.dropna(how="any", axis=0, inplace=True)

    # Discard observations for latitudes and longitudes outside bounds
    if lat_bounds:
        df.drop(df[(df["lat"] <= lat_bounds[0]) | (
            df["lat"] >= lat_bounds[1])].index, inplace=True)
    if lon_bounds:
        df.drop(df[(df["lon"] <= lon_bounds[0]) | (
            df["lon"] >= lon_bounds[1])].index, inplace=True)

    # Include datetimes of file creation and start and end of scan (UTC-3)
    df["creation"] = creation  # - timedelta(hours=3)

    if include_dataset_name:
        df["name"] = "_".join(dataset.dataset_name.split("_")[:2])
    gc.collect()
    return df


def process_satellite(
    rain_events_file, product="ABI-L2-RRQPEF", location="rio_de_janeiro", num_workers=16
):
    # open yaml file with rain events information
    rain_events_name = Path(rain_events_file).stem
    with open(rain_events_file, "r") as file:
        rain_events_list = yaml.safe_load(file)

    match product:
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
        case _:
            raise ValueError("Unsupported product selected.")

    lat_min = locations_dict[location]["lat_min"]
    lat_max = locations_dict[location]["lat_max"]
    lon_min = locations_dict[location]["lon_min"]
    lon_max = locations_dict[location]["lon_max"]

    lat_bounds = lat_min, lat_max
    lon_bounds = lon_min, lon_max

    for i, event_dict in enumerate(rain_events_list):
        start = pd.to_datetime(event_dict["start"])
        end = pd.to_datetime(event_dict["end"])+datetime.timedelta(hours=1)
        files = set()
        for dt in pd.date_range(start, end, freq="1h"):
            year = dt.year
            day = dt.timetuple().tm_yday
            hour = dt.hour
            files = files.union(
                glob(f"data/raw/satellite/{product}/{year}/{day:03d}/{hour:02d}/*.nc"))
        files = list(files)
        dataframe = pd.concat(
            Parallel(n_jobs=num_workers)(
                delayed(process_file)(file, bands, lat_bounds, lon_bounds, include_dataset_name) for file in tqdm(files)
            )
        )
        dataframe.reset_index(drop=True, inplace=True)
        output_path = Path(
            f"data/processed/satellite/{product}/{location}/{rain_events_name}")
        output_path.mkdir(exist_ok=True, parents=True)
        dataframe.to_feather(
            f"{output_path}/event_id={i:04d}.feather")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("rain_events_file", type=str)
    parser.add_argument("--product", type=str, default="ABI-L2-RRQPEF")
    parser.add_argument("--location", type=str, default="rio_de_janeiro")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    process_satellite(**vars(args))
