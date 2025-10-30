import argparse
import pandas as pd
from collections import defaultdict
import yaml
import datetime
import pathlib

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy import interpolate
from tqdm import tqdm

from src.data.process.SatelliteData import SatelliteData

DT_FORMAT = "%Y-%m-%d %H:%M:%S"


def interp_at_grid(x, y, values, target_grid):
    nan_indices = np.logical_or(np.isnan(x), np.isnan(y))
    x = x[~nan_indices]
    y = y[~nan_indices]
    values = values[~nan_indices]
    points = np.stack((x, y)).T
    shape = target_grid.shape[:2]
    try:
        interp_values = interpolate.griddata(
            points,
            values,
            (target_grid[:, :, 1].flatten(), target_grid[:, :, 0].flatten()),
            method="linear",
        ).reshape(shape)
    except ValueError:
        return np.zeros(shape)

    return interp_values


parser = argparse.ArgumentParser()
parser.add_argument("rain_events_file", type=str, help="List of datetimes")
parser.add_argument(
    "--overwrite",
    "-o",
    action="store_true",
    help="If true, overwrites output; otherwise, skips existing files.",
)
parser.add_argument(
    "--product",
    default="ABI-L2-RRQPEF",
    help="Satellite product to be processed.",
    type=str,
)
parser.add_argument(
    "--n_processes",
    "-n",
    type=int,
    default=1,
    help="Number of processes for parallelization.",
)
parser.add_argument(
    "--location",
    "-loc",
    default="rio_de_janeiro",
    type=str,
    help="Location to build dataframe.",
)
parser.add_argument("--timestep", "-ts", default=10,
                    type=int, help="Timestep in minutes.")
parser.add_argument("--value", "-val", default="RRQPE",
                    type=str, help="Satellite value.")
parser.add_argument("--band", "-b", default="RRQPE",
                    type=str, help="Satellite band.")
args = parser.parse_args()

assert args.location in [
    "rio_de_janeiro",
    "manaus",
    "la_paz",
    "miami",
    "toronto",
]

sat_folder = f"data/processed/satellite/{args.product}/{args.location}"

output_filename = f"{args.location}"
with open(args.rain_events_file, "r") as file:
    rain_events_list = yaml.safe_load(file)

output_filepath = pathlib.Path(
    f"data/goes16_rrqpe/SAT-{args.product}-{pathlib.Path(output_filename).stem}.hdf"
)

pathlib.Path(output_filepath).parents[0].mkdir(parents=True, exist_ok=True)

grid = np.load(
    f"data/dataframe_grids/{args.location}-res=2km-256x256.npy").astype(np.float32)
ni, nj = grid.shape[:2]

with h5py.File(output_filepath, "a") as f:
    if "what" not in f:
        what = f.create_group("what")
        what.attrs["feature"] = args.product
        what.attrs["process_type"] = "-"
        what.attrs["ni"] = ni
        what.attrs["nj"] = nj
        what.attrs["timestep"] = args.timestep

    step = args.n_processes

    event_to_dts = defaultdict(list)
    found = False
    for i, event in enumerate(rain_events_list):
        for dt in pd.date_range(event["start"], event["end"], freq=f"{args.timestep}min"):
            # write only the dates that weren't processed yet or corrupted array
            if not found:
                datetime_key = dt.strftime("%Y%m%d/%H%M")
                try:
                    arr = np.array(f[datetime_key])
                    continue
                except KeyError:
                    print(
                        "{dt} not found in file, adding to processing list".format(dt=dt))
                    found = True
                except Exception as e:
                    print(e)
                    print(
                        "corrupted array for {dt}, adding to processing list".format(dt=dt))
                    found = True
            event_to_dts[i].append(dt)

    import time

    for event_id, dts in tqdm(event_to_dts.items(), total=len(event_to_dts)):
        t0 = time.time()
        df_file = f"{sat_folder}/event_id={i:04d}.feather"
        sd = SatelliteData.load_data(
            df_file, product=args.product, value=args.value)
        sd.cast_to_float32()
        results = []
        data_dict = {}
        creation = sd.data["creation"].values
        lon_arr = sd.data["lon"].values
        lat_arr = sd.data["lat"].values
        val_arr = sd.data[sd.value].values

        unique_ts, inverse_indices = np.unique(
            creation, return_inverse=True)
        timestamp_to_indices = {ts: np.where(inverse_indices == i)[
            0].tolist() for i, ts in enumerate(unique_ts)}

        sorted_creation = np.sort(np.unique(creation))

        for timestamp in dts:
            time_itr = time.time()
            idx = (
                np.searchsorted(
                    sorted_creation.astype("datetime64[us]"),
                    timestamp + datetime.timedelta(minutes=2),
                    side="right",
                )
                - 1
            )
            if idx < 0:
                x = np.array([])
                y = np.array([])
                values = np.array([])
            else:
                closest_timestamp = sorted_creation[idx]
                indices = timestamp_to_indices[closest_timestamp]
                x = lon_arr[indices]
                y = lat_arr[indices]
                values = val_arr[indices]
            data_dict[timestamp] = {
                "x": x,
                "y": y,
                "values": values,
            }

        def process_dt(dt, x, y, values):
            datetime_key = dt.strftime("%Y%m%d/%H%M")
            data = interp_at_grid(x, y, values, grid)
            assert data.shape == (ni, nj)
            return datetime_key, data

        results = Parallel(n_jobs=args.n_processes)(
            delayed(process_dt)(
                dt, data_dict[dt]["x"], data_dict[dt]["y"], data_dict[dt]["values"])
            for dt in tqdm(dts)
        )

        for datetime_key, data in tqdm(results):
            f.create_dataset(datetime_key, data=data.astype(
                np.float32), compression="lzf")
