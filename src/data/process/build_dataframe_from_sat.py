import argparse
import datetime
import glob
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
parser.add_argument("--datetimes_file", type=str, required=True, help="List of datetimes")
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Print verbose output.",
)
parser.add_argument(
    "--overwrite",
    "-o",
    action="store_true",
    help="If true, overwrites output; otherwise, skips existing files.",
)
parser.add_argument(
    "--product",
    "-pr",
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
parser.add_argument("--timestep", "-ts", default=10, type=int, help="Timestep in minutes.")
parser.add_argument("--value", "-val", default="RRQPE", type=str, help="Satellite value.")
parser.add_argument("--band", "-b", default="RRQPE", type=str, help="Satellite band.")
args = parser.parse_args()

assert args.location in [
    "rj",
    "manaus",
    "la_paz",
    "miami",
    "toronto",
]

if args.location == "rj":
    sat_folder = f"../rio_rain/data/processed/satellite/rio_de_janeiro/{args.product}"
else:
    sat_folder = f"data/processed/satellite/{args.location}/{args.product}"

output_filename = f"{args.location}"
with open(args.datetimes_file, "r") as f:
    datetimes = [datetime.datetime.strptime(line.strip(), DT_FORMAT) for line in f.readlines()]
full_datetimes = list(datetimes)
full_datetimes = sorted(full_datetimes)

output_filepath = pathlib.Path(
    f"data/goes16_rrqpe_rain_events/SAT-{args.product}-{pathlib.Path(output_filename).stem}.hdf"
)

pathlib.Path(output_filepath).parents[0].mkdir(parents=True, exist_ok=True)

grid_small = np.load(f"data/dataframe_grids/{args.location}-res=2km-256x256.npy").astype(np.float32)
grid_large = np.load(f"data/dataframe_grids/{args.location}-res=4km-256x256.npy").astype(np.float32)
assert grid_small.shape == grid_large.shape
ni, nj = grid_small.shape[:2]
with h5py.File(output_filepath, "a") as f:
    if "what" not in f:
        what = f.create_group("what")
        what.attrs["feature"] = args.product
        what.attrs["process_type"] = "-"
        what.attrs["ni"] = ni
        what.attrs["nj"] = nj
        what.attrs["timestep"] = args.timestep
        what.create_dataset(
            "datetime_keys",
            data=np.asarray([dt.strftime("%Y%m%d/%H%M") for dt in datetimes], dtype="S"),
        )

    step = args.n_processes
    size = len(full_datetimes)

    from collections import defaultdict

    date_to_dts = defaultdict(list)
    found = 0
    for dt in full_datetimes:
        # write only the dates that weren't processed yet or corrupted array
        if not found:
            datetime_key = dt.strftime("%Y%m%d/%H%M")
            try:
                arr = np.array(f[datetime_key])
                continue
            except KeyError:
                print("{dt} not found in file, adding to processing list".format(dt=dt))
                found = 1
            except Exception as e:
                print(e)
                print("corrupted array for {dt}, adding to processing list".format(dt=dt))
                found = 1
        date_to_dts[dt.date()].append(dt)

    import time

    for date, dts in tqdm(date_to_dts.items(), total=len(date_to_dts)):
        date_str = date.strftime("%Y-%m-%d")
        t0 = time.time()
        file_list = list(glob.glob(f"{sat_folder}/{date_str}.feather", recursive=True))
        print(f"File search for {date_str}: {time.time() - t0:.2f}s")

        assert (
            len(file_list) <= 1
        ), f"there should be at most one filepath with desired format {sat_folder}/{date_str}.feather"

        t1 = time.time()
        if len(file_list) < 1:
            results = []
            for dt in dts:
                datetime_key = dt.strftime("%Y%m%d/%H%M")
                data = np.ones((ni, nj, 2)) * np.nan
                results.append((datetime_key, data))
        else:
            file = file_list[0]
            t_load = time.time()
            sd = SatelliteData.load_data(file, args.value)
            t_corr = time.time()
            print(t_corr - t_load, "s to load and correct data for", file)
            t1 = time.time()
            sd._load_previous_day()
            # try:
            #     sd = sd.correct_parallax()
            # except Exception as e:
            #     print(f"Error correcting parallax for {file}: {e}")
            print("time to load and correct data:", time.time() - t1)
            sd.cast_to_float32()
            results = []
            data_dict = {}
            creation = sd.data["creation"].values
            lon_arr = sd.data["lon"].values
            lat_arr = sd.data["lat"].values
            val_arr = sd.data[sd.value].values

            from collections import defaultdict

            print("time to prepare data for", date_str, ":", time.time() - t1)
            # timestamp_to_indices = defaultdict(list)
            # for idx, ts in enumerate(creation):
            #     timestamp_to_indices[ts].append(idx)
            unique_ts, inverse_indices = np.unique(creation, return_inverse=True)
            timestamp_to_indices = {ts: np.where(inverse_indices == i)[0].tolist() for i, ts in enumerate(unique_ts)}

            print("time else", time.time() - t1)
            sorted_creation = np.sort(np.unique(creation))
            print("time to sort and else", time.time() - t1)

            t1 = time.time()
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
            print(f"time to prepare data for {date_str}: {time.time() - t1:.2f}s")

            def process_dt(dt, x, y, values):
                datetime_key = dt.strftime("%Y%m%d/%H%M")
                data_small = interp_at_grid(x, y, values, grid_small)
                data_large = interp_at_grid(x, y, values, grid_large)
                assert data_small.shape == (ni, nj)
                assert data_large.shape == (ni, nj)
                data = np.dstack([data_small, data_large])
                return datetime_key, data

            results = Parallel(n_jobs=args.n_processes)(
                delayed(process_dt)(dt, data_dict[dt]["x"], data_dict[dt]["y"], data_dict[dt]["values"])
                for dt in tqdm(dts)
            )

        print(f"total time for {date_str}: {time.time() - t1:.2f}s")

        for datetime_key, data in tqdm(results):
            assert data.shape == (ni, nj, 2), f"data shape mismatch for {datetime_key}: {data.shape}"
            f.create_dataset(datetime_key, data=data.astype(np.float32), compression="lzf")
