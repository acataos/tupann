import argparse
import datetime
import pathlib

import torch
import yaml
from tqdm import tqdm

from src.data.dataset_handlers import DatasetHandlerFactory

DT_FORMAT = "%Y-%m-%d %H:%M:%S"


def main():
    parser = argparse.ArgumentParser(description="Save motion fields and intensities for a dataset and datetimes.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--datetimes_file", type=str, required=True, help="List of datetimes")
    parser.add_argument("--locations", nargs="+", help="Locations to process", default=["rj"])
    parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing files")
    parser.add_argument("--timestep_radius", "-tr", help="Radius of timesteps for each interval", type=int)
    parser.add_argument("--threshold", "-thr", help="Threshold for rain", type=int)
    args = parser.parse_args()

    dataset = args.dataset
    datetimes_file = args.datetimes_file
    # read datetimes from file
    with open(datetimes_file, "r") as f:
        datetimes = [datetime.datetime.strptime(line.strip(), DT_FORMAT) for line in f.readlines()]

    dataset_handler = DatasetHandlerFactory.create_handler(dataset, args.locations)
    for location in args.locations:
        output_filepath = (
            f"data/rain_events/{args.dataset}-{location}-thr={args.threshold}-tr={args.timestep_radius}.yaml"
        )
        pathlib.Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        rain_events_list = []
        bounds = []
        for dt in tqdm(datetimes):
            context = 3
            dts = [dt + datetime.timedelta(minutes=i * 10) for i in range(-context, context + 1)]
            tensor = torch.nan_to_num(torch.tensor(dataset_handler.fetch(dts, location)))
            if tensor.sum() < args.threshold:
                continue
            lower_bound = dt - args.timestep_radius * datetime.timedelta(minutes=10)
            upper_bound = dt + args.timestep_radius * datetime.timedelta(minutes=10)
            if len(bounds):
                if lower_bound <= bounds[-1][1] and upper_bound >= bounds[-1][0]:
                    lower_bound = min(lower_bound, bounds[-1][0])
                    upper_bound = max(upper_bound, bounds[-1][1])
                    bounds[-1] = (lower_bound, upper_bound)
                    continue
            bounds.append((lower_bound, upper_bound))
        for bound in bounds:
            rain_events_list.append(
                {
                    "start": bound[0],
                    "end": bound[1],
                }
            )
        # dump rain events to yaml file
        yaml.safe_dump(
            rain_events_list,
            open(output_filepath, "w"),
            default_flow_style=False,
        )


if __name__ == "__main__":
    main()
