import argparse
import pathlib

import pandas as pd
import yaml

DT_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTEP_SIZE = 10


def main(rain_events_file, exclude_first=0, exclude_last=0):
    file = pathlib.Path(rain_events_file)
    rain_events_list = yaml.safe_load(open(file))
    results = []
    for event in rain_events_list:
        start = event["start"]
        end = event["end"]
        dt_range = pd.date_range(start, end, freq=f"{TIMESTEP_SIZE}min")
        dt_range = dt_range[exclude_first: len(dt_range)-exclude_last]
        results.extend(dt_range)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save motion fields and intensities for a dataset and datetimes.")
    parser.add_argument("--yaml_file", "-yf",
                        help="Yaml file for datetimes parsing", type=str)
    parser.add_argument("--exclude_first", "-ef",
                        help="Exclude first N timesteps", type=int, default=0)
    parser.add_argument("--exclude_last", "-el",
                        help="Exclude last N timesteps", type=int, default=0)
    args = parser.parse_args()
    file = pathlib.Path(args.yaml_file)
    output_filepath = f"configs/data/datetimes[{args.exclude_first}:-{args.exclude_last}]-{file.stem}-.txt"
    results = main(args.yaml_file, args.exclude_first, args.exclude_last)
    with open(output_filepath, "w") as f:
        for dt in results:
            f.write(dt.strftime(DT_FORMAT) + "\n")
