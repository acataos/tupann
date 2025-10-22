import argparse
import pathlib

import pandas as pd
import yaml

DT_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTEP_SIZE = 10


def main():
    parser = argparse.ArgumentParser(description="Save motion fields and intensities for a dataset and datetimes.")
    parser.add_argument("--yaml_file", "-yf", help="Yaml file for datetimes parsing", type=str)
    parser.add_argument("--inner", "-in", help="Boolean to indicate whether to use inner datetimes", type=int)
    args = parser.parse_args()
    file = pathlib.Path(args.yaml_file)
    rain_events_list = yaml.safe_load(open(file))

    if args.inner == 1:
        output_filepath = f"configs/data/inner_datetimes-{file.name}.txt"
    elif args.inner == 2:
        output_filepath = f"configs/data/FI_datetimes-{file.name}.txt"
    else:
        output_filepath = f"configs/data/datetimes-{file.name}.txt"
    with open(output_filepath, "w") as f:
        for event in rain_events_list:
            start = event["start"]
            end = event["end"]
            dt_range = pd.date_range(start, end, freq=f"{TIMESTEP_SIZE}min")
            if args.inner == 1:
                dt_range = dt_range[10:-18]
            elif args.inner == 2:
                dt_range = dt_range[10:]
            for dt in dt_range:
                f.write(dt.strftime(DT_FORMAT) + "\n")


if __name__ == "__main__":
    main()
