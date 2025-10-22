import argparse
import pathlib
import random

import pandas as pd
import yaml

DT_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTEP_SIZE = 10
SPLIT_PROPORTION = [0.7, 0.85, 1]  # Train, Val, Test


def main():
    parser = argparse.ArgumentParser(description="Split rain events into train, val and test.")
    parser.add_argument("--yaml_file", "-yf", help="Yaml file for datetimes parsing", type=str)
    parser.add_argument("--fields_intensities", "-fi", help="Extended timestamps", action="store_true")
    args = parser.parse_args()
    file = pathlib.Path(args.yaml_file)
    rain_events_list = yaml.safe_load(open(file))
    random.seed(42)  # For reproducibility
    random.shuffle(rain_events_list)

    total_len = len(rain_events_list)
    split_list = [
        rain_events_list[int(p * total_len) : int(q * total_len)]
        for (p, q) in zip([0] + SPLIT_PROPORTION[:-1], SPLIT_PROPORTION)
    ]

    for i, split in enumerate(["train", "val", "test"]):
        if args.fields_intensities:
            output_filepath = f"configs/data/{split}_FI_datetimes-{file.stem}.txt"
        else:
            output_filepath = f"configs/data/{split}_datetimes-{file.stem}.txt"
        with open(output_filepath, "w") as f:
            for event in split_list[i]:
                start = event["start"]
                end = event["end"]
                dt_range = pd.date_range(start, end, freq=f"{TIMESTEP_SIZE}min")
                if args.fields_intensities:
                    dt_range = dt_range[10:-1]
                else:
                    dt_range = dt_range[10:-18]
                for dt in dt_range:
                    f.write(dt.strftime(DT_FORMAT) + "\n")


if __name__ == "__main__":
    main()
