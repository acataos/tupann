import argparse
import pathlib
import random

import pandas as pd
import yaml

DT_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTEP_SIZE = 10
SPLIT_PROPORTION = [0.7, 0.85, 1]  # Train, Val, Test


def main(yaml_file, exclude_first=0, exclude_last=0):
    rain_events_list = yaml.safe_load(open(file))
    random.seed(42)  # For reproducibility
    random.shuffle(rain_events_list)

    total_len = len(rain_events_list)
    split_list = [
        rain_events_list[int(p * total_len): int(q * total_len)]
        for (p, q) in zip([0] + SPLIT_PROPORTION[:-1], SPLIT_PROPORTION)
    ]

    split_dict = {}
    for i, split in enumerate(["train", "val", "test"]):
        split_dict[split] = []
        for event in split_list[i]:
            start = event["start"]
            end = event["end"]
            dt_range = pd.date_range(
                start, end, freq=f"{TIMESTEP_SIZE}min")
            length = len(dt_range)
            dt_range = dt_range[exclude_first:length - exclude_last]
            split_dict[split].extend(dt_range)
    return split_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split rain events into train, val and test.")
    parser.add_argument("--yaml_file", "-yf",
                        help="Yaml file for datetimes parsing", type=str)
    parser.add_argument("--exclude_first", "-ef", type=int, default=0,
                        help="Number of timesteps to exclude from the start of each rain event")
    parser.add_argument("--exclude_last", "-el", type=int, default=0,
                        help="Number of timesteps to exclude from the end of each rain event")
    args = parser.parse_args()
    file = pathlib.Path(args.yaml_file)
    split_dict = main(file, args.exclude_first, args.exclude_last)
    for split, datetimes in split_dict.items():
        output_filepath = f"configs/data/{split}_datetimes-{file.stem}-exclude_first={args.exclude_first}-exclude_last={args.exclude_last}.txt"
        with open(output_filepath, "w") as f:
            for dt in datetimes:
                f.write(dt.strftime(DT_FORMAT) + "\n")
