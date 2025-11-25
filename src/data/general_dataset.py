import datetime
import itertools
import pathlib
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from src.data.dataset_handlers import DatasetHandlerFactory
from src.utils.data_utils import calc_dict_hash, center_crop
from src.utils.train_utils import fetch_transform_params, transformations

DT_FORMAT = "%Y-%m-%d %H:%M:%S"
MIN_WEIGHT = 100


class GeneralDataset(Dataset):
    def __init__(self, dataset_specs, return_metadata=False):
        """
        Initialize the GeneralDataset class.

        Args:s
            dataset_specs (dict): A dictionary containing dataset specifications.
        """
        super().__init__()
        self.dataset_specs = dataset_specs
        self.locations = dataset_specs["locations"]
        self.datetimes_files = dataset_specs["datetimes"]
        if isinstance(self.datetimes_files, str):
            self.datetimes_files = [self.datetimes_files]
        self.datetimes_dict = {}
        for location, datetime_file in zip(self.locations, self.datetimes_files):
            with open(datetime_file, "r") as f:
                self.datetimes_dict[location] = [
                    datetime.datetime.strptime(line.strip(), DT_FORMAT) for line in f.readlines()
                ]
        self.input = dataset_specs["input"]
        self.target = dataset_specs["target"]
        self.config_name = dataset_specs["config"]
        self.split = dataset_specs["split"]
        self.transform_parameters = fetch_transform_params(self.config_name)
        self.return_metadata = return_metadata
        # assert target size is constant
        assert (
            len(set([len(target_lags["lags"]) for target_lags in self.target.values()])) == 1
        ), "Target size must be constant across locations"
        self.target_length = len(list(self.target.values())[0]["lags"])
        self.leadtime_length = dataset_specs.get("leadtime_length", self.target_length)
        assert self.leadtime_length <= self.target_length, "Leadtime length must be less than or equal to target length"
        self.dt_len = np.cumsum(np.array([len(dts) for dts in self.datetimes_dict.values()]))
        self.ndt = self.dt_len[-1]
        self.nlt = self.target_length - self.leadtime_length + 1

        self._handlers = {}
        self._initialize_handlers()

    def _initialize_handlers(self):
        """Initialize dataset handlers for all datasets used in config."""
        all_datasets = set()

        # Collect all dataset names from inputs and targets
        for dataset_name in self.input.keys():
            all_datasets.add(dataset_name)

        for dataset_name in self.target.keys():
            all_datasets.add(dataset_name)

        # Initialize handlers for each unique dataset
        for dataset_name in all_datasets:
            self._handlers[dataset_name] = DatasetHandlerFactory.create_handler(
                dataset_name,
                locations=self.locations,
                split=self.split,
            )

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.ndt * self.nlt

    def __getitem__(self, index, calc_weights=False):
        identity_transform = transformations["identity"]
        if calc_weights:
            no_transform = True
            nlt = 1
            leadtime_length = self.target_length
        else:
            no_transform = False
            nlt = self.nlt
            leadtime_length = self.leadtime_length

        dt_index = index // nlt
        lt_index = index % nlt
        input_dict = OrderedDict()
        loc_index = np.sum(self.dt_len <= dt_index)
        for dataset_name, input_specs in self.input.items():
            data_key = dataset_name
            if "#" in dataset_name:
                data_key = dataset_name.split("#")[0]
            lags = input_specs["lags"]
            if not no_transform:
                transform = transformations[input_specs.get("transform", "identity")]
                transformation = partial(
                    transform,
                    **self.transform_parameters[dataset_name][self.locations[loc_index]],
                )
            else:
                transformation = identity_transform
            curr_dt = self.datetimes_dict[self.locations[loc_index]][
                dt_index - (0 if loc_index == 0 else self.dt_len[loc_index - 1])
            ]
            dts = [curr_dt + datetime.timedelta(minutes=lag) for lag in lags]
            input_dict[data_key] = transformation(
                torch.from_numpy(
                    self._handlers[dataset_name].fetch(
                        dts,
                        self.locations[loc_index],
                        curr_dt=curr_dt,
                        split=self.split,
                    )
                )
            ).float()
            if calc_weights:
                return input_dict[data_key]

        target_dict = OrderedDict()
        for dataset_name, target_specs in self.target.items():
            data_key = dataset_name
            if "#" in dataset_name:
                data_key = dataset_name.split("#")[0]
            lags = target_specs["lags"]
            if not no_transform:
                transform = transformations[target_specs.get("transform", "identity")]
                transformation = partial(
                    transform,
                    **self.transform_parameters[dataset_name][self.locations[loc_index]],
                )
            else:
                transformation = identity_transform
            dts = [curr_dt + datetime.timedelta(minutes=lag) for lag in lags[lt_index : lt_index + leadtime_length]]
            target_dict[data_key] = transformation(
                torch.from_numpy(
                    self._handlers[dataset_name].fetch(
                        dts,
                        self.locations[loc_index],
                        curr_dt=curr_dt,
                        split=self.split,
                    )
                )
            ).float()
        if self.return_metadata:
            return (
                input_dict,
                target_dict,
                lt_index,
                {
                    "location": self.locations[loc_index],
                    "datetime": str(curr_dt),
                    "index": index,
                },
            )
        return (input_dict, target_dict, lt_index)

    # This function computes sample weights based on the input data.
    # THE WEIGHTS ARE BASED ONLY ON THE FIRST INPUT DATASET.
    def get_sample_weights(self, overwrite_if_exists=False):
        weights_list = []
        assert self.split == "train", "Sample weights can only be computed for the training set."
        if (
            self.config_name == "goes16_cascast"
            or self.config_name == "goes16_cascast_indentity"
            or self.config_name == "goes16_cascast_indentity_new"
            or self.config_name == "goes16_cascast_standard_new"
        ):
            weights_filepath = pathlib.Path("data/weights/goes16_rio.npy")
        elif self.config_name == "goes16_cascast_indentity_toronto":
            weights_filepath = pathlib.Path("data/weights/goes16_toronto.npy")
        elif self.config_name == "goes16_cascast_indentity_manaus":
            weights_filepath = pathlib.Path("data/weights/goes16_manaus.npy")
        elif self.config_name == "goes16_cascast_indentity_miami":
            weights_filepath = pathlib.Path("data/weights/goes16_miami.npy")
        else:
            # breakpoint()
            data_size = np.arange(len(self))
            loc_index = np.sum(np.array(self.dt_len).reshape(-1, 1) <= data_size.reshape(1, -1), axis=0)
            for loc, dt_file in enumerate(self.datetimes_files):
                weights_dict = {
                    "dataset_name": list(self.input.keys())[0],
                    "locations": [self.locations[loc]],
                    "lags": list(self.input.values())[0]["lags"],
                    "datetimes": dt_file,
                }
                print(weights_dict)
                weights_hash = calc_dict_hash(weights_dict)
                # if self.cropped_window is not None:
                #     weights_hash += f"_{self.cropped_window}"
                weights_filepath = pathlib.Path(f"data/weights/{weights_hash}.npy")
                print(weights_filepath)
                if not overwrite_if_exists and weights_filepath.is_file():
                    weights = np.load(weights_filepath)
                    weights = weights / weights.sum()
                    weights = np.tile(weights, (self.nlt, 1)).T.flatten()

                    weights_list.append(weights)
                else:
                    def task(i):
                        X = self.__getitem__(i, calc_weights=True)
                        X[X < 0] = 0
                        assert torch.all(X >= -0.000001), "Input data must be positive for weight calculation."
                        weight = torch.sum(1 - torch.exp(-torch.nan_to_num(X)))
                        weight += MIN_WEIGHT
                        return weight

                    # cpu_count = os.cpu_count()
                    # weights = Parallel(n_jobs=cpu_count)(delayed(task)(i) for i in tqdm.tqdm(range(len(self) // self.nlt)))
                    idx = np.where(loc_index == loc)[0]
                    weights = [task(i) for i in tqdm.tqdm(idx)]
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    weights_filepath.parents[0].mkdir(parents=True, exist_ok=True)
                    np.save(weights_filepath, weights)
                    weights = np.tile(weights, (self.nlt, 1)).T.flatten()
                    weights_list.append(weights)
            weights = np.concatenate(weights_list)
            # breakpoint()
            assert len(weights) == len(self)
            return weights

        weights = []

        def task(i):
            X = self.__getitem__(i, calc_weights=True)
            # if self.cropped_window is not None:
            #     X = center_crop(X, self.cropped_window, self.cropped_window)
            X[X < 0] = 0
            assert torch.all(X >= -0.000001), "Input data must be positive for weight calculation."
            weight = torch.sum(1 - torch.exp(-torch.nan_to_num(X)))
            weight += MIN_WEIGHT
            return weight

        # cpu_count = os.cpu_count()
        # weights = Parallel(n_jobs=cpu_count)(delayed(task)(i) for i in tqdm.tqdm(range(len(self) // self.nlt)))
        weights = [task(i) for i in tqdm.tqdm(range(len(self) // self.nlt))]
        weights = np.array(weights)
        weights = weights / weights.sum()
        np.save(weights_filepath, weights)
        weights = np.tile(weights, (self.nlt, 1)).T.flatten()

        assert len(weights) == len(self)
        return weights

    def get_locations(self):
        """
        Return the list of locations in the dataset.

        Returns:
            list: List of locations.
        """
        data_size = np.arange(len(self))
        dt_index = data_size // self.nlt
        loc_index = np.sum(np.array(self.dt_len).reshape(-1, 1) <= dt_index.reshape(1, -1), axis=0)
        locations = [self.locations[i] for i in loc_index]
        return locations

    def get_datetimes(self):
        """
        Return the list of datetimes in the dataset.

        Returns:
            list: List of datetimes.
        """
        data_size = np.arange(len(self))
        dt_index = data_size // self.nlt
        concat_lists = list(itertools.chain(*self.datetimes_dict.values()))
        datetimes = [concat_lists[i] for i in dt_index]
        return datetimes
