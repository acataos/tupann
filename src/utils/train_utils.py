import datetime
import pathlib
from functools import partial

import numpy as np
import yaml
from tqdm import tqdm

from src.data.dataset_handlers import DatasetHandlerFactory
from src.utils.data_utils import get_transform_params_filename

DT_FORMAT = "%Y-%m-%d %H:%M:%S"
params_for_transform = {
    "standardize": ["mean", "std"],
    "identity": [],
    "minmax": ["min", "max"],
}


def get_parameter(param_name, dataset_name, location, datetimes_file):
    dataset_handler = DatasetHandlerFactory.create_handler(
        dataset_name, [location], split="train")
    with open(datetimes_file, "r") as f:
        datetimes = [datetime.datetime.strptime(
            line.strip(), DT_FORMAT) for line in f.readlines()]
    dataset = dataset_handler.fetch(datetimes, location)
    data_len = len(dataset)
    if param_name == "mean" or param_name == "std":
        data_mean = 0
        data_mean_squared = 0
        for i in tqdm(range(data_len)):
            data = np.nan_to_num(dataset[i])
            data_mean += data.mean()
            data_mean_squared += (data**2).mean()
        if param_name == "mean":
            return data_mean / data_len
        else:
            return (data_mean_squared / data_len - (data_mean / data_len) ** 2) ** 0.5
    elif param_name == "max":
        current_max = -np.inf
        for i in tqdm(range(data_len)):
            data = np.nan_to_num(dataset[i])
            data_max = data.max()
            if data_max > current_max:
                current_max = data_max
        return current_max
    elif param_name == "min":
        current_min = np.inf
        for i in tqdm(range(data_len)):
            data = np.nan_to_num(dataset[i])
            data_min = data.min()
            if data_min < current_min:
                current_min = data_min
        return current_min
    else:
        raise ValueError(f"Requested parameter {param_name} not recognized.")


def fetch_transform_params(data_config):
    data_dict = yaml.safe_load(
        pathlib.Path(f"configs/data/{data_config}.yaml").read_text(),
    )
    locations = data_dict["train"]["locations"]
    inputs = list(data_dict["input"].items())
    targets = list(data_dict["target"].items())
    input_targets = inputs + targets
    full_dict = {}
    for dataset_name, dataset_specs in input_targets:
        full_dict[dataset_name] = {}
        for i, location in enumerate(locations):
            full_dict[dataset_name][location] = {}
            datetimes_file_name = data_dict["train"]["datetimes"]
            if isinstance(datetimes_file_name, str):
                datetimes_file_name = [datetimes_file_name]
            datetimes_file = datetimes_file_name[i]
            transform_params = params_for_transform.get(
                dataset_specs.get("transform", "identity"))
            transform_params_file = pathlib.Path(
                get_transform_params_filename(
                    dataset_name,
                    datetimes_file,
                )
            )
            if pathlib.Path(transform_params_file).exists():
                transform_params_dict = yaml.safe_load(
                    pathlib.Path(transform_params_file).read_text())
            else:
                transform_params_dict = {}
                transform_params_file.parents[0].mkdir(
                    parents=True, exist_ok=True)

                if location not in transform_params_dict:
                    transform_params_dict[location] = {}
                for param_name in transform_params:
                    if param_name not in transform_params_dict[location]:
                        transform_param = get_parameter(
                            param_name,
                            dataset_name,
                            location,
                            datetimes_file,
                        )
                        transform_params_dict[location][param_name] = transform_param.item(
                        )
            full_dict[dataset_name][location] = transform_params_dict[location]
            yaml.dump(
                transform_params_dict,
                pathlib.Path(transform_params_file).open("w"),
                default_flow_style=False,
            )

    return full_dict


def inv_standardize(X, **kwargs):
    """
    Standardize the input tensor by subtracting the mean and dividing by the standard deviation.

    Args:
        X (torch.Tensor): Input tensor to be standardized.

    Returns:
        torch.Tensor: Standardized tensor.
    """
    mean = kwargs["mean"]
    std = kwargs["std"]
    if isinstance(X, list):
        return [std * x.nan_to_num() + mean for x in X]
    return std * X.nan_to_num() + mean


def standardize_transform(X, **kwargs):
    """
        Standardize the input tensor by subtracting the mean and dividing by the standard deviation.

        Args:
        X (torch.Tensor): Input tensor to be standardized.

    Returns:
        torch.Tensor: Standardized tensor.
    """

    mean = kwargs["mean"]
    std = kwargs["std"]
    if isinstance(X, list):
        return [(x.nan_to_num() - mean) / std for x in X]
    return (X.nan_to_num() - mean) / std


def inv_minmax_transform(X, **kwargs):
    """
    Inverse Min-Max normalization of the input tensor.

    Args:
        X (torch.Tensor): Input tensor to be inverse normalized.

    Returns:
        torch.Tensor: Inverse normalized tensor.
    """
    min_val = kwargs["min"]
    max_val = kwargs["max"]
    if isinstance(X, list):
        return [x.nan_to_num() * (max_val - min_val) + min_val for x in X]
    return X.nan_to_num() * (max_val - min_val) + min_val


def minmax_transform(X, **kwargs):
    """
    Min-Max normalization of the input tensor.

    Args:
        X (torch.Tensor): Input tensor to be normalized.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    min_val = kwargs["min"]
    max_val = kwargs["max"]
    if isinstance(X, list):
        return [(x.nan_to_num() - min_val) / (max_val - min_val) for x in X]
    return (X.nan_to_num() - min_val) / (max_val - min_val)


def identity_transform(X, **kwargs):
    """
    Identity transformation of the input tensor.

    Args:
        X (torch.Tensor): Input tensor to be transformed.

    Returns:
        torch.Tensor: Transformed tensor.
    """
    if isinstance(X, list):
        return [x.nan_to_num() for x in X]
    return X.nan_to_num()


inv_transforms = {
    "standardize": inv_standardize,
    "identity": identity_transform,
    "minmax": inv_minmax_transform,
}

transformations = {
    "standardize": standardize_transform,
    "identity": identity_transform,
    "minmax": minmax_transform,
}


def get_transforms(data_dict, folds, params, targets):
    # get inverse transform
    transforms_output = {}
    inv_transforms_output = {}
    for target in targets:
        transform = data_dict["target"][target].get("transform", "identity")
        if target.split("#")[0] in ["fields_intensities", "tupann_autoenc"]:
            target = target.split("#")[0]

        inv_transforms_output[target] = {}
        transforms_output[target] = {}
        inv_transform_name = inv_transforms.get(transform, "identity")
        transform_name = transformations.get(transform, "identity")
        params_needed = params_for_transform[transform]
        transform_parameters = {}
        for fold in folds:
            locations = data_dict[fold]["locations"]
            for location in locations:
                if location in transform_parameters.keys():
                    continue
                transform_parameters[location] = {}
                if len(params_needed) == 0:
                    transform_parameters[location] = []
                    inv_transforms_output[target][location] = inv_transform_name
                    transforms_output[target][location] = transform_name
                else:
                    for param in params_needed:
                        transform_parameters[location][param] = params[target][location][param]
                        inv_transforms_output[target][location] = partial(
                            inv_transform_name, **transform_parameters[location]
                        )
                        transforms_output[target][location] = partial(
                            transform_name, **transform_parameters[location])
    return transforms_output, inv_transforms_output
