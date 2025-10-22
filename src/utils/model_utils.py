import hashlib
import json
import pathlib
import time

import torch

from src.utils.general_utils import get_logger


def calc_ckpt_code(old_code: str, filepath: pathlib.Path):
    try:
        with open(filepath) as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = []
    lines = [line.strip("\n") for line in lines]
    lines_to_compare = [line for line in lines if len(line) == len(old_code) + 1 and line[:-1] == old_code]
    if lines_to_compare == []:
        return old_code + "0"
    lines_to_compare.sort()
    last_char = lines_to_compare[-1][-1]
    new_ord = ord(last_char) + 1
    if new_ord >= 58 and new_ord <= 96:
        new_ord = 97
    elif new_ord >= 123:
        raise ValueError("Code went beyond limit.")
    return old_code + chr(new_ord)


def get_model_path(hparams, model_name, dataframe):
    model_parameters = sorted(hparams.keys())
    submodel_name = ""
    submodel_hash = ""
    for parameter in model_parameters:
        submodel_name += f"--{parameter}={hparams[parameter]} "
        submodel_hash += f"{parameter}={hparams[parameter]}-"
    submodel_name = submodel_name[:-1]
    submodel_hash = submodel_hash[:-1]
    submodel_name = f"{model_name}/{submodel_name}"
    submodel_hash = f"{model_name}/{submodel_hash}"
    hash = hashlib.md5(submodel_hash.encode()).hexdigest()

    model_path = f"{model_name}/{(hash)}"
    new_model_path_dict = {
        "model_name": submodel_name,
        "model_path": model_path,
    }
    output_path = f"models/{model_path}/train/{dataframe}"
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    # define logger
    logger = get_logger(
        name="train_logger",
        save_dir=output_path,
        distributed_rank=0,
        filename="log.log",
        resume=True,
    )

    # save model params
    tmp_filepath = pathlib.Path(".model_names.tmp")
    lock(tmp_filepath, logger)
    with open("model_names.jsonl", "a") as outfile:
        json.dump(new_model_path_dict, outfile)
        outfile.write("\n")
    unlock(tmp_filepath)
    return model_path, output_path, logger


def lock(tmp_filepath, logger):
    while True:
        try:
            open(tmp_filepath, "x")
            break
        except FileExistsError:
            logger.warn(f"tmp file {tmp_filepath} already exists. Waiting...")
            time.sleep(0.1)


def unlock(tmp_filepath):
    tmp_filepath.unlink()


def overwrite_file(filename, overwrite, logger):
    if isinstance(filename, dict):
        for value in filename.values():
            overwrite_file(value, overwrite, logger)
        return
    if isinstance(filename, str):
        filename = pathlib.Path(filename)
    if filename.is_file():
        if overwrite:
            warning_message = f"Warning: overwriting existing file {filename}"
            logger.warn(warning_message)
            filename.unlink()
        else:
            warning_message = f"Warning: file {filename} already exists. Pass -o to overwrite. Terminating..."
            logger.warn(warning_message)
            raise FileExistsError("File already exists.")


def get_image_eval(dataset, indices, return_full, length, target_length):
    return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)


def stack_dict(dicts_tuples):
    input_batch = {}
    target_batch = {}
    lt_index = []
    metadata = {}
    for tup in dicts_tuples:
        x, y, lt_index_temp, metadata_temp = tup
        for key in x.keys():
            if key not in input_batch:
                input_batch[key] = x[key].unsqueeze(0)
            else:
                input_batch[key] = torch.cat((input_batch[key], x[key].unsqueeze(0)), dim=0)
        for key in y.keys():
            if key not in target_batch:
                target_batch[key] = y[key].unsqueeze(0)
            else:
                target_batch[key] = torch.cat((target_batch[key], y[key].unsqueeze(0)), dim=0)

        for key in metadata_temp.keys():
            if key not in metadata:
                metadata[key] = [metadata_temp[key]]
            else:
                metadata[key].append(metadata_temp[key])

        lt_index.append(lt_index_temp)

    return (input_batch, target_batch, lt_index, metadata)


# def get_image_eval(dataset, indices, return_full, length, target_length):
#     try:
#         truth = torch.stack([dataset[indices[i]][1] for i in range(len(indices))], dim=0)
#         context = torch.stack([dataset[indices[i]][0] for i in range(len(indices))], dim=0)
#         if return_full:
#             full_motion_field = torch.zeros((len(indices), target_length, 2, truth.shape[-1], truth.shape[-1]))
#             full_intensities = torch.zeros((len(indices), target_length, truth.shape[-1], truth.shape[-1]))
#             for i in range(len(indices)):
#                 motion_field = torch.stack(
#                     [
#                         torch.from_numpy(dataset[indices[i]][2][f"{le-1}->{le}"])
#                         for le in range(length, length + target_length)
#                     ],
#                     dim=0,
#                 )
#                 intensities = torch.stack(
#                     [
#                         torch.from_numpy(dataset[indices[i]][3][f"{li-1}->{li}"])
#                         for li in range(length, length + target_length)
#                     ],
#                     dim=0,
#                 )
#                 full_motion_field[i] = motion_field
#                 full_intensities[i] = intensities
#         else:
#             full_motion_field = torch.zeros(1)
#             full_intensities = torch.zeros(1)
#     except TypeError:
#         # This happens if the inputs are dicts
#         input_concat = []
#         target_image = []
#         for i in range(len(indices)):
#             input_concat.append(
#                 torch.cat([torch.from_numpy(arr).float() for arr in dataset[indices[i]][0].values()], dim=0)
#             )
#             target_image.append(torch.from_numpy(dataset[indices[i]][1]["goes16_rrqpe"]).float())
#         truth = torch.stack(target_image, dim=0)
#         context = torch.stack(input_concat, dim=0)
#         if return_full:
#             raise NotImplementedError("Full motion field and intensities not implemented for dict inputs.")
#         else:
#             full_motion_field = torch.zeros(1)
#             full_intensities = torch.zeros(1)
#     return truth, context, full_motion_field, full_intensities
