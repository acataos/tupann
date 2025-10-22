import hashlib

import yaml


def center_crop(tensor, hf=48, wf=48):
    """
    Crop a tensor from (h,w) to (hf,wf) by taking the center square.

    Args:
        tensor: Input tensor with shape (..., h, w)

    Returns:
        Cropped tensor with shape (..., hf, wf)
    """
    # Get the original shape
    original_shape = tensor.shape

    # Calculate the start and end indices for cropping
    h, w = original_shape[-2], original_shape[-1]

    start_h = (h - hf) // 2
    start_w = (w - wf) // 2

    # Crop the tensor
    return tensor[..., start_h : start_h + hf, start_w : start_w + wf]


def calc_dict_hash(d):
    yaml_str = yaml.dump(d, default_flow_style=False)
    hash = hashlib.md5(yaml_str.encode()).hexdigest()

    return hash


def get_transform_params_filename(dataset_name, datetimes_file):
    datetimes_file = datetimes_file.replace(".txt", "").replace("configs/data/", "")
    return f"data/transform_params/{dataset_name}.{datetimes_file}.yaml"
