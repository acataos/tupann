import argparse
import datetime
import pathlib

import h5py
import numpy as np
import torch
from pysteps import motion
from pysteps.utils import conversion, transformation
from tqdm import tqdm

from src.data.dataset_handlers import DatasetHandlerFactory

DT_FORMAT = "%Y-%m-%d %H:%M:%S"

CONTEXT_DICT = {
    "lk": 10,
    "darts": 10,
}
motion_field_methods = ["lk", "darts"]

dataset_dict = {
    "goes16_rrqpe": {
        "timestep": 10,
    },
    "imerg": {
        "timestep": 30,
    },
}


def compute_motion_field(Xt, method="lk"):
    metadata = {"transform": None, "zerovalue": -15.0, "threshold": 0.1}
    zero = metadata["zerovalue"]
    a_z = 223.0
    b_z = 1.53
    metadata["unit"] = "mm/h"
    motion_field = np.zeros((2, Xt.shape[-1], Xt.shape[-1]))
    Xt = Xt.cpu().numpy()[:, :]
    train_precip_pre = conversion.to_rainrate(Xt, metadata, zr_a=a_z, zr_b=b_z)[0]
    train_precip = transformation.dB_transform(train_precip_pre, metadata)[0]
    train_precip[~np.isfinite(train_precip)] = zero

    oflow_method = motion.get_method(method)
    motion_field = oflow_method(train_precip)
    return motion_field


def make_grid(input):
    B, C, H, W = input.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    return grid


def warp(input, flow, grid, mode="bilinear", padding_mode="zeros", fill_value=0.0):
    B, C, H, W = input.size()
    vgrid = grid - flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = (
        torch.nn.functional.grid_sample(
            input - fill_value,
            vgrid,
            padding_mode=padding_mode,
            mode=mode,
            align_corners=True,
        )
        + fill_value
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="Save motion fields and intensities for a dataset and datetimes.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--datetimes_file", type=str, required=True, help="List of datetimes")
    parser.add_argument("--locations", nargs="+", help="Locations to process", default=["rj"])
    parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    sample_tensor = torch.zeros(1, 1, 256, 256)
    grid = make_grid(sample_tensor)
    dataset = args.dataset
    datetimes_file = args.datetimes_file
    # read datetimes from file
    with open(datetimes_file, "r") as f:
        datetimes = [datetime.datetime.strptime(line.strip(), DT_FORMAT) for line in f.readlines()]

    dataset_handler = DatasetHandlerFactory.create_handler(dataset, args.locations)
    for motion_field_method in motion_field_methods:
        for location in args.locations:
            fields_intensities_hdf_file = pathlib.Path(
                f"data/fields_intensities_rain_events/{motion_field_method}/{args.dataset}_{location}.hdf"
            )
            fields_intensities_hdf_file.parent.mkdir(parents=True, exist_ok=True)
            # check if file exists
            with h5py.File(fields_intensities_hdf_file, "a") as hdf:
                min_val = 0
                for dt in tqdm(datetimes):
                    dts = []
                    try:
                        _ = np.array(hdf[f"motion_fields/{dt.strftime('%Y%m%d/%H%M')}"])
                        _ = np.array(hdf[f"intensities/{dt.strftime('%Y%m%d/%H%M')}"])
                        continue
                    except KeyError:
                        pass
                    for i in range(-CONTEXT_DICT[motion_field_method] + 1, 2):
                        dts.append(dt + datetime.timedelta(minutes=i * dataset_dict[dataset]["timestep"]))
                    tensor = torch.tensor(dataset_handler.fetch(dts, location))
                    print(tensor.max())
                    tensor = torch.nan_to_num(tensor)
                    tensor = tensor.float()
                    # compute motion field
                    motion_field = np.array(compute_motion_field(tensor, motion_field_method)).astype(np.float32)
                    # compute intensities
                    cuda_motion_field = torch.tensor(motion_field).float().unsqueeze(0).cuda()
                    X0 = tensor[-2].unsqueeze(0).unsqueeze(0).cuda()
                    X1 = tensor[-1].cuda()
                    pred_image = warp(
                        X0,
                        cuda_motion_field,
                        grid,
                        padding_mode="zeros",
                        fill_value=min_val,
                    )
                    intensities = X1 - pred_image
                    intensities = intensities.cpu().numpy().astype(np.float32)

                    # save both
                    try:
                        hdf.create_group("motion_fields")
                        hdf.create_group("intensities")
                    except ValueError:
                        pass
                    hdf.create_dataset(
                        f"motion_fields/{dt.strftime('%Y%m%d/%H%M')}",
                        data=motion_field,
                        compression="lzf",
                    )
                    hdf.create_dataset(
                        f"intensities/{dt.strftime('%Y%m%d/%H%M')}",
                        data=intensities,
                        compression="lzf",
                    )


if __name__ == "__main__":
    main()
