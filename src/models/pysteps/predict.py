import pathlib
from functools import partial

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from pysteps import motion, nowcasts
from pysteps.utils import conversion, transformation
from tqdm import tqdm

from src.data.general_dataset import GeneralDataset
from src.eval.metrics.metrics import SEVIRSkillScore, is_dist_avail_and_initialized
from src.utils.data_utils import get_transform_params_filename
from src.utils.general_utils import get_logger
from src.utils.model_utils import get_model_path, overwrite_file
from src.utils.train_utils import get_transforms


def r_to_z(value: int, a_z: float = 223.0, b_z: float = 1.56):
    """Convert from rain rate to reflectivity using Z-R relation

    Args:
        value: value to be converted
        a_z: linear coefficient of Z-R relation
        b_z: exponent of Z-R relation

    Returns: Float of converted value
    """

    return a_z * (10 ** (value / 10)) ** b_z


def make_pred(
    it: int, ds: GeneralDataset, adv_scheme: str, motion_f: str, inv_transform: dict, transform: dict
) -> np.array:
    """Make predictions for reflectivity using pySTEPS

    Args:
        it: int with entry of dataset to be used (set so function can run in parallel)
        ds: GeneralDataset object with data
        adv_scheme: string with advection scheme name accepted by pySTEPS
        motion_f: string with motion field method name accepted by pySTEPS
        data_type: string discerning radar from satellite observations

    Returns:
        Array with predictions
    """
    # breakpoint()
    X_dict, y_dict = ds[it][:2]
    metadata_location = ds[it][3]
    input_dataset = list(X_dict.keys())[0]
    X = X_dict[input_dataset]
    y = y_dict[list(y_dict.keys())[0]]
    locations = metadata_location["location"]

    n_locations = len(set(locations)) if isinstance(locations, list) else 1
    if n_locations > 1:
        raise ValueError(
            "Multiple locations not supported for Pysteps. Predict each location separately.")
    else:
        location = locations[0] if isinstance(locations, list) else locations
    # transformation_data = transform[list(y_dict.keys())[0]]
    inv_transformation_data = inv_transform[list(y_dict.keys())[0]]
    # zerovalue = transformation_data[location](torch.Tensor([0])).item()
    metadata = {"transform": None, "zerovalue": -15.0, "threshold": 0.1}

    # Define variables for conversion to precipitation
    zero = metadata["zerovalue"]
    thr = metadata["threshold"]
    a_z = 223.0
    b_z = 1.53

    # Transform data
    Xt = np.array(X)
    yt = np.array(y)

    # Set motion field estimator
    oflow_method = motion.get_method(motion_f)
    # Set advection scheme
    extrapolate = nowcasts.get_method(adv_scheme)
    n_leadtimes = yt.shape[0]
    if adv_scheme == "steps":
        metadata["unit"] = "mm/h"
        if "goes16_rrqpe" in input_dataset:
            km = 2
            t = 10
        elif "imerg" in input_dataset:
            km = 10
            t = 30
        # breakpoint()

        # Transform data to rain rate for correct use of STEPS method
        # Use Z-R relation
        # Note that not much is changed if unit is set to mm/h
        train_precip_pre = conversion.to_rainrate(
            Xt, metadata, zr_a=a_z, zr_b=b_z)[0]
        train_precip_pre[torch.isclose(torch.tensor(
            train_precip_pre), torch.tensor(0.0), atol=1e-04)] = 0.0

        # Change to dB
        train_precip = transformation.dB_transform(
            train_precip_pre, metadata)[0]

        # Set zerovalue
        train_precip[~np.isfinite(train_precip)] = zero

        # Predict motion field
        # breakpoint()
        motion_field = oflow_method(train_precip)

        # Apply advection/extrapolation
        # Predict array of nan when there are no previous times to use in computation
        try:
            # needs to be updated to get km and time from data
            precip_forecast_ens = extrapolate(
                train_precip,
                motion_field,
                n_leadtimes,
                n_ens_members=16,
                n_cascade_levels=8,
                precip_thr=10 * np.log10(thr),
                kmperpixel=km,
                timestep=t,
            )

            # Compute mean value from ensemble
            precip_forecast_mean = torch.nanmean(
                torch.from_numpy(precip_forecast_ens), dim=0)

            # Undo transformations made for STEPS model
            precip_forecast = 10 ** (precip_forecast_mean / 10)
        except ValueError:
            precip_forecast = torch.ones(yt.shape) * np.nan
        except np.linalg.LinAlgError:
            precip_forecast = torch.ones(yt.shape) * np.nan
        except RuntimeError:
            print("RuntimeError: No previous times to use in STEPS computation.")
            precip_forecast = torch.ones(yt.shape) * np.nan

    else:
        # Predict motion field
        motion_field = oflow_method(Xt)

        last_observation = Xt[-1]

        # Extrapolate
        last_observation[~np.isfinite(last_observation)] = 0
        precip_forecast = extrapolate(Xt[-1], motion_field, n_leadtimes)

    # Undo transformations on original data
    precip_forecast = torch.Tensor(precip_forecast)
    yt = torch.Tensor(yt)
    if n_locations == 1:
        location = locations[0] if isinstance(locations, list) else locations
        precip_forecast = inv_transformation_data[location](precip_forecast)
        yt = inv_transformation_data[location](yt)
    else:
        raise ValueError(
            "Multiple locations not supported for Pysteps. Predict each location separately.")
        # precip_forecast = transform_multiple_loc(inv_transformation_data, precip_forecast, locations)
        # yt = transform_multiple_loc(inv_transformation_data, yt, locations)

    return precip_forecast, yt


def main(args):
    data_dict = yaml.safe_load(
        pathlib.Path(f"configs/data/{args.data_config}.yaml").read_text(),
    )
    # Assert target is of the same format as input and that there is only one of each
    assert len(list(data_dict["target"].keys())
               ) == 1, "Ony one input may be passed."
    assert len(list(data_dict["input"].keys())
               ) == 1, "Ony one target may be passed."
    assert (
        list(data_dict["target"].keys())[0] == list(
            data_dict["input"].keys())[0]
    ), "Input dataset and target dataset must be the same."

    splits = ["train", "val", "test"]
    fold_dict = dict(
        [(k, v) for k, v in data_dict.items() if k not in splits] + [(k, v)
                                                                     for k, v in data_dict[args.fold].items()]
    )
    test_dataset = GeneralDataset(
        {**fold_dict, "split": args.fold, "config": args.data_config}, return_metadata=True)
    targets = list(data_dict["target"].keys())
    if not len(targets) == 1:
        raise Exception("Prediction must be done for one dataset.")
    # breakpoint()
    train_datetimes_file = data_dict["train"]["datetimes"]

    params_for_transform = {}
    for target in targets:
        params_for_transform_target = yaml.safe_load(
            pathlib.Path(
                get_transform_params_filename(
                    target,
                    train_datetimes_file,
                )
            ).read_text(),
        )
        params_for_transform[target] = params_for_transform_target

    h_params = yaml.safe_load(
        pathlib.Path(f"configs/models/{args.hparams_config}.yaml").read_text(),
    )

    model_name = h_params["model_name"]
    data_name = args.data_config
    model_path = get_model_path(
        h_params,
        model_name,
        data_name,
    )[0]
    print(f"Model path: {model_path}")
    eval_metrics_lag = SEVIRSkillScore(
        layout="NTCHW",
        seq_len=test_dataset.target_length,
        preprocess_type="identity",
        mode="1",
        dist_eval=True if is_dist_avail_and_initialized() else False,
    )

    eval_metrics_agg = SEVIRSkillScore(
        layout="NTCHW",
        seq_len=test_dataset.target_length,
        preprocess_type="identity",
        mode="0",
        dist_eval=True if is_dist_avail_and_initialized() else False,
    )
    step = args.num_workers
    size = len(test_dataset)
    # Get indices to slice ds into chuncks
    chunk_indices = np.arange(0, size, step)
    if chunk_indices[-1] != size:
        chunk_indices = np.append(chunk_indices, [size])
    transform, inv_transform = get_transforms(
        data_dict, [args.fold], params_for_transform, targets)
    # Fix second entry of make_pred function to use parallelism
    task = partial(
        make_pred,
        ds=test_dataset,
        adv_scheme=h_params["advection"],
        motion_f=h_params["motion_field"],
        inv_transform=inv_transform,
        transform=transform,
    )
    # Run predictions for each chunk
    # predictions_full = []
    # breakpoint()
    for i, j in tqdm(zip(chunk_indices, chunk_indices[1:]), total=len(chunk_indices) - 1):
        chunk_iterable = range(i, j)
        # breakpoint()
        full_list = Parallel(n_jobs=step, backend="threading")(
            delayed(task)(it) for it in chunk_iterable)
        predicts_list = [item[0] for item in full_list]
        target_list = [item[1] for item in full_list]

        # Concatenate predictions
        predict = torch.from_numpy(np.stack(predicts_list, axis=0))
        target = torch.from_numpy(np.stack(target_list, axis=0))

        # predictions_full.append(predict)
        eval_metrics_lag.update(
            target=target[:, :, None],
            pred=predict[:, :, None],
        )
        eval_metrics_agg.update(
            target=target[:, :, None],
            pred=predict[:, :, None],
            metadata=None,
        )
        # if args.save_all_predictions:
        #     array_to_pred_hdf(predict, ds.keys[i:j], ds.future_keys[i:j], output_predict_filepath)

    # compute and save predictions
    output_path = f"models/{model_path}/predictions/{data_name}"
    output_model_filepath = pathlib.Path(
        f"{output_path}/predict_{args.fold}.npy")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # define logger
    logger = get_logger(
        name="predict_logger",
        save_dir=output_path,
        distributed_rank=0,
        filename="log.log",
        resume=True,
    )

    overwrite_file(output_model_filepath, args.overwrite, logger)

    # Save the metrics
    output_path_metrics = f"eval/{model_path}/{args.data_config}"
    pathlib.Path(output_path_metrics).mkdir(parents=True, exist_ok=True)

    # define logger
    logger = get_logger(
        name="metric_logger",
        save_dir=output_path_metrics,
        distributed_rank=0,
        filename="log.log",
        resume=True,
    )

    if args.ckpt_file is not None:
        output_metrics_filepath = pathlib.Path(
            f"{output_path_metrics}/metrics_{args.ckpt_file.replace('.ckpt', '')}_{args.fold}.csv"
        )
        output_metrics_filepath_agg = pathlib.Path(
            f"{output_path_metrics}/metrics_{args.ckpt_file.replace('.ckpt', '')}_{args.fold}_agg.csv"
        )
    else:
        output_metrics_filepath = pathlib.Path(
            f"{output_path_metrics}/metrics_{args.fold}.csv")
        output_metrics_filepath_agg = pathlib.Path(
            f"{output_path_metrics}/metrics_{args.fold}_agg.csv")
    overwrite_file(output_metrics_filepath, args.overwrite, logger)
    overwrite_file(output_metrics_filepath_agg, args.overwrite, logger)

    # save predictions
    # if args.save_all_predictions:
    #     predictions = torch.cat(predictions_full, dim=0)
    #     np.save(output_model_filepath, predictions.numpy())
    #     ok_message = "OK: Saved predictions successfully."
    #     logger.info(ok_message)

    # metrics = model.eval_metrics.compute()
    metrics_agg, ssim_agg = eval_metrics_agg.compute()
    metrics_lag, ssim_lag = eval_metrics_lag.compute()
    rows_metrics = []
    rows_metrics_agg = []
    for threshold, values in metrics_lag.items():
        for type_metric, value in values.items():
            row = pd.DataFrame(value, columns=[f"{type_metric}_{threshold}"])
            rows_metrics.append(row)

    row = pd.DataFrame(ssim_lag.cpu().numpy(), columns=["ssim"])
    rows_metrics.append(row)

    for threshold, values in metrics_agg.items():
        for type_metric, value in values.items():
            row = pd.DataFrame([value], columns=[f"{type_metric}_{threshold}"])
            rows_metrics_agg.append(row)

    row = pd.DataFrame([ssim_agg.cpu().numpy()], columns=["ssim"])
    rows_metrics_agg.append(row)

    metrics_dataframe = pd.concat(rows_metrics, axis=1)
    metrics_dataframe_agg = pd.concat(rows_metrics_agg, axis=1)

    # Save metrics
    if not output_metrics_filepath.exists() or args.overwrite:
        metrics_dataframe.to_csv(output_metrics_filepath, index=False)
        ok_message = "OK: Saved metrics successfully."
        logger.info(ok_message)

    if not output_metrics_filepath_agg.exists() or args.overwrite:
        metrics_dataframe_agg.to_csv(output_metrics_filepath_agg, index=False)
        ok_message = "OK: Saved aggregated metrics successfully."
        logger.info(ok_message)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="If true, overwrites output; otherwise, skips existing files.",
    )
    parser.add_argument(
        "--save_all_predictions",
        "-sap",
        action="store_true",
        help="If true, save the predicitions.",
    )
    parser.add_argument(
        "--data_config",
        "-dconf",
        default="sevir",
        type=str,
        help="Name of .yaml with data configurations",
    )
    parser.add_argument(
        "--hparams_config",
        "-hconf",
        default="unet",
        type=str,
        help="Name of .yaml with model configurations (optional)",
    )
    parser.add_argument(
        "--num_workers",
        "-nw",
        default=20,
        help="Number of jobs for parallelization",
        type=int,
    )
    parser.add_argument(
        "--ckpt_file",
        "-ckpt",
        default=None,
        help="ckpt.",
        type=str,
    )
    parser.add_argument(
        "--fold",
        "-df",
        default="test",
        help="Fold to train.",
        type=str,
    )
    args = parser.parse_args()
    main(args)
