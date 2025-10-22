import pathlib

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.general_dataset import GeneralDataset
from src.utils.data_utils import get_transform_params_filename
from src.utils.dataframe_utils import fetch_future_datetimes
from src.utils.general_utils import get_logger
from src.utils.model_utils import overwrite_file
from src.utils.predict_utils import (
    array_to_pred_hdf_chunked,
    load_autoencoder_kl_model,
    model_to_pred_hdf_chunked,
    obtain_model,
)
from src.utils.train_utils import get_transforms


def main(args):
    data_dict = yaml.safe_load(
        pathlib.Path(f"configs/data/{args.data_config}.yaml").read_text(),
    )
    splits = ["train", "val", "test"]
    fold_dict = dict(
        [(k, v) for k, v in data_dict.items() if k not in splits] + [(k, v) for k, v in data_dict[args.fold].items()]
    )

    targets = list(data_dict["target"].keys())
    train_datetimes_file = data_dict["train"]["datetimes"]
    if isinstance(train_datetimes_file, str):
        train_datetimes_file = [train_datetimes_file]

    locations = data_dict["train"]["locations"]
    params_for_transform = {}
    for target in targets:
        params_for_transform[target] = {}
        for i, file in enumerate(train_datetimes_file):
            params_for_transform_target = yaml.safe_load(
                pathlib.Path(
                    get_transform_params_filename(
                        target,
                        file,
                    )
                ).read_text(),
            )
            params_for_transform[target][locations[i]] = params_for_transform_target[locations[i]]

    transforms, inv_transforms = get_transforms(data_dict, ["train", "val"], params_for_transform, targets)

    fold_dict = dict(
        [(k, v) for k, v in data_dict.items() if k not in splits] + [(k, v) for k, v in data_dict[args.fold].items()]
    )
    test_dataset = GeneralDataset({**fold_dict, "split": args.fold, "config": args.data_config}, return_metadata=True)

    assert test_dataset.nlt == 1

    datetimes = test_dataset.get_datetimes()
    locations = test_dataset.get_locations()

    # load configs
    h_params_init = yaml.safe_load(
        pathlib.Path(f"configs/models/{args.hparams_config}.yaml").read_text(),
    )

    # assert the leadtime are both in the dataset as well as in the parameters
    leadtime_length = test_dataset.leadtime_length

    hparams_leadtime_length = h_params_init.get("leadtime_length", leadtime_length)
    assert leadtime_length == hparams_leadtime_length, "Leadtime length in data config and hparams config must match."
    # load data
    input_shape_dict = dict([(k, v.shape) for k, v in test_dataset[0][0].items()])
    target_shape_dict = dict([(k, v.shape) for k, v in test_dataset[0][1].items()])

    torch.set_float32_matmul_precision("medium")

    # load data
    data_name = args.data_config
    data_target = data_dict["target"]

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size if args.batch_size is not None else h_params_init["batch_size"],
        num_workers=args.num_workers,
        # sampler=sampler,
    )
    # obtain model
    model, model_path = obtain_model(
        h_params_init,
        args.ckpt_file,
        data_name,
        input_shape_dict,
        target_shape_dict,
        transforms,
        inv_transforms,
        data_target,
        is_averaged=args.averaged_model,
        type_of_latent_pred=args.type_of_latent_pred,
        args=args,
    )
    # compute and save predictions
    latent = "_latent" if args.autoencoder_kl_save_latent_predictions is not None else ""
    mean_logvar = f"_{args.type_of_latent_pred}" if args.type_of_latent_pred is not None else ""
    if args.autoencoder_ckpt_file is not None:
        latent += f"_autoencoder_kl_{args.autoencoder_ckpt_file.replace('.ckpt', '')}"
    output_path = f"models/{model_path}/predictions/{data_name}{latent}{mean_logvar}"

    if args.ckpt_file is not None:
        file_name = f"/predict_{args.ckpt_file.replace('.ckpt', '')}_{args.fold}.hdf"
    else:
        file_name = f"/predict_{args.fold}.hdf"
    output_filepaths = dict([(location, f"{output_path}/{location}{file_name}") for location in test_dataset.locations])

    for file in output_filepaths.values():
        pathlib.Path(file).parent.mkdir(parents=True, exist_ok=True)

    # define logger
    logger = get_logger(
        name="predict_logger",
        save_dir=output_path,
        distributed_rank=0,
        filename="log.log",
        resume=True,
    )

    overwrite_file(output_filepaths, args.overwrite, logger)

    # Define metrics output path
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
        output_metrics_filepath = pathlib.Path(f"{output_path_metrics}/metrics_{args.fold}.csv")
        output_metrics_filepath_agg = pathlib.Path(f"{output_path_metrics}/metrics_{args.fold}_agg.csv")
    overwrite_file(output_metrics_filepath, args.overwrite, logger)
    overwrite_file(output_metrics_filepath_agg, args.overwrite, logger)

    if args.save_prediction_options == 1:
        lags = data_dict["target"][targets[0]]["lags"]
        future_datetimes = fetch_future_datetimes(datetimes, lags)
        # Create file

        if args.autoencoder_kl_save_latent_predictions is not None:
            print("Autoencoder KL model is specified, encoding predictions...")
            model_autoencoder_kl, transforms_auto = load_autoencoder_kl_model(
                args,
                input_shape_dict,
                inv_transforms,
                data_target,
                params_for_transform,
                targets,
                locations,
            )
        else:
            model_autoencoder_kl = None
            transforms_auto = None
        model_to_pred_hdf_chunked(
            test_dataloader,
            model,
            datetimes,
            locations,
            future_datetimes,
            output_filepaths,
            args,
            transforms_auto,
            model_autoencoder_kl,
        )
        ok_message = "OK: Saved predictions successfully."
        logger.info(ok_message)

    elif args.save_prediction_options == 2:
        predictions = pl.Trainer(
            devices=args.devices, logger=False, enable_checkpointing=False, inference_mode=True
        ).predict(model, test_dataloader)
        if args.autoencoder_kl_save_latent_predictions is not None:
            print("Autoencoder KL model is specified, encoding predictions...")

            model_autoencoder_kl, transforms_auto = load_autoencoder_kl_model(
                args,
                input_shape_dict,
                inv_transforms,
                data_target,
                params_for_transform,
                targets,
                locations,
            )

            print("Encoding predictions with autoencoder KL model...")
            # transforms = transforms["goes16_rrqpe"]

            for i in tqdm(range(len(predictions))):
                # pred is a single tensor
                t = predictions[i].shape[1]
                pred_rearranged = rearrange(transforms_auto(predictions[i]), "b t h w -> (b t) h w")
                # encode the prediction
                predictions[i] = rearrange(
                    model_autoencoder_kl.encode_stage(pred_rearranged[:, None].to(args.devices[0])).cpu(),
                    "(b t) c h w -> b t c h w",
                    t=t,
                )

        # predictions is a list of lists, each containing tensor
        # predictions = torch.cat(predictions, axis=0)
        # save predictions

        lags = data_dict["target"][targets[0]]["lags"]
        future_datetimes = fetch_future_datetimes(datetimes, lags)
        array_to_pred_hdf_chunked(
            predictions,
            datetimes,
            locations,
            future_datetimes,
            output_filepaths,
        )
        ok_message = "OK: Saved predictions successfully."
        logger.info(ok_message)

    else:
        pl.Trainer(devices=args.devices, logger=False, enable_checkpointing=False, inference_mode=True).predict(
            model, test_dataloader, return_predictions=False
        )

    # Compute metrics
    metrics_agg, ssim_agg = model.eval_metrics_agg.compute()
    metrics_lag, ssim_lag = model.eval_metrics_lag.compute()

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
        "--save_prediction_options",
        "-spo",
        default=0,
        help="Options for saving predictions: 0 - no save, 1 - save predictions while predicting, 2 - save predictions.",
        type=int,
    )
    parser.add_argument(
        "--autoencoder_kl_save_latent_predictions",
        "-slp",
        default=None,
        help="If true, save the latent of the prediction of any model.",
        type=str,
    )
    parser.add_argument(
        "--autoencoder_ckpt_file",
        "-ackpt",
        default=None,
        help="ckpt of the autoencoder model to use for latent predictions.",
        type=str,
    )
    parser.add_argument(
        "--autoencoder_kl_dataconfig",
        "-adconf",
        default=None,
        help="Name of .yaml with data configurations for the autoencoder KL model (optional)",
        type=str,
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
        "--devices",
        "-dv",
        default=1,
        help="GPUs to use",
        nargs="+",
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
        default="val",
        help="Fold to make predictions",
        type=str,
    )
    parser.add_argument(
        "--averaged_model",
        "-am",
        action="store_true",
        help="If true, model is an averaged model.",
    )
    parser.add_argument(
        "--type_of_latent_pred",
        "-tl",
        type=str,
        default=None,
        choices=["mean_logvar", "sample"],
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        default=None,
        type=int,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()
    main(args)
