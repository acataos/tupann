import importlib
import pathlib
import warnings
from itertools import product

import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.general_dataset import GeneralDataset
from src.utils.data_utils import get_transform_params_filename
from src.utils.model_utils import calc_ckpt_code, get_model_path, lock, overwrite_file, stack_dict, unlock
from src.utils.train_utils import get_transforms


def main(args):
    # load data config for each split
    data_dict = yaml.safe_load(
        pathlib.Path(f"configs/data/{args.data_config}.yaml").read_text(),
    )
    splits = ["train", "val", "test"]
    dicts_per_split = {
        split: dict(
            [(k, v) for k, v in data_dict.items() if k not in splits] + [(k, v) for k, v in data_dict[split].items()]
        )
        for split in splits
    }
    datasets = {
        split: GeneralDataset(
            {**dicts_per_split[split], "config": args.data_config, "split": split}, return_metadata=True
        )
        for split in splits
    }
    # load configs
    eval_dict = yaml.safe_load(
        pathlib.Path(f"configs/eval/{args.eval_config}.yaml").read_text(),
    )
    h_params_init = yaml.safe_load(
        pathlib.Path(f"configs/models/{args.hparams_config}.yaml").read_text(),
    )

    # assert the leadtime are both in the dataset as well as in the parameters
    leadtime_length = datasets["train"].leadtime_length

    hparams_leadtime_length = h_params_init.get("leadtime_length", leadtime_length)
    assert leadtime_length == hparams_leadtime_length, "Leadtime length in data config and hparams config must match."
    # load data
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    input_shape_dict = dict([(k, v.shape) for k, v in train_dataset[0][0].items()])
    target_shape_dict = dict([(k, v.shape) for k, v in train_dataset[0][1].items()])
    target_shape_dict_val = dict([(k, v.shape) for k, v in val_dataset[0][1].items()])
    weights = train_dataset.get_sample_weights()
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    if leadtime_length < datasets["train"].target_length:
        dummy_data_dict = dicts_per_split["train"].copy()
        dummy_data_dict["leadtime_length"] = datasets["train"].target_length
        dummy_train = GeneralDataset(
            {**dummy_data_dict, "split": "train", "config": args.data_config}, return_metadata=True
        )
    else:
        dummy_train = train_dataset

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=h_params_init["batch_size"],
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=h_params_init["batch_size"],
        num_workers=args.num_workers,
        pin_memory=True,
    )

    list_params = []
    for param, value in h_params_init.items():
        if type(value) is list:
            list_params.append(param)
    h_params_list = {key: h_params_init[key] for key in list_params}
    non_list_params = set(h_params_init.keys()) - set(list_params)
    h_params_prod = [dict(zip(h_params_list.keys(), values)) for values in product(*h_params_list.values())]
    for h_params in h_params_prod:
        for param in non_list_params:
            h_params[param] = h_params_init[param]

        # define model
        model_name = h_params["model_name"]
        model_location = f"src.models.{model_name}.lightning"
        try:
            model_class = importlib.import_module(model_location).model
        except ModuleNotFoundError:
            main_model, submodel = model_name.split("_")
            warnings.warn(f"Training submodel {submodel} from {main_model}")
            model_location = f"src.models.{main_model}.{submodel}_lightning"
            model_class = importlib.import_module(model_location).model

        # TODO: Update this to use the new get_image_eval function
        batch_train = stack_dict(
            [dummy_train[idx] for idx in eval_dict["train_idx"]],
        )
        batch_val = stack_dict(
            [val_dataset[idx] for idx in eval_dict["val_idx"]],
        )

        train_datetimes_file = data_dict["train"]["datetimes"]
        targets = list(data_dict["target"].keys())
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

        model = model_class(
            input_shape_dict=input_shape_dict,
            target_shape_dict=target_shape_dict,
            target_shape_dict_val=target_shape_dict_val,
            batch_train=batch_train,
            batch_val=batch_val,
            target_is_imerg=("imerg" in data_dict["target"]),
            transforms=transforms,
            inv_transforms=inv_transforms,
            **h_params,
        )

        model_path, output_path, logger = get_model_path(
            h_params,
            model_name,
            args.data_config,
        )

        ckpt_codes_filepath = pathlib.Path(output_path + "/ckpt_codes.csv")

        if args.ckpt_file is not None:
            old_ckpt_code = args.ckpt_file.split("--")[0]
            epoch = int(args.ckpt_file.split("--")[1].split("=")[1].replace(".ckpt", ""))
            new_ckpt_code = calc_ckpt_code(old_ckpt_code, ckpt_codes_filepath)
            output_model_filepath = pathlib.Path(
                f"{output_path}/model_train-ckpt={args.ckpt_file.replace('.ckpt', '')}" + ".pt"
            )
            ckpt_filepath = pathlib.Path(f"{output_path}/{args.ckpt_file}")
        else:
            new_ckpt_code = calc_ckpt_code("", ckpt_codes_filepath)
            epoch = 0
            ckpt_filepath = None
            output_model_filepath = pathlib.Path(f"{output_path}/model_train" + ".pt")

        output_parameters_filepath = pathlib.Path(f"{output_path}/h_params.yaml")

        overwrite_file(output_model_filepath, args.overwrite, logger)
        # overwrite_file(output_parameters_filepath, args.overwrite, logger)

        # define tensorboard logger and model checkpoint
        torch.set_float32_matmul_precision("medium")
        if args.tensorboard_logger_dir is not None:
            logger_path = f"experiments/{args.tensorboard_logger_dir}"
            tensorboard_logger = pl.loggers.TensorBoardLogger(
                logger_path,
                model_path.replace("/", "-"),
            )
        else:
            logger_path = f"models/{model_path}/logs/{args.data_config}"
            tensorboard_logger = pl.loggers.TensorBoardLogger(
                logger_path,
                None,
            )
        pathlib.Path(logger_path).mkdir(parents=True, exist_ok=True)
        checkpoint_save_best = pl.callbacks.ModelCheckpoint(
            dirpath=output_model_filepath.parents[0],
            save_top_k=eval_dict["save_top"],
            monitor=eval_dict["monitored_loss"],
            filename=f"{new_ckpt_code}--" + "best-{epoch}",
            mode="max" if eval_dict["monitored_loss"][:3] == "CSI" else "min",
        )
        # train model
        trainer = pl.Trainer(
            strategy="ddp_find_unused_parameters_true",
            profiler="simple",
            max_epochs=h_params["n_epochs"] + epoch,
            accelerator="gpu",
            logger=tensorboard_logger,
            devices=args.devices,
            callbacks=[
                checkpoint_save_best,
            ],
            # val_check_interval=0.00001,
            # fast_dev_run=True
        )

        if ckpt_filepath is not None:
            model_state = torch.load(ckpt_filepath, weights_only=False)
            model.load_state_dict(model_state["state_dict"])
            print(f"Resuming from checkpoint {ckpt_filepath}, epoch {epoch}")

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # save model
        best_k_models_keys = checkpoint_save_best.best_k_models.keys()
        best_k_models_filenames = [k.split("/")[-1] for k in best_k_models_keys]
        best_k_model_vals = [checkpoint_save_best.best_k_models[k] for k in best_k_models_keys]
        ckpt_filepath = pathlib.Path(f"{output_path}/{args.ckpt_file}")
        ckpt_tree_filepath = pathlib.Path(output_path + "/ckpt_tree.jsonl")

        lock(ckpt_tree_filepath.with_suffix(".tmp"), logger)
        with open(ckpt_tree_filepath, "a") as outfile:
            for filename, value in zip(best_k_models_filenames, best_k_model_vals):
                ckpt_dict = {
                    "filename": filename,
                    "loss": value.detach().item(),
                    "parent_ckpt": str(ckpt_filepath).split("/")[-1],
                }
                yaml.dump(ckpt_dict, outfile)
                outfile.write("\n")
        unlock(ckpt_tree_filepath.with_suffix(".tmp"))

        torch.save(model.state_dict(), output_model_filepath)

        ok_message = "OK: Trained and saved model successfully."
        logger.info(ok_message)

        with open(output_parameters_filepath, "w") as outfile:
            yaml.dump(h_params, outfile)

        ok_message = "OK: Saved parameters file successfully."
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
        "--data_config",
        "-dconf",
        default="sevir",
        type=str,
        help="Name of .yaml with data configurations",
    )
    parser.add_argument(
        "--eval_config",
        "-econf",
        default="sevir_eval",
        type=str,
        help="Name of .yaml with evaluation configurations",
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
        "--tensorboard_logger_dir",
        "-ld",
        default=None,
        help="Directory to store logs",
        type=str,
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

    args = parser.parse_args()
    main(args)
