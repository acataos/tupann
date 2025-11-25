import importlib
import pathlib

import h5py
import torch
import yaml
from einops import rearrange
from tqdm import tqdm

from src.data.general_dataset import GeneralDataset
from src.utils.model_utils import get_model_path
from src.utils.train_utils import get_transforms


def array_to_pred_hdf(arr, keys, future_keys, output_filepath):
    with h5py.File(output_filepath, "a") as hdf:
        for i, key in enumerate(keys):
            try:
                key = key.decode("utf-8")
            except AttributeError:
                pass

            for j, future_key in enumerate(future_keys[i]):
                try:
                    future_key = future_key.decode("utf-8")
                except AttributeError:
                    pass
                pred_key = f"{key}/{future_key.replace('/','-')}"
                hdf.create_dataset(pred_key, data=arr[i, j], compression="lzf")


def array_to_pred_hdf_chunked(prediction_chunks, keys, locations, future_keys, output_filepaths):
    """
    Save predictions to HDF5 file, processing list of tensors directly.

    Args:
        prediction_chunks: List of tensors from model.predict()
        keys: Datetimes for each sample
        future_keys: Future datetimes for predictions
        output_filepath: Path to save the HDF file
    """
    # Create file
    hdfs_dict = dict([(location, h5py.File(file, "a"))
                     for (location, file) in output_filepaths.items()])
    # Process each chunk
    sample_idx = 0
    for chunk_idx, chunk in enumerate(prediction_chunks):
        # Get number of samples in this chunk
        chunk_size = chunk.shape[0]

        # Move to CPU and detach
        chunk = chunk.detach().cpu()

        # Get the corresponding keys for this chunk
        chunk_end = min(sample_idx + chunk_size, len(keys))
        chunk_keys = keys[sample_idx:chunk_end]
        chunk_locations = locations[sample_idx:chunk_end]
        chunk_future_keys = future_keys[sample_idx:chunk_end]

        # Process each sample in the chunk
        for i in range(chunk_size):
            if sample_idx + i >= len(keys):
                break  # Avoid index out of bounds

            key = chunk_keys[i]
            location = chunk_locations[i]
            try:
                key = key.decode("utf-8")
            except AttributeError:
                pass

            # Save each future prediction
            for j, future_key in enumerate(chunk_future_keys[i]):
                try:
                    future_key = future_key.decode("utf-8")
                except AttributeError:
                    pass

                # Create dataset path
                pred_key = f"{key}/{future_key.replace('/','-')}"

                # Save this specific prediction
                hdfs_dict[location].create_dataset(
                    pred_key, data=chunk[i, j], compression="lzf")

        # Update sample index
        sample_idx += chunk_size

        # Free memory
        del chunk
        torch.cuda.empty_cache()
    for hdf in hdfs_dict.values():
        hdf.close()


def load_autoencoder_kl_model(
    args, input_shape_dict, inv_transforms, data_target, params_for_transform, targets, locations
):
    # load configs
    autoencoder_params = yaml.safe_load(
        pathlib.Path(
            f"configs/models/{args.autoencoder_kl_save_latent_predictions}.yaml").read_text(),
    )
    autoencoder_data_dict = yaml.safe_load(
        pathlib.Path(
            f"configs/data/{args.autoencoder_kl_dataconfig}.yaml").read_text(),
    )
    transforms_auto_dict, inv_transform_dict = get_transforms(
        autoencoder_data_dict, [args.fold], params_for_transform, targets
    )
    transforms_auto = transforms_auto_dict[list(
        autoencoder_data_dict["target"].keys())[0]][locations[0]]
    inv_transform_auto = inv_transform_dict[list(
        autoencoder_data_dict["target"].keys())[0]][locations[0]]
    splits = ["train", "val", "test"]
    dicts_per_split_auto = {
        split: dict(
            [(k, v) for k, v in autoencoder_data_dict.items() if k not in splits]
            + [(k, v) for k, v in autoencoder_data_dict[split].items()]
        )
        for split in splits
    }
    dummy_train = GeneralDataset(
        {**dicts_per_split_auto["train"], "split": "train",
            "config": args.autoencoder_kl_dataconfig},
        return_metadata=True,
    )
    target_shape_dict_val = dict([(k, v.shape)
                                 for k, v in dummy_train[0][1].items()])

    model_autoencoder_kl, _ = obtain_model(
        autoencoder_params,
        ckpt_file=args.autoencoder_ckpt_file,
        data_name=args.autoencoder_kl_dataconfig,
        input_shape_dict=input_shape_dict,
        target_shape_dict=target_shape_dict_val,
        inv_transforms=inv_transform_auto,
        transforms=transforms_auto,
        data_target=data_target,
        is_averaged=args.averaged_model,
        args=args,
    )
    model_autoencoder_kl.to(args.devices[0])
    model_autoencoder_kl.eval()
    # set the prediction to latent space
    model_autoencoder_kl.predict_latent = True
    model_autoencoder_kl.inv_transforms = inv_transform_auto
    return model_autoencoder_kl, transforms_auto


def model_to_pred_hdf_chunked(
    pred_dataloader,
    model,
    keys,
    locations,
    future_keys,
    output_filepaths,
    args,
    transforms_auto=None,
    model_autoencoder_kl=None,
):
    """
    Save predictions to HDF5 file, processing list of tensors directly.

    Args:
        prediction_chunks: List of tensors from model.predict()
        keys: Datetimes for each sample
        future_keys: Future datetimes for predictions
        output_filepath: Path to save the HDF file
    """
    # Create file
    hdfs_dict = dict([(location, h5py.File(file, "a"))
                     for (location, file) in output_filepaths.items()])
    # Process each chunk
    sample_idx = 0
    model.eval()
    model.to(args.devices[0])
    for i, batch in enumerate(tqdm(pred_dataloader)):
        chunk = model.predict_step(
            batch, i, update_metrics=True, return_full=False)
        # Get number of samples in this chunk
        chunk_size = chunk.shape[0]

        # Get the corresponding keys for this chunk
        chunk_end = min(sample_idx + chunk_size, len(keys))
        chunk_keys = keys[sample_idx:chunk_end]
        chunk_locations = locations[sample_idx:chunk_end]
        chunk_future_keys = future_keys[sample_idx:chunk_end]

        if args.autoencoder_kl_save_latent_predictions is not None:
            # pred is a single tensor
            pred_rearranged = rearrange(
                transforms_auto(chunk), "b t h w -> (b t) h w")
            # encode the prediction
            chunk = rearrange(
                model_autoencoder_kl.encode_stage(
                    pred_rearranged[:, None].to(args.devices[0])).cpu(),
                "(b t) c h w -> b t c h w",
                b=chunk_size,
            )

        # Move to CPU and detach
        chunk = chunk.detach().cpu()
        # Process each sample in the chunk
        for i in range(chunk_size):
            if sample_idx + i >= len(keys):
                break  # Avoid index out of bounds

            key = chunk_keys[i]
            location = chunk_locations[i]
            try:
                key = key.decode("utf-8")
            except AttributeError:
                pass

            # Save each future prediction
            for j, future_key in enumerate(chunk_future_keys[i]):
                try:
                    future_key = future_key.decode("utf-8")
                except AttributeError:
                    pass

                # Create dataset path
                pred_key = f"{key}/{future_key.replace('/','-')}"

                # Save this specific prediction
                hdfs_dict[location].create_dataset(
                    pred_key, data=chunk[i, j], compression="lzf")

        # Update sample index
        sample_idx += chunk_size

        # Free memory
        # del chunk
        # torch.cuda.empty_cache()
    for hdf in hdfs_dict.values():
        hdf.close()


def obtain_model(
    h_params,
    ckpt_file,
    data_name,
    input_shape_dict,
    target_shape_dict,
    transforms,
    inv_transforms,
    data_target,
    is_averaged=False,
    type_of_latent_pred=None,
    args=None,
):
    """
    Obtain the model based on the hyperparameters and configuration.
    Args:
        h_params (dict): Hyperparameters for the model.
        args (argparse.Namespace): Command line arguments.
        data_name (str): Name of the dataset.
        input_shape (tuple): Shape of the input data used in the model.
        target_shape (tuple): Shape of the target data used in the model.
        inv_transforms (callable): Inverse transformations functions.
    Returns:
        model (torch.nn.Module): The loaded model.
        model_path (str): Path to the model.
    """
    # define model
    model_name = h_params["model_name"]
    if model_name == "tupann_autoenc":
        model_location = "src.models.tupann.autoenc_lightning"
    else:
        model_location = f"src.models.{model_name}.lightning"
    model_class = importlib.import_module(model_location).model
    model_path, input_path, _ = get_model_path(
        h_params,
        model_name,
        data_name,
    )

    target_shape_dict_val = target_shape_dict
    if ckpt_file is not None:
        input_model_filepath = pathlib.Path(
            f"{input_path}/{ckpt_file.replace('.ckpt', '')}" + ".ckpt")
    else:
        if is_averaged:
            input_model_filepath = pathlib.Path(
                f"{input_path}/averaged_model" + ".pt")
        else:
            input_model_filepath = pathlib.Path(
                f"{input_path}/model_train" + ".pt")

    print(f"Loading model from {input_model_filepath}")
    if pathlib.Path(input_model_filepath).suffix == ".pt":
        model = model_class(
            input_shape_dict,
            target_shape_dict,
            target_shape_dict_val=target_shape_dict_val,
            transforms=transforms,
            inv_transforms=inv_transforms,
            **h_params,
        )
        model.load_state_dict(torch.load(
            input_model_filepath, map_location=f"cuda:{args.devices[0]}"))
    elif pathlib.Path(input_model_filepath).suffix == ".ckpt":
        if model_name == "cascast":
            checkpoint = torch.load(
                input_model_filepath, map_location=f"cuda:{args.devices[0]}", weights_only=False)
            # need to remove the unnecessary keys in the state dict
            # especially when loading a model trained with autoencoderklgan
            cast_former_params = {k: v for k, v in checkpoint["state_dict"].items(
            ) if not k.startswith("autoencoder.")}
            checkpoint["state_dict"] = cast_former_params
            model = model_class(
                input_shape_dict,
                target_shape_dict,
                target_shape_dict_val=target_shape_dict_val,
                transforms=transforms,
                inv_transforms=inv_transforms,
                **h_params,
            )
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = model_class(
                input_shape_dict=input_shape_dict,
                target_shape_dict=target_shape_dict,
                target_shape_dict_val=target_shape_dict_val,
                transforms=transforms,
                inv_transforms=inv_transforms,
                **h_params,
            )
            model.load_state_dict(
                torch.load(input_model_filepath, weights_only=False,
                           map_location=torch.device("cuda:0"))["state_dict"],
                strict=True,
            )

    model.inv_transforms = inv_transforms
    if h_params["model_name"] == "autoencoderklgan":
        model.predict_latent = True
        print("Setting model to predict in latent space")
    elif h_params["model_name"] == "tupann_autoenc":
        model.predict_latent = type_of_latent_pred
    return model, model_path


""
