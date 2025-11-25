import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from moviepy.video.io.bindings import mplfig_to_npimage


def compare_true_pred(X_true, X_pred, cmap, X_model=None, v_max=1, v_min=0):
    if X_model is None:
        X_norm = [X_true, X_pred]

    else:
        X_norm = [X_true, X_pred, X_model]

    X_norm = torch.stack(X_norm, dim=0)
    X_norm[X_norm > v_max] = v_max
    X_norm[X_norm < v_min] = v_min
    if X_model is None:
        fig, (ax1, ax2) = plt.subplots(1, X_norm.shape[0], figsize=(10, 5))
        ax1.imshow(X_norm[0].detach().cpu(), cmap=cmap, vmin=v_min, vmax=v_max)
        ax2.imshow(X_norm[1].detach().cpu(), cmap=cmap, vmin=v_min, vmax=v_max)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(
            1,
            X_norm.shape[0],
            figsize=(10, 5),
        )
        ax1.imshow(X_norm[0].detach().cpu(), cmap=cmap, vmin=v_min, vmax=v_max)
        ax2.imshow(X_norm[1].detach().cpu(), cmap=cmap, vmin=v_min, vmax=v_max)
        ax3.imshow(X_norm[2].detach().cpu(), cmap=cmap, vmin=v_min, vmax=v_max)

    return fig


def calc_weights(y, xmax, loss_type, weight_type):
    match weight_type:
        case "uniform":
            weights = torch.ones_like(y)
        case "linear_clamped":
            weights = torch.clamp(y + 1, min=1, max=xmax)
        case _:
            raise ValueError("Invalid weight type")
    if loss_type == "l2":
        weights = torch.sqrt(weights)
    return weights


# assume concatenation is on the first axis
def calc_concat_shape_dict(shape_dict: dict):
    dummy_tensors = []
    for k, v in shape_dict.items():
        assert isinstance(
            v, tuple), f"Unsupported type {type(v)} for key {k} in shape_dict"
        dummy_tensors.append(torch.zeros(v))
    dummy_tensor = torch.cat(dummy_tensors, dim=0)
    return dummy_tensor.shape


def plot_fields_intensities(ax, arr):
    assert arr.ndim == 3, "Array must be 3D (C, H, W)"
    fields = arr[0:2]
    intensities = arr[2]
    np.random.seed(42)  # for reproducibility
    arr_size = intensities.shape[-1]
    inds = np.random.choice(arr_size**2, 200, replace=False)
    inds_i = inds // arr_size
    inds_j = inds % arr_size
    ax.quiver(
        inds_i,
        inds_j,
        fields[0, inds_i, inds_j].cpu().detach(),
        fields[1, inds_i, inds_j].cpu().detach(),
        color="red",
    )
    rainbow = mlp.colormaps["rainbow"]
    # Create colormap for radar
    list_color = rainbow(np.linspace(0, 1, 256))
    cmpradar = ListedColormap(list_color)
    v_max = 30.0
    v_min = -30.0
    ax.matshow(intensities.cpu().detach(),
               cmap=cmpradar, vmin=v_min, vmax=v_max)


def plot_rain(ax, arr, vmin=0, vmax=25):
    assert arr.ndim == 2, "Array must be 2D (H, W)"
    rainbow = mlp.colormaps["rainbow"]
    white = np.array([1, 1, 1, 1])
    # Create colormap for radar
    list_color = rainbow(np.linspace(0, 1, 256))
    list_color[: (256 // 6), :] = white
    cmpradar = ListedColormap(list_color)
    ax.imshow(arr, vmin=vmin, vmax=vmax, origin="lower", cmap=cmpradar)
    ax.axis("off")


plot_dict = {
    "imerg": plot_rain,
    "goes16_rrqpe": plot_rain,
    "fields_intensities": plot_fields_intensities,
}


def make_gif_tensor(y, y_hat, dataset_name, plot_params_dict={}):
    # assume y and y_hat are tensors of shape (T, (C,) H, W)
    T = y.shape[0]
    frames = []
    for t in range(T):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plot_func = plot_dict[dataset_name]
        plot_func(ax[0], y[t], **plot_params_dict)
        plot_func(ax[1], y_hat[t], **plot_params_dict)

        fig.tight_layout()
        frame = mplfig_to_npimage(fig)

        frames.append(frame)

    return np.array(frames)


def transform_multiple_loc(transforms, y, locations):
    """
    Apply multiple transformations to the input tensor based on the provided locations.
    """
    y_transformed = y.clone()  # Ensure we do not modify the original tensor
    for i in range(y.shape[0]):
        location = locations[i]
        if location in transforms:
            y_transformed[i] = transforms[location](y[i])
        else:
            raise ValueError(f"Location {location} not found in transforms.")
    return y_transformed
