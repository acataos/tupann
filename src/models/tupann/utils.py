import numpy as np
import torch


def warp(input, flow, grid, mode="bilinear", padding_mode="zeros", fill_value=0.0):
    B, C, H, W = input.size()
    vgrid = grid - flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = (
        torch.nn.functional.grid_sample(
            input - fill_value, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True
        )
        + fill_value
    )

    return output


def make_grid(input, device="cuda"):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    return grid


def sigmoid(x, scale=0.5, translation=15):
    return 1 / (1 + np.exp(translation - scale * x))


def combine_mult_fields(self, field_class, field_pred):
    shape = field_pred.shape
    field_pred = field_pred.reshape((shape[0], shape[1] // 2, 2, shape[-1], shape[-1]))
    field_class = field_class.repeat(2, 1)
    assert shape[1] // 2 == self.n_fields
    out_field = torch.empty((shape[0], 2, shape[-1], shape[-1]), device=self.device)
    for i in range(self.n_fields):
        class_bool = field_class == i
        out_field += class_bool * field_pred[:, i, :, :, :]
    return out_field
