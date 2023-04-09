import numpy as np
from typing import List
import torch
import torch.nn.functional as F


def get_staggered_x(input: torch.Tensor) -> torch.Tensor:
    if len(input.shape) == 4:
        grid_pad_x = F.pad(input, pad=(1, 1, 0, 0), mode="replicate")
        grid_staggered_x = (grid_pad_x[..., 1:] + grid_pad_x[..., :-1]) * 0.5
        return grid_staggered_x
    elif len(input.shape) == 5:
        grid_pad_x = F.pad(input, pad=(1, 1, 0, 0, 0, 0), mode="replicate")
        grid_staggered_x = (grid_pad_x[..., 1:] + grid_pad_x[..., :-1]) * 0.5
        return grid_staggered_x
    else:
        raise RuntimeError("A grid has to be 2D(3D) [B, C, (D), H, W] to be staggered")


def get_staggered_y(input: torch.Tensor) -> torch.Tensor:
    if len(input.shape) == 4:
        grid_pad_y = F.pad(input, pad=(0, 0, 1, 1), mode="replicate")
        grid_staggered_y = (grid_pad_y[..., 1:, :] + grid_pad_y[..., :-1, :]) * 0.5
        return grid_staggered_y
    elif len(input.shape) == 5:
        grid_pad_y = F.pad(input, pad=(0, 0, 1, 1, 0, 0), mode="replicate")
        grid_staggered_y = (grid_pad_y[..., 1:, :] + grid_pad_y[..., :-1, :]) * 0.5
        return grid_staggered_y
    else:
        raise RuntimeError("A grid has to be 2D(3D) [B, C, (D), H, W] to be staggered")


def get_staggered_z(input: torch.Tensor) -> torch.Tensor:
    if len(input.shape) == 5:
        grid_pad_z = F.pad(input, pad=(0, 0, 0, 0, 1, 1), mode="replicate")
        grid_staggered_z = (
            grid_pad_z[..., 1:, :, :] + grid_pad_z[..., :-1, :, :]
        ) * 0.5
        return grid_staggered_z
    else:
        raise RuntimeError("A grid has to be 3D [B, C, D, H, W] to be staggered")


def get_staggered(input: torch.Tensor) -> List[torch.Tensor]:
    dim = input.shape[1]
    if dim < 2 or dim > 3:
        raise RuntimeError("Only 2D or 3D scene supported")

    output = [
        get_staggered_x(input=input[:, 0:1, ...]),
        get_staggered_y(input=input[:, 1:2, ...]),
    ]
    if dim == 3:
        output.append(get_staggered_z(input=input[:, 2:3, ...]))

    return output
