import numpy as np
from typing import List
import torch
import torch.nn.functional as F


def get_staggered_x(input: torch.Tensor, mode: str = "replicate") -> torch.Tensor:
    if len(input.shape) == 4:
        grid_staggered_x = (input[..., 1:] + input[..., :-1]) * 0.5
        grid_staggered_x_pad = F.pad(
            grid_staggered_x, pad=(1, 1, 0, 0), mode=mode, value=0
        )
        return grid_staggered_x_pad
    elif len(input.shape) == 5:
        grid_staggered_x = (input[..., 1:] + input[..., :-1]) * 0.5
        grid_staggered_x_pad = F.pad(
            grid_staggered_x, pad=(1, 1, 0, 0, 0, 0), mode=mode, value=0
        )
        return grid_staggered_x_pad
    else:
        raise RuntimeError("A grid has to be 2D(3D) [B, C, (D), H, W] to be staggered")


def get_staggered_y(input: torch.Tensor, mode: str = "replicate") -> torch.Tensor:
    if len(input.shape) == 4:
        grid_staggered_y = (input[..., 1:, :] + input[..., :-1, :]) * 0.5
        grid_staggered_y_pad = F.pad(
            grid_staggered_y, pad=(0, 0, 1, 1), mode=mode, value=0
        )
        return grid_staggered_y_pad
    elif len(input.shape) == 5:
        grid_staggered_y = (input[..., 1:, :] + input[..., :-1, :]) * 0.5
        grid_staggered_y_pad = F.pad(
            grid_staggered_y, pad=(0, 0, 1, 1, 0, 0), mode=mode, value=0
        )
        return grid_staggered_y_pad
    else:
        raise RuntimeError("A grid has to be 2D(3D) [B, C, (D), H, W] to be staggered")


def get_staggered_z(input: torch.Tensor, mode: str = "replicate") -> torch.Tensor:
    if len(input.shape) == 5:
        grid_staggered_z = (input[..., 1:, :, :] + input[..., :-1, :, :]) * 0.5
        grid_staggered_z_pad = F.pad(
            grid_staggered_z, pad=(0, 0, 0, 0, 1, 1), mode=mode, value=0
        )
        return grid_staggered_z_pad
    else:
        raise RuntimeError("A grid has to be 3D [B, C, D, H, W] to be staggered")


def get_staggered(input: torch.Tensor, mode: str = "replicate") -> List[torch.Tensor]:
    dim = input.shape[1]
    if dim < 2 or dim > 3:
        raise RuntimeError("Only 2D or 3D scene supported")

    output = [
        get_staggered_x(input=input[:, 0:1, ...], mode=mode),
        get_staggered_y(input=input[:, 1:2, ...], mode=mode),
    ]
    if dim == 3:
        output.append(get_staggered_z(input=input[:, 2:3, ...], mode=mode))

    return output
