import torch
import torch.nn.functional as F

from src.LBM.LBM_propagation import AbstractLBMPropagation
from src.LBM.utils import CellType


class LBMPropagation2d(AbstractLBMPropagation):
    rank = 2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(LBMPropagation2d, self).__init__(*args, **kwargs)

    def propagation(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f (torch.Tensor): f before streaming [B, Q, res]

        Returns:
            torch.Tensor: f after streaming [B, Q, res]
        """
        f_pad = F.pad(f, pad=(1, 1, 1, 1), mode="replicate")
        f_new_pad = torch.zeros_like(f_pad)

        # Providing periodic condition as default
        # If any further condition applies, boundary will be handeled subsequently
        f_pad[..., 1:-1, 0] = f_pad[..., 1:-1, -2]
        f_pad[..., 1:-1, -1] = f_pad[..., 1:-1, 1]
        f_pad[..., 0, 1:-1] = f_pad[..., -2, 1:-1]
        f_pad[..., -1, 1:-1] = f_pad[..., 1, 1:-1]

        f_pad[..., 0, 0] = f_pad[..., -2, -2]
        f_pad[..., 0, -1] = f_pad[..., -2, 1]
        f_pad[..., -1, 0] = f_pad[..., 1, -2]
        f_pad[..., -1, -1] = f_pad[..., 1, 1]

        # center
        f_new_pad[..., 0, :, :] = f_pad[..., 0, :, :]

        # pos x
        f_new_pad[..., 1, :, 1:] = f_pad[..., 1, :, :-1]

        # pos y
        f_new_pad[..., 2, 1:, :] = f_pad[..., 2, :-1, :]

        # neg x
        f_new_pad[..., 3, :, :-1] = f_pad[..., 3, :, 1:]

        # neg y
        f_new_pad[..., 4, :-1, :] = f_pad[..., 4, 1:, :]

        # pos x, pos y
        f_new_pad[..., 5, 1:, 1:] = f_pad[..., 5, :-1, :-1]

        # neg x, pos y
        f_new_pad[..., 6, 1:, :-1] = f_pad[..., 6, :-1, 1:]

        # neg x, neg y
        f_new_pad[..., 7, :-1, :-1] = f_pad[..., 7, 1:, 1:]

        # pos x, neg y
        f_new_pad[..., 8, :-1, 1:] = f_pad[..., 8, 1:, :-1]

        return f_new_pad[..., 1:-1, 1:-1]

    def rebounce_obstacle(self, f: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        inverted_f = torch.zeros_like(f)
        inverted_f[:, 0, ...] = f[:, 0, ...]

        inverted_f[:, 1, ...] = f[:, 3, ...]
        inverted_f[:, 2, ...] = f[:, 4, ...]
        inverted_f[:, 3, ...] = f[:, 1, ...]
        inverted_f[:, 4, ...] = f[:, 2, ...]

        inverted_f[:, 5, ...] = f[:, 7, ...]
        inverted_f[:, 6, ...] = f[:, 8, ...]
        inverted_f[:, 7, ...] = f[:, 5, ...]
        inverted_f[:, 8, ...] = f[:, 6, ...]

        f_new = torch.where(flags == int(CellType.OBSTACLE), inverted_f, f)

        return f_new
