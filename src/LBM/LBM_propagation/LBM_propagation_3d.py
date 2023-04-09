import torch
import torch.nn.functional as F

from src.LBM.LBM_propagation import AbstractLBMPropagation
from src.LBM.utils import CellType


class LBMPropagation3d(AbstractLBMPropagation):
    rank = 2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(LBMPropagation3d, self).__init__(*args, **kwargs)

    def propagation(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f (torch.Tensor): f before streaming [B, Q, res]

        Returns:
            torch.Tensor: f after streaming [B, Q, res]
        """
        f_pad = F.pad(f, pad=(1, 1, 1, 1, 1, 1), mode="replicate")
        f_new_pad = torch.zeros_like(f_pad)

        # Providing periodic condition as default
        # If any further condition applies, boundary will be handeled subsequently
        # boundary faces (6)
        f_pad[..., 1:-1, 1:-1, 0] = f_pad[..., 1:-1, 1:-1, -2]
        f_pad[..., 1:-1, 1:-1, -1] = f_pad[..., 1:-1, 1:-1, 1]
        f_pad[..., 1:-1, 0, 1:-1] = f_pad[..., 1:-1, -2, 1:-1]
        f_pad[..., 1:-1, -1, 1:-1] = f_pad[..., 1:-1, 1, 1:-1]
        f_pad[..., 0, 1:-1, 1:-1] = f_pad[..., -2, 1:-1, 1:-1]
        f_pad[..., -1, 1:-1, 1:-1] = f_pad[..., 1, 1:-1, 1:-1]

        # bounday edges (12)
        f_pad[..., 1:-1, 0, 0] = f_pad[..., 1:-1, -2, -2]
        f_pad[..., 1:-1, 0, -1] = f_pad[..., 1:-1, -2, 1]
        f_pad[..., 1:-1, -1, 0] = f_pad[..., 1:-1, 1, -2]
        f_pad[..., 1:-1, -1, -1] = f_pad[..., 1:-1, 1, 1]

        f_pad[..., 0, 1:-1, 0] = f_pad[..., -2, 1:-1, -2]
        f_pad[..., 0, 1:-1, -1] = f_pad[..., -2, 1:-1, 1]
        f_pad[..., -1, 1:-1, 0] = f_pad[..., 1, 1:-1, -2]
        f_pad[..., -1, 1:-1, -1] = f_pad[..., 1, 1:-1, 1]

        f_pad[..., 0, 0, 1:-1] = f_pad[..., -2, -2, 1:-1]
        f_pad[..., 0, -1, 1:-1] = f_pad[..., -2, 1, 1:-1]
        f_pad[..., -1, 0, 1:-1] = f_pad[..., 1, -2, 1:-1]
        f_pad[..., -1, -1, 1:-1] = f_pad[..., 1, 1, 1:-1]

        # boundary points (8)
        f_pad[..., 0, 0, 0] = f_pad[..., -2, -2, -2]
        f_pad[..., 0, 0, -1] = f_pad[..., -2, -2, 1]
        f_pad[..., 0, -1, 0] = f_pad[..., -2, 1, -2]
        f_pad[..., 0, -1, -1] = f_pad[..., -2, 1, 1]

        f_pad[..., -1, 0, 0] = f_pad[..., 1, -2, -2]
        f_pad[..., -1, 0, -1] = f_pad[..., 1, -2, 1]
        f_pad[..., -1, -1, 0] = f_pad[..., 1, 1, -2]
        f_pad[..., -1, -1, -1] = f_pad[..., 1, 1, 1]

        # center
        f_new_pad[..., 0, :, :, :] = f_pad[..., 0, :, :, :]

        # pos x
        f_new_pad[..., 1, :, :, 1:] = f_pad[..., 1, :, :, :-1]
        # pos y
        f_new_pad[..., 2, :, 1:, :] = f_pad[..., 2, :, :-1, :]
        # neg x
        f_new_pad[..., 3, :, :, :-1] = f_pad[..., 3, :, :, 1:]
        # neg y
        f_new_pad[..., 4, :, :-1, :] = f_pad[..., 4, :, 1:, :]

        # pos x, pos y
        f_new_pad[..., 5, :, 1:, 1:] = f_pad[..., 5, :, :-1, :-1]
        # neg x, pos y
        f_new_pad[..., 6, :, 1:, :-1] = f_pad[..., 6, :, :-1, 1:]
        # neg x, neg y
        f_new_pad[..., 7, :, :-1, :-1] = f_pad[..., 7, :, 1:, 1:]
        # pos x, neg y
        f_new_pad[..., 8, :, :-1, 1:] = f_pad[..., 8, :, 1:, :-1]

        # pos z
        f_new_pad[..., 9, 1:, :, :] = f_pad[..., 9, :-1, :, :]

        # pos x, pos z
        f_new_pad[..., 10, 1:, :, 1:] = f_pad[..., 10, :-1, :, :-1]
        # pos y, pos z
        f_new_pad[..., 11, 1:, 1:, :] = f_pad[..., 11, :-1, :-1, :]
        # neg x, pos z
        f_new_pad[..., 12, 1:, :, :-1] = f_pad[..., 12, :-1, :, 1:]
        # neg y, pos z
        f_new_pad[..., 13, 1:, :-1, :] = f_pad[..., 13, :-1, 1:, :]

        # neg z
        f_new_pad[..., 14, :-1, :, :] = f_pad[..., 14, 1:, :, :]

        # pos x, neg z
        f_new_pad[..., 15, :-1, :, 1:] = f_pad[..., 15, 1:, :, :-1]
        # pos y, neg z
        f_new_pad[..., 16, :-1, 1:, :] = f_pad[..., 16, 1:, :-1, :]
        # neg x, neg z
        f_new_pad[..., 17, :-1, :, :-1] = f_pad[..., 17, 1:, :, 1:]
        # neg y, neg z
        f_new_pad[..., 18, :-1, :-1, :] = f_pad[..., 18, 1:, 1:, :]

        return f_new_pad[..., 1:-1, 1:-1, 1:-1]

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

        inverted_f[:, 9, ...] = f[:, 14, ...]
        inverted_f[:, 14, ...] = f[:, 9, ...]

        inverted_f[:, 10, ...] = f[:, 17, ...]
        inverted_f[:, 11, ...] = f[:, 18, ...]
        inverted_f[:, 12, ...] = f[:, 15, ...]
        inverted_f[:, 13, ...] = f[:, 16, ...]

        inverted_f[:, 15, ...] = f[:, 12, ...]
        inverted_f[:, 16, ...] = f[:, 13, ...]
        inverted_f[:, 17, ...] = f[:, 10, ...]
        inverted_f[:, 18, ...] = f[:, 11, ...]

        f_new = torch.where(flags == int(CellType.OBSTACLE), inverted_f, f)

        return f_new
