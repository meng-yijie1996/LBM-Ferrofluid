import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from src.LBM.LBM_collision import LBMCollisionMRT2d, LBMCollision2d
from src.LBM.utils import CellType


class LBMCollisionSC2d(LBMCollision2d):
    rank = 2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(LBMCollisionSC2d, self).__init__(*args, **kwargs)
        self._density_wall = 0.5 * (self._density_gas + self._density_liquid)

    def calculate_force(
        self,
        dx: float,
        dt: float,
        density: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        c = dx / dt
        cs2 = c * c / 3.0
        RT = cs2
        a = 12.0 * cs2
        b = 4.0

        G1 = -1.0 / 3.0

        psx = density * b / 4.0
        psx = (
            RT * (1.0 + (4.0 * psx - 2.0 * psx * psx) / torch.pow(1.0 - psx, 3))
            - a * density
            - cs2
        )
        psx = torch.sqrt(2.0 * density * psx / G1 / cs2)

        # pressure = density * cs2 + 0.5 * cs2 * G1 * psx * psx

        psx_wall = self._density_wall * b / 4.0
        psx_wall = (
            RT
            * (
                1.0
                + (4.0 * psx_wall - 2.0 * psx_wall * psx_wall)
                / math.pow(1.0 - psx_wall, 3)
            )
            - a * self._density_wall
            - cs2
        )
        psx_wall = math.sqrt(2.0 * self._density_wall * psx_wall / G1 / cs2)

        G1 = -1.0 / 3.0
        psx_pad = F.pad(
            torch.where(
                flags == int(CellType.OBSTACLE), psx_wall * torch.ones_like(psx), psx
            ),
            pad=(1, 1, 1, 1),
            mode="constant",
        )
        psx_neighbors = torch.cat(
            (
                psx_pad[..., 1:-1, 2:],
                psx_pad[..., 2:, 1:-1],
                psx_pad[..., 1:-1, :-2],
                psx_pad[..., :-2, 1:-1],
                psx_pad[..., 2:, 2:],
                psx_pad[..., 2:, :-2],
                psx_pad[..., :-2, :-2],
                psx_pad[..., :-2, 2:],
            ),
            dim=1,
        )  # [B, Q-1, *res]
        force = (
            -G1
            * psx
            * c
            * (
                self._weight[:, 1:, ...].unsqueeze(2)
                * self._e[:, 1:, ...]
                * psx_neighbors.unsqueeze(2)
            ).sum(dim=1)
        )  # [B, dim, *res]

        force = torch.where(
            flags == int(CellType.FLUID), force, torch.zeros_like(force)
        )

        return force

    def collision(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        density: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor,
        KBC_type: int = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            f: f before streaming [B, Q, res]
            rho: density [B, 1, res]
            vel: velocity [B, dim, res]
            flags: flags [B, 1, res]
            force: force [B, dim, res]
            KBC_type: int = [None, 'A', 'B', 'C', 'D'], where None is LBGK case, 'A/B/C/D' is different KBC cases

        Returns:
            torch.Tensor: f after streaming [B, Q, res]
        """
        force = self.calculate_force(
            dx=dx, dt=dt, density=density, flags=flags, force=force
        )
        f_new = super(LBMCollisionSC2d, self).collision(
            dx=dx,
            dt=dt,
            f=f,
            rho=rho,
            vel=vel,
            flags=flags,
            force=force,
            KBC_type=KBC_type,
        )

        return f_new
