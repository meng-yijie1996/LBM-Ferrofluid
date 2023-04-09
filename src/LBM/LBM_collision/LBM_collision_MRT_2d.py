import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from src.LBM.LBM_collision import LBMCollision2d
from src.LBM.utils import CellType, KBCType


class LBMCollisionMRT2d(LBMCollision2d):
    rank = 2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(LBMCollisionMRT2d, self).__init__(*args, **kwargs)

        self.C_mat = torch.zeros((self._Q, 3, 3)).to(self.device).to(self.dtype)

    def preset_KBC(self, dx: float, dt: float):
        dim = 2
        c = dx / dt
        c2 = c * c
        c3 = c * c2
        c4 = c2 * c2

        # Q - x - y
        self.C_mat[:, 0, 0] = 1 * torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]).to(
            self.device
        ).to(self.dtype)
        self.C_mat[:, 1, 0] = c * torch.Tensor([0, 1, 0, -1, 0, 1, -1, -1, 1]).to(
            self.device
        ).to(self.dtype)
        self.C_mat[:, 0, 1] = c * torch.Tensor([0, 0, 1, 0, -1, 1, 1, -1, -1]).to(
            self.device
        ).to(self.dtype)
        self.C_mat[:, 2, 0] = c2 * torch.Tensor([0, 1, 0, 1, 0, 1, 1, 1, 1]).to(
            self.device
        ).to(self.dtype)
        self.C_mat[:, 0, 2] = c2 * torch.Tensor([0, 0, 1, 0, 1, 1, 1, 1, 1]).to(
            self.device
        ).to(self.dtype)
        self.C_mat[:, 1, 1] = c2 * torch.Tensor([0, 0, 0, 0, 0, 1, -1, 1, -1]).to(
            self.device
        ).to(self.dtype)
        self.C_mat[:, 2, 2] = c4 * torch.Tensor([0, 0, 0, 0, 0, 1, 1, 1, 1]).to(
            self.device
        ).to(self.dtype)
        self.C_mat[:, 2, 1] = c3 * torch.Tensor([0, 0, 0, 0, 0, 1, 1, -1, -1]).to(
            self.device
        ).to(self.dtype)
        self.C_mat[:, 1, 2] = c3 * torch.Tensor([0, 0, 0, 0, 0, 1, -1, -1, 1]).to(
            self.device
        ).to(self.dtype)

        #  [B, Q, 3, 3, *res]
        self.C_mat = self.C_mat.reshape(1, self._Q, 3, 3, *([1] * dim))

    def get_s_by_KBC(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        KBC_type: int = None,
    ):
        """
        Args:
            f: f before streaming [B, Q, res]
            rho: density [B, 1, res]

        Returns:
            torch.Tensor: f after streaming [B, Q, res]
        """
        dim = 2
        c = dx / dt
        assert self._Q == 9

        need_centering = KBCType.is_KBC_AB(KBC_type)

        if need_centering:
            p = (
                torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
                .reshape(1, 1, 3, 3, *([1] * dim))
                .to(self.device)
                .to(torch.int32)
            )
            q = (
                torch.Tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
                .reshape(1, 1, 3, 3, *([1] * dim))
                .to(self.device)
                .to(torch.int32)
            )
            vp = c * torch.pow(
                (self._e[:, :, 0:1, ...] - vel[:, 0:1, ...]).unsqueeze(2), p
            )  # [B, Q, 3, 3, *res]
            vq = c * torch.pow(
                (self._e[:, :, 1:2, ...] - vel[:, 1:2, ...]).unsqueeze(2), q
            )  # [B, Q, 3, 3, *res]
            vp_vq = vp * vq  # [B, Q, 3, 3, *res]
            moments_flux = (vp_vq * (f / rho).unsqueeze(2).unsqueeze(2)).sum(
                dim=1
            )  # [B, 3, 3, *res]
        else:
            moments_flux = (self.C_mat * (f / rho).unsqueeze(2).unsqueeze(2)).sum(
                dim=1
            )  # [B, 3, 3, *res]

        T = moments_flux[:, 2, 0, ...] + moments_flux[:, 0, 2, ...]
        N = moments_flux[:, 2, 0, ...] - moments_flux[:, 0, 2, ...]
        PI_xy = moments_flux[:, 1, 1, ...]

        Qxxy = moments_flux[:, 2, 1, ...]
        Qxyy = moments_flux[:, 1, 2, ...]
        A = moments_flux[:, 2, 2, ...]

        s = torch.zeros_like(f)
        # Defaultly owns T and PI_xy
        s[:, 0, ...] += rho[:, 0, ...] * (1 - T)
        sigma_ = 1
        s[:, 1, ...] += 0.5 * rho[:, 0, ...] * (0.5 * (T))
        sigma_ = -1
        s[:, 3, ...] += 0.5 * rho[:, 0, ...] * (0.5 * (T))
        lambda_ = 1
        s[:, 2, ...] += 0.5 * rho[:, 0, ...] * (0.5 * (T))
        lambda_ = -1
        s[:, 4, ...] += 0.5 * rho[:, 0, ...] * (0.5 * (T))
        sigma_, lambda_ = 1, 1
        s[:, 5, ...] += 0.25 * rho[:, 0, ...] * (sigma_ * lambda_ * PI_xy)
        sigma_, lambda_ = -1, 1
        s[:, 6, ...] += 0.25 * rho[:, 0, ...] * (sigma_ * lambda_ * PI_xy)
        sigma_, lambda_ = -1, -1
        s[:, 7, ...] += 0.25 * rho[:, 0, ...] * (sigma_ * lambda_ * PI_xy)
        sigma_, lambda_ = 1, -1
        s[:, 8, ...] += 0.25 * rho[:, 0, ...] * (sigma_ * lambda_ * PI_xy)

        is_KBC_AC = KBCType.is_KBC_AC(KBC_type)
        if is_KBC_AC:
            # s owns N
            sigma_ = 1
            s[:, 1, ...] += 0.5 * rho[:, 0, ...] * (0.5 * (N))
            sigma_ = -1
            s[:, 3, ...] += 0.5 * rho[:, 0, ...] * (0.5 * (N))
            lambda_ = 1
            s[:, 2, ...] += 0.5 * rho[:, 0, ...] * (0.5 * (-N))
            lambda_ = -1
            s[:, 4, ...] += 0.5 * rho[:, 0, ...] * (0.5 * (-N))

        is_KBC = KBCType.is_KBC(KBC_type)
        if not is_KBC:
            # s owns Qxxy, Qxyy, A
            s[:, 0, ...] += rho[:, 0, ...] * (A)
            sigma_ = 1
            s[:, 1, ...] += (
                0.5 * rho[:, 0, ...] * (sigma_ * (vel[:, 0, ...] - Qxyy) - A)
            )
            sigma_ = -1
            s[:, 3, ...] += (
                0.5 * rho[:, 0, ...] * (sigma_ * (vel[:, 0, ...] - Qxyy) - A)
            )
            lambda_ = 1
            s[:, 2, ...] += (
                0.5 * rho[:, 0, ...] * (lambda_ * (vel[:, 1, ...] - Qxxy) - A)
            )
            lambda_ = -1
            s[:, 4, ...] += (
                0.5 * rho[:, 0, ...] * (lambda_ * (vel[:, 1, ...] - Qxxy) - A)
            )
            sigma_, lambda_ = 1, 1
            s[:, 5, ...] += 0.25 * rho[:, 0, ...] * (A + sigma_ * Qxyy + lambda_ * Qxxy)
            sigma_, lambda_ = -1, 1
            s[:, 6, ...] += 0.25 * rho[:, 0, ...] * (A + sigma_ * Qxyy + lambda_ * Qxxy)
            sigma_, lambda_ = -1, -1
            s[:, 7, ...] += 0.25 * rho[:, 0, ...] * (A + sigma_ * Qxyy + lambda_ * Qxxy)
            sigma_, lambda_ = 1, -1
            s[:, 8, ...] += 0.25 * rho[:, 0, ...] * (A + sigma_ * Qxyy + lambda_ * Qxxy)

        return s

    def collision(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor,
        g: torch.Tensor = None,
        pressure: torch.Tensor = None,
        dfai: torch.Tensor = None,
        dprho: torch.Tensor = None,
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
        feq = self.get_feq_(dx=dx, dt=dt, rho=rho, vel=vel, force=force)

        ds = self.get_s_by_KBC(
            dx=dx, dt=dt, f=f, rho=rho, vel=vel, KBC_type=KBC_type
        ) - self.get_s_by_KBC(dx=dx, dt=dt, f=feq, rho=rho, vel=vel, KBC_type=KBC_type)
        dh = (f - feq) - ds

        beta = 0.5 / self._tau
        gamma = 1.0 / beta - (2.0 - 1.0 / beta) * (ds * dh / feq).sum(dim=1).unsqueeze(
            1
        ) / (dh * dh / feq).sum(dim=1).unsqueeze(1)
        collision_f = f + beta * (-2.0 * ds - gamma * dh)

        f_new = torch.where(flags == int(CellType.OBSTACLE), f, collision_f)

        return f_new
