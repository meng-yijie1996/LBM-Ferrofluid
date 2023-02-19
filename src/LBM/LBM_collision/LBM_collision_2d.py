import math
import torch
import torch.nn.functional as F

from src.LBM.LBM_collision import AbstractLBMCollision
from src.LBM.utils import CellType, KBCType


class LBMCollision2d(AbstractLBMCollision):
    rank = 2

    def __init__(
        self,
        Q: int = 9,
        tau: float = 1.0,
        density_liquid: float = 0.265,
        density_gas: float = 0.038,
        rho_liquid: float = 0.265,
        rho_gas: float = 0.038,
        kappa: float =  0.08,
        tau_f: float = 0.7,
        tau_g: float = 0.7,
        contact_angle: float = math.pi / 2.0,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        super(LBMCollision2d, self).__init__(*args, **kwargs)
        self._Q = Q
        self._tau = tau

        # parameters for multiphase case
        self._density_liquid = density_liquid
        self._density_gas = density_gas
        self._rho_liquid = rho_liquid
        self._rho_gas = rho_gas
        self._kappa = kappa
        self._tau_f = tau_f
        self._tau_g = tau_g
        self._contact_angle = contact_angle

        self.device = device
        self.dtype = dtype

        self._weight = torch.Tensor(
            [4.0 / 9.0,
            1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0]
        ).reshape(1, Q, 1, 1).to(self.device).to(self.dtype)

        # x, y direction
        self._e = torch.Tensor(
            [[0, 0],
            [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]]
        ).reshape(1, Q, 2, 1, 1).to(self.device).to(torch.int64)
    
    def set_gravity(self, gravity: float):
        dim = 2
        self._gravity = torch.Tensor(
            [0.0, -gravity]
        ).reshape(1, dim, *([1] * dim)).to(self.device).to(self.dtype)
    
    def get_feq_(
        self,
        dx: float,
        dt: float,
        rho: torch.Tensor,
        vel: torch.Tensor,
        force: torch.Tensor = None,
    ) -> torch.Tensor:
        dim = 2
        tau = self._tau
        if force is not None:
            vel = vel + tau * force / rho
        
        c = dx / dt
        cs2 = c * c / 3.0

        temp_val = torch.sqrt(1.0 + 3.0 * vel * vel / c / c)
        feq = rho * self._weight * (
            (2.0 - temp_val[:, 0:1, ...]) *
            (2.0 - temp_val[:, 1:2, ...]) *
            torch.pow((2.0 * vel[:, 0:1, ...] / c + temp_val[:, 0:1, ...]) / (1.0 - vel[:, 0:1, ...] / c), self._e[:, :, 0, ...]) *
            torch.pow((2.0 * vel[:, 1:2, ...] / c + temp_val[:, 1:2, ...]) / (1.0 - vel[:, 1:2, ...] / c), self._e[:, :, 1, ...])
        )

        # # constraint of nan
        # eps = 1e-4
        # feq = torch.where(
        #     (torch.abs(vel - 1.0) <= eps).any(dim=1).unsqueeze(1).repeat(1, self._Q, *([1] * dim)),
        #     torch.zeros_like(feq),
        #     feq
        # )

        # uv = (vel * vel).sum(dim=1).unsqueeze(1)  # [B, 1, res]
        # eu = (vel.unsqueeze(1) * self._e * c).sum(dim=2)  # [B, Q, res]
        # feq = rho * self._weight * (
        #     1.0 + eu / cs2 + 0.5 * eu * eu / cs2 / cs2 - 0.5 * uv / cs2
        # )
        
        return feq

    def get_geq_(
        self,
        dx: float,
        dt: float,
        rho: torch.Tensor,
        vel: torch.Tensor,
        pressure: torch.Tensor,
        force: torch.Tensor,
        feq: torch.Tensor = None
    ) -> torch.Tensor:
        c = dx / dt
        cs2 = c * c / 3.0
        if feq is None:
            feq = self.get_feq_(dx=dx, dt=dt, rho=rho, vel=vel, force=force)

        geq = self._weight * (
            pressure + cs2 * rho * (
                feq / self._weight / rho - 1.0
            )
        )
        
        return geq
    
    def KBC_postprocess(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        feq: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        KBC_type: int=int(KBCType.KBC_C)
    ):
        beta = 0.5 / self._tau

        c = dx / dt

        moment = torch.zeros((*rho.shape, 3, 3)).to(rho.device).to(rho.dtype)
        moment_eq = torch.zeros((*rho.shape, 3, 3)).to(rho.device).to(rho.dtype)
        if KBC_type == int(KBCType.KBC_C) or KBC_type == int(KBCType.KBC_D):
            moment = (
                self.C_mat * f.unsqueeze(-1).unsqueeze(-1)
            ).sum(dim=1).unsqueeze(1)  # [B, 1, res, 3, 3]
            moment_eq = (
                self.C_mat * feq.unsqueeze(-1).unsqueeze(-1)
            ).sum(dim=1).unsqueeze(1)  # [B, 1, res, 3, 3]
        elif KBC_type == int(KBCType.KBC_A) or KBC_type == int(KBCType.KBC_B):
            for i in range(3):
                for j in range(3):
                    temp_val =  (
                        torch.pow(self._e[:, :, 0, ...] * c - vel[:, 0:1, ...], i) *
                        torch.pow(self._e[:, :, 1, ...] * c - vel[:, 1:2, ...], j)
                    )  # [B, Q, res]
                    moment[..., i, j] = (f * temp_val).sum(dim=1).unsqueeze(1)
                    moment_eq[..., i, j] = (feq * temp_val).sum(dim=1).unsqueeze(1)

        # KBC - parameters
        KBC_T = moment[..., 2, 0] + moment[..., 0, 2]
        KBC_N = moment[..., 2, 0] - moment[..., 0, 2]
        KBC_PIxy = moment[..., 1, 1]
        # KBC_Qxxy = moment[..., 2, 1]
        # KBC_Qxyy = moment[..., 1, 2]
        # KBC_A = moment[..., 2, 2]

        KBC_T_eq = moment_eq[..., 2, 0] + moment_eq[..., 0, 2]
        KBC_N_eq = moment_eq[..., 2, 0] - moment_eq[..., 0, 2]
        KBC_PIxy_eq = moment_eq[..., 1, 1]
        # KBC_Qxxy_eq = moment_eq[..., 2, 1]
        # KBC_Qxyy_eq = moment_eq[..., 1, 2]
        # KBC_A_eq = moment_eq[..., 2, 2]

        KBC_ds = torch.zeros_like(f)
        if KBC_type == int(KBCType.KBC_A) or KBC_type == int(KBCType.KBC_C):
            # T, N, PI only
            KBC_ds[:, 0:1, ...] = (
                (1.0 - KBC_T) - (1.0 - KBC_T_eq)
            )  # 0, 0

            KBC_ds[:, 1:2, ...] = 0.5 * (
                (0.5 * (KBC_T + KBC_N) + 1 * vel[:, 0:1, ...]) -
                (0.5 * (KBC_T_eq + KBC_N_eq) + 1 * vel[:, 0:1, ...])
            )  # 1, 0
            KBC_ds[:, 3:4, ...] = 0.5 * (
                (0.5 * (KBC_T + KBC_N) - 1 * vel[:, 0:1, ...]) -
                (0.5 * (KBC_T_eq + KBC_N_eq) - 1 * vel[:, 0:1, ...])
            )  # -1, 0
            KBC_ds[:, 2:3, ...] = 0.5 * (
                (0.5 * (KBC_T - KBC_N) + 1 * vel[:, 1:2, ...]) -
                (0.5 * (KBC_T_eq - KBC_N_eq) + 1 * vel[:, 1:2, ...])
            )  # 0, 1
            KBC_ds[:, 4:5, ...] = 0.5 * (
                (0.5 * (KBC_T - KBC_N) - 1 * vel[:, 1:2, ...]) -
                (0.5 * (KBC_T_eq - KBC_N_eq) - 1 * vel[:, 1:2, ...])
            )  # 0, -1

            KBC_ds[:, 5:6, ...] = 0.25 * (
                (KBC_PIxy) -
                (KBC_PIxy_eq)
            )  # 1, 1
            KBC_ds[:, 6:7, ...] = 0.25 * (
                (-KBC_PIxy) -
                (-KBC_PIxy_eq)
            )  # -1, 1
            KBC_ds[:, 7:8, ...] = 0.25 * (
                (KBC_PIxy) -
                (KBC_PIxy_eq)
            )  # -1, -1
            KBC_ds[:, 8:9, ...] = 0.25 * (
                (-KBC_PIxy) -
                (-KBC_PIxy_eq)
            )   # 1, -1
        elif KBC_type ==  int(KBCType.KBC_B) or KBC_type == int(KBCType.KBC_D):
            # N, PI only
            KBC_ds[:, 0:1, ...] = (
                (1.0) - (1.0)
            )  # 0, 0

            KBC_ds[:, 1:2, ...] = 0.5 * (
                (0.5 * (KBC_N) + 1 * vel[:, 0:1, ...]) -
                (0.5 * (KBC_N_eq) + 1 * vel[:, 0:1, ...])
            )  # 1, 0
            KBC_ds[:, 3:4, ...] = 0.5 * (
                (0.5 * (KBC_N) - 1 * vel[:, 0:1, ...]) -
                (0.5 * (KBC_N_eq) - 1 * vel[:, 0:1, ...])
            )  # -1, 0
            KBC_ds[:, 2:3, ...] = 0.5 * (
                (0.5 * (-KBC_N) + 1 * vel[:, 1:2, ...]) -
                (0.5 * (-KBC_N_eq) + 1 * vel[:, 1:2, ...])
            )  # 0, 1
            KBC_ds[:, 4:5, ...] = 0.5 * (
                (0.5 * (-KBC_N) - 1 * vel[:, 1:2, ...]) -
                (0.5 * (-KBC_N_eq) - 1 * vel[:, 1:2, ...])
            )  # 0, -1

            KBC_ds[:, 5:6, ...] = 0.25 * (
                (KBC_PIxy) -
                (KBC_PIxy_eq)
            )  # 1, 1
            KBC_ds[:, 6:7, ...] = 0.25 * (
                (-KBC_PIxy) -
                (-KBC_PIxy_eq)
            )  # -1, 1
            KBC_ds[:, 7:8, ...] = 0.25 * (
                (KBC_PIxy) -
                (KBC_PIxy_eq)
            )  # -1, -1
            KBC_ds[:, 8:9, ...] = 0.25 * (
                (-KBC_PIxy) -
                (-KBC_PIxy_eq)
            )   # 1, -1
        
        KBC_dh = f - feq - KBC_ds
        gamma_nominator = KBC_ds * KBC_dh / (feq + 1e-7)
        gamma_determinator = KBC_dh * KBC_dh / (feq + 1e-7)
        KBC_gamma = gamma_nominator / (gamma_determinator + 1e-7)
        KBC_gamma = (1.0 - (2.0 * beta - 1.0) * KBC_gamma) / beta

        return f - 2.0 * beta * KBC_ds - beta * KBC_dh * KBC_gamma
    
    def collision(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor,
        KBC_type: int=None
    ) -> torch.Tensor:
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
        dim = 2
        tau = self._tau

        feq = self.get_feq_(dx=dx, dt=dt, rho=rho, vel=vel, force=force)
        collision_f = (1.0 - 1.0 / tau) * f + feq / tau
        f_new = torch.where(
            flags == int(CellType.OBSTACLE),
            f,
            collision_f
        )

        return f_new
