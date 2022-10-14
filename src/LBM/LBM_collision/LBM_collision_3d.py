import math
import torch
import torch.nn.functional as F

from src.LBM.LBM_collision import AbstractLBMCollision
from src.LBM.utils import CellType


class LBMCollision3d(AbstractLBMCollision):
    rank = 3

    def __init__(
        self,
        Q: int = 19,
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
        super(LBMCollision3d, self).__init__(*args, **kwargs)
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
            [1.0 / 3.0,
            1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
            1.0 / 18.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
            1.0 / 18.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0]
        ).reshape(1, Q, 1, 1, 1).to(self.device).to(self.dtype)

        # x, y, z direction
        self._e = torch.Tensor(
            [[0, 0, 0],
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
            [1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0],
            [0, 0, 1],
            [1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
            [0, 0, -1],
            [1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]]
        ).reshape(1, Q, 3, 1, 1, 1).to(self.device).to(torch.int64)
    
    def set_gravity(self, gravity: float):
        dim = 3
        self._gravity = torch.Tensor(
            [0.0, -gravity, 0.0]
        ).reshape(1, dim, *([1] * dim)).to(self.device).to(self.dtype)
    
    def get_feq_(
        self,
        dx: float,
        dt: float,
        rho: torch.Tensor,
        vel: torch.Tensor,
        force: torch.Tensor = None,
    ) -> torch.Tensor:
        dim = 3
        tau = self._tau
        if force is not None:
            vel = vel + tau * force / rho
        
        c = dx / dt
        cs2 = c * c / 3.0

        temp_val = torch.sqrt(1.0 + 3.0 * vel * vel / c / c)
        feq = rho * self._weight * (
            (2.0 - temp_val[:, 0:1, ...]) *
            (2.0 - temp_val[:, 1:2, ...]) *
            (2.0 - temp_val[:, 2:3, ...]) *
            torch.pow((2.0 * vel[:, 0:1, ...] / c + temp_val[:, 0:1, ...]) / (1.0 - vel[:, 0:1, ...] / c), self._e[:, :, 0, ...]) *
            torch.pow((2.0 * vel[:, 1:2, ...] / c + temp_val[:, 1:2, ...]) / (1.0 - vel[:, 1:2, ...] / c), self._e[:, :, 1, ...]) *
            torch.pow((2.0 * vel[:, 2:3, ...] / c + temp_val[:, 2:3, ...]) / (1.0 - vel[:, 2:3, ...] / c), self._e[:, :, 2, ...])
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
    
    def collision(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            f: f before streaming [B, Q, res]
            vel: velocity [B, dim, res]
            force: force [B, dim, res]

        Returns:
            torch.Tensor: f after streaming [B, Q, res]
        """
        dim = 3
        tau = self._tau

        feq = self.get_feq_(dx=dx, dt=dt, rho=rho, vel=vel, force=force)

        collision_f = (1.0 - 1.0 / tau) * f + feq / tau
        f_new = torch.where(
            flags == int(CellType.OBSTACLE),
            f,
            collision_f
        )

        return f_new
