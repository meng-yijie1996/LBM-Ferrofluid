import torch
import torch.nn.functional as F
from typing import List

from src.LBM.LBM_macro_compute import AbstractLBMMacroCompute
from src.LBM.utils import CellType


class LBMMacroCompute2d(AbstractLBMMacroCompute):
    rank = 2

    def __init__(
        self,
        Q: int = 9,
        tau: float = 1.0,
        density_liquid: float = 0.265,
        density_gas: float = 0.038,
        rho_liquid: float = 0.265,
        rho_gas: float = 0.038,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        super(LBMMacroCompute2d, self).__init__(*args, **kwargs)
        self._Q = Q
        self._tau = tau

        # parameters for multiphase case
        self._density_liquid = density_liquid
        self._density_gas = density_gas
        self._rho_liquid = rho_liquid
        self._rho_gas = rho_gas
        
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

    def get_pressure(self, dx: float, dt: float, density: torch.Tensor) -> torch.Tensor:
        c = dx / dt
        cs2 = c * c / 3.0
        RT = cs2
        a = 12.0 * RT
        b = 4.0

        temp_density = b * density / 4.0
        pressure = density * RT * temp_density * (4.0 - 2.0 * temp_density) / torch.pow((1 - temp_density), 3) - \
            a * density * density + \
            density * RT
        
        return pressure
    
    def macro_compute(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        g: torch.Tensor=None,
        pressure: torch.Tensor=None,
        density: torch.Tensor=None,
    ) -> List[torch.Tensor]:
        dim = 2
        c = dx / dt

        macro_rho = f.sum(dim=1).unsqueeze(1)  # [B, 1, res]
        rho_new = torch.where(
            flags == int(CellType.OBSTACLE),
            rho,
            macro_rho
        )

        macro_vel = (f.unsqueeze(2) * self._e).sum(dim=1) * (c / rho_new)  # [B, dim, res]
        vel_new = torch.where(
            (flags == int(CellType.OBSTACLE)).repeat(1, dim, *([1] * dim)),
            vel,
            macro_vel
        )

        if density is not None:
            density_liquid = self._density_liquid
            density_gas = self._density_gas
            rho_liquid = self._rho_liquid
            rho_gas = self._rho_gas
            density = density_gas + (density_liquid - density_gas) * (
                (rho_new - rho_gas) / (rho_liquid - rho_gas)
            )
            if pressure is not None:
                pressure = self.get_pressure(dx=dx, dt=dt, density=density)
            
            return [rho_new, vel_new, density]

        return [rho_new, vel_new]
    
    def get_vort(self, vel: torch.Tensor, dx: float) -> torch.Tensor:
        vort = (
            (vel[..., 0:1, 2:, 1:-1] - vel[..., 0:1, :-2, 1:-1]) - 
            (vel[..., 1:2, 1:-1, 2:] - vel[..., 1:2, 1:-1, :-2])
        ) / (2.0 * dx)

        vort_pad = F.pad(vort, pad=(1, 1, 1, 1), mode="replicate")

        return vort_pad
