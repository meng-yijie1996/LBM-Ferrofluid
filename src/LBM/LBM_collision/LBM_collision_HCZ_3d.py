import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from src.LBM.LBM_collision import LBMCollision3d
from src.LBM.utils import CellType


class LBMCollisionHCZ3d(LBMCollision3d):
    rank = 3

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(LBMCollisionHCZ3d, self).__init__(*args, **kwargs)
    
    def capillary_process(
        self,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor,
        dt: float = 1.0,
        dx: float  = 1.0,
        g: torch.Tensor = None,
        density: torch.Tensor = None,
        pressure: torch.Tensor = None,
        H2: torch.Tensor = None,
        phi: torch.Tensor = None,
    ):
        dim = 3
        pad = (1, 1, 1, 1, 1, 1)
        eps = 1e-6

        c = dx / dt
        cs2 = c * c / 3.0
        RT = cs2
        a = 12.0 * RT
        b = 4.0

        temp_rho = b * rho / 4.0
        prho = F.pad(
            (
                pressure - RT * density
            )[..., 1:-1, 1:-1, 1:-1],
            pad=pad,
            mode="replicate"
        )
        fai = F.pad(
            (rho * RT * (
                    4.0 * temp_rho - 2.0 * temp_rho * temp_rho
                ) / torch.pow(1.0 - temp_rho, 3) - a * rho * rho
            )[..., 1:-1, 1:-1, 1:-1],
            pad=pad,
            mode="replicate"
        )

        # ===========================
        #      Contact Angle
        # ===========================
        # 1. neg x
        hlp_CA = torch.sqrt(
            eps +
            (rho[..., 2:, 1:-1, 1] - rho[..., :-2, 1:-1, 1]) * (rho[..., 2:, 1:-1, 1] - rho[..., :-2, 1:-1, 1]) +
            (rho[..., 1:-1, 2:, 1] - rho[..., 1:-1, :-2, 1]) * (rho[..., 1:-1, 2:, 1] - rho[..., 1:-1, :-2, 1])
        )
        rho[..., 1:-1, 1:-1, 0] = torch.where(
            flags[..., 1:-1, 1:-1, 0] == int(CellType.OBSTACLE),
            rho[..., 1:-1, 1:-1, 2] + torch.tan(
                math.pi / 2.0 - self._contact_angle
            ) * hlp_CA,
            rho[..., 1:-1, 1:-1, 0]
        )
        # 2. pos x
        hlp_CA = torch.sqrt(
            eps +
            (rho[..., 2:, 1:-1, -2] - rho[..., :-2, 1:-1, -2]) * (rho[..., 2:, 1:-1, -2] - rho[..., :-2, 1:-1, -2]) +
            (rho[..., 1:-1, 2:, -2] - rho[..., 1:-1, :-2, -2]) * (rho[..., 1:-1, 2:, -2] - rho[..., 1:-1, :-2, -2])
        )
        rho[..., 1:-1, 1:-1, -1] = torch.where(
            flags[..., 1:-1, 1:-1, -1] == int(CellType.OBSTACLE),
            rho[..., 1:-1, 1:-1, -3] + torch.tan(
                math.pi / 2.0 - self._contact_angle
            ) * hlp_CA,
            rho[..., 1:-1, 1:-1, -1]
        )
        # 3. neg y
        hlp_CA = torch.sqrt(
            eps +
            (rho[..., 2:, 1, 1:-1] - rho[..., :-2, 1, 1:-1]) * (rho[..., 2:, 1, 1:-1] - rho[..., :-2, 1, 1:-1]) +
            (rho[..., 1:-1, 1, 2:] - rho[..., 1:-1, 1, :-2]) * (rho[..., 1:-1, 1, 2:] - rho[..., 1:-1, 1, :-2])
        )
        rho[..., 1:-1, 0, 1:-1] = torch.where(
            flags[..., 1:-1, 0, 1:-1] == int(CellType.OBSTACLE),
            rho[..., 1:-1, 2, 1:-1] + torch.tan(
                math.pi / 2.0 - self._contact_angle
            ) * hlp_CA,
            rho[..., 1:-1, 0, 1:-1]
        )
        # 4. pos y
        hlp_CA = torch.sqrt(
            eps +
            (rho[..., 2:, -2, 1:-1] - rho[..., :-2, -2, 1:-1]) * (rho[..., 2:, -2, 1:-1] - rho[..., :-2, -2, 1:-1]) +
            (rho[..., 1:-1, -2, 2:] - rho[..., 1:-1, -2, :-2]) * (rho[..., 1:-1, -2, 2:] - rho[..., 1:-1, -2, :-2])
        )
        rho[..., 1:-1, -1, 1:-1] = torch.where(
            flags[..., 1:-1, -1, 1:-1] == int(CellType.OBSTACLE),
            rho[..., 1:-1, -3, 1:-1] + torch.tan(
                math.pi / 2.0 - self._contact_angle
            ) * hlp_CA,
            rho[..., 1:-1, -1, 1:-1]
        )
        # 5. neg z
        hlp_CA = torch.sqrt(
            eps +
            (rho[..., 1, 2:, 1:-1] - rho[..., 1, :-2, 1:-1]) * (rho[..., 1, 2:, 1:-1] - rho[..., 1, :-2, 1:-1]) +
            (rho[..., 1, 1:-1, 2:] - rho[..., 1, 1:-1, :-2]) * (rho[..., 1, 1:-1, 2:] - rho[..., 1, 1:-1, :-2])
        )
        rho[..., 0, 1:-1, 1:-1] = torch.where(
            flags[..., 0, 1:-1, 1:-1] == int(CellType.OBSTACLE),
            rho[..., 2, 1:-1, 1:-1],
            rho[..., 0, 1:-1, 1:-1]
        )
        # 6. pos z
        hlp_CA = torch.sqrt(
            eps +
            (rho[..., -2, 2:, 1:-1] - rho[..., -2, :-2, 1:-1]) * (rho[..., -2, 2:, 1:-1] - rho[..., -2, :-2, 1:-1]) +
            (rho[..., -2, 1:-1, 2:] - rho[..., -2, 1:-1, :-2]) * (rho[..., -2, 1:-1, 2:] - rho[..., -2, 1:-1, :-2])
        )
        rho[..., -1, 1:-1, 1:-1] = torch.where(
            flags[..., -1, 1:-1, 1:-1] == int(CellType.OBSTACLE),
            rho[..., -3, 1:-1, 1:-1],
            rho[..., -1, 1:-1, 1:-1]
        )

        # 7. edge lines (12)
        rho[..., 1:-1, 0, 0] = 0.5 * (rho[..., 1:-1, 0, 1] + rho[..., 1:-1, 1, 0])
        rho[..., 1:-1, 0, -1] = 0.5 * (rho[..., 1:-1, 0, -2] + rho[..., 1:-1, 1, -1])
        rho[..., 1:-1, -1, 0] = 0.5 * (rho[..., 1:-1, -1, 1] + rho[..., 1:-1, -2, 0])
        rho[..., 1:-1, -1, -1] = 0.5 * (rho[..., 1:-1, -1, -2] + rho[..., 1:-1, -2, -1])

        rho[..., 0, 1:-1, 0] = 0.5 * (rho[..., 0, 1:-1, 1] + rho[..., 1, 1:-1, 0])
        rho[..., 0, 1:-1, -1] = 0.5 * (rho[..., 0, 1:-1, -2] + rho[..., 1, 1:-1, -1])
        rho[..., -1, 1:-1, 0] = 0.5 * (rho[..., -1, 1:-1, 1] + rho[..., -2, 1:-1, 0])
        rho[..., -1, 1:-1, -1] = 0.5 * (rho[..., -1, 1:-1, -2] + rho[..., -2, 1:-1, -1])

        rho[..., 0, 0, 1:-1] = 0.5 * (rho[..., 0, 1, 1:-1] + rho[..., 1, 0, 1:-1])
        rho[..., 0, -1, 1:-1] = 0.5 * (rho[..., 0, -2, 1:-1] + rho[..., 1, -1, 1:-1])
        rho[..., -1, 0, 1:-1] = 0.5 * (rho[..., -1, 1, 1:-1] + rho[..., -2, 0, 1:-1])
        rho[..., -1, -1, 1:-1] = 0.5 * (rho[..., -1, -2, 1:-1] + rho[..., -2, -1, 1:-1])

        # 8. edge points (8)
        rho[..., 0, 0, 0] = (rho[..., 0, 0, 1] + rho[..., 0, 1, 0] + rho[..., 1, 0, 0]) / 3.0
        rho[..., 0, 0, -1] = (rho[..., 0, 0, -2] + rho[..., 0, 1, -1] + rho[..., 1, 0, -1]) / 3.0
        rho[..., 0, -1, 0] = (rho[..., 0, -1, 1] + rho[..., 0, -2, 0] + rho[..., 1, -1, 0]) / 3.0
        rho[..., 0, -1, -1] = (rho[..., 0, -1, -2] + rho[..., 0, -2, -1] + rho[..., 1, -1, -1]) / 3.0

        rho[..., -1, 0, 0] = (rho[..., -1, 0, 1] + rho[..., -1, 1, 0] + rho[..., -2, 0, 0]) / 3.0
        rho[..., -1, 0, -1] = (rho[..., -1, 0, -2] + rho[..., -1, 1, -1] + rho[..., -2, 0, -1]) / 3.0
        rho[..., -1, -1, 0] = (rho[..., -1, -1, 1] + rho[..., -1, -2, 0] + rho[..., -2, -1, 0]) / 3.0
        rho[..., -1, -1, -1] = (rho[..., -1, -1, -2] + rho[..., -1, -2, -1] + rho[..., -2, -1, -1]) / 3.0

        density = self._density_gas + (self._density_liquid - self._density_gas) * (
            (rho - self._rho_gas) / (self._rho_liquid - self._rho_gas)
        )

        # ===========================
        #      Laplacian of density
        # ===========================
        laplacian_density = self.get_laplacian(input_=density, dx=dx)

        # ===========================
        #      Get your forces
        # ===========================
        force = self._kappa * density * LBMCollisionHCZ3d.get_grad(input_=laplacian_density, dx=dx)
        force += self._gravity * density
        if H2 is not None:
            mu0 = 4 * math.pi * 1e-7
            k = 0.33
            chi = k * (self.smooth_phi(phi=phi, eps=0.1 * dx))
            force += -0.5 * mu0 * H2 * LBMCollisionHCZ3d.get_grad(input_=chi, dx=dx)
        dfai = LBMCollisionHCZ3d.get_grad(input_=fai, dx=dx)
        dprho = LBMCollisionHCZ3d.get_grad(input_=prho, dx=dx)

        # ===========================
        #      Get your real macro-varaibles (besides from velocity)
        # ===========================
        macro_vel = (
            (
                (g.unsqueeze(2) * self._e).sum(dim=1) * c +
                0.5 * dt * RT * force
            )
        ) / RT / density  # [B, dim, res]
        vel = torch.where(
            (flags == int(CellType.FLUID)).repeat(1, dim, *([1] * dim)),
            macro_vel,
            vel
        )

        macro_pressure = (
            g.sum(dim=1).unsqueeze(1) -
            0.5 * dt * (vel * dprho).sum(dim=1).unsqueeze(1)
        )  # [B, 1, res]
        pressure = torch.where(
            flags == int(CellType.FLUID),
            macro_pressure,
            pressure
        )

        return [rho, vel, density, pressure, force, dfai, dprho]
    
    def smooth_phi(self, phi: torch.Tensor, eps: float) -> torch.Tensor:
        result = (phi > eps) * 1.0 + (torch.abs(phi) <= eps) * \
                  (0.5 +
                   (0.5 / eps) * phi +
                   (0.5 / np.pi) * torch.sin((np.pi / eps) * phi)
                   )
        return result
    
    def get_laplacian(self, input_: torch.Tensor, dx: float) -> torch.Tensor:
        output_ = F.pad(
            (
                2.0 * (
                    input_[..., 1:-1, 1:-1, 2:] + input_[..., 1:-1, 1:-1, :-2] +
                    input_[..., 1:-1, 2:, 1:-1] + input_[..., 1:-1, :-2, 1:-1] +
                    input_[..., 2:, 1:-1, 1:-1] + input_[..., :-2, 1:-1, 1:-1]
                ) + 
                (
                    input_[..., 1:-1, 2:, 2:] + input_[..., 1:-1, 2:, :-2] +
                    input_[..., 1:-1, :-2, 2:] + input_[..., 1:-1, :-2, :-2] +
                    input_[..., 2:, 1:-1, 2:] + input_[..., 2:, 1:-1, :-2] +
                    input_[..., :-2, 1:-1, 2:] + input_[..., :-2, 1:-1, :-2] +
                    input_[..., 2:, 2:, 1:-1] + input_[..., 2:, :-2, 1:-1] +
                    input_[..., :-2, 2:, 1:-1] + input_[..., :-2, :-2, 1:-1]
                ) -
                (
                    24 * input_[..., 1:-1, 1:-1, 1:-1]
                )
            ) / 6.0 / (dx * dx),
            pad=(1, 1, 1, 1, 1, 1),
            mode="constant",
            value=0
        )

        return output_
    
    @staticmethod
    def get_grad(input_: torch.Tensor, dx: float) -> torch.Tensor:
        if input_.shape[1] != 1:
            raise RuntimeError("To get your grad operation, channel dim has to be 1")
        
        dim = 3
        pad = (1, 1, 1, 1, 1, 1)

        output = torch.zeros_like(input_[..., 1:-1, 1:-1, 1:-1]).repeat(1, 3, 1, 1, 1)
        output[:, 0:1, ...] = (
            2.0 * (
                input_[..., 1:-1, 1:-1, 2:] - input_[..., 1:-1, 1:-1, :-2]
            ) +
            (
                input_[..., 2:, 1:-1, 2:] - input_[..., :-2, 1:-1, :-2] +
                input_[..., :-2, 1:-1, 2:] - input_[..., 2:, 1:-1, :-2] +
                input_[..., 1:-1, 2:, 2:] - input_[..., 1:-1, :-2, :-2] +
                input_[..., 1:-1, :-2, 2:] - input_[..., 1:-1, 2:, :-2]
            )
        ) / 12.0 / dx

        output[:, 1:2, ...] = (
            2.0 * (
                input_[..., 1:-1, 2:, 1:-1] - input_[..., 1:-1, :-2, 1:-1]
            ) +
            (
                input_[..., 2:, 2:, 1:-1] - input_[..., :-2, :-2, 1:-1] +
                input_[..., :-2, 2:, 1:-1] - input_[..., 2:, :-2, 1:-1] +
                input_[..., 1:-1, 2:, 2:] - input_[..., 1:-1, :-2, :-2] +
                input_[..., 1:-1, 2:, :-2] - input_[..., 1:-1, :-2, 2:]
            )
        ) / 12.0 / dx

        output[:, 2:3, ...] = (
            2.0 * (
                input_[..., 2:, 1:-1, 1:-1] - input_[..., :-2, 1:-1, 1:-1]
            ) +
            (
                input_[..., 2:, 2:, 1:-1] - input_[..., :-2, :-2, 1:-1] +
                input_[..., 2:, :-2, 1:-1] - input_[..., :-2, 2:, 1:-1] +
                input_[..., 2:, 1:-1, 2:] - input_[..., :-2, 1:-1, :-2] +
                input_[..., 2:, 1:-1, :-2] - input_[..., :-2, 1:-1, 2:]
            )
        ) / 12.0 / dx

        output_pad = F.pad(
            output, pad=pad, mode="replicate"
        )

        return output_pad
    
    def compute_Gamma(self, dx: float, dt: float, vel: torch.Tensor):
        c = dx / dt
        cs2 = c * c / 3.0

        uv = (vel * vel).sum(dim=1).unsqueeze(1)  # [B, 1, res]
        eu = (vel.unsqueeze(1) * self._e * c).sum(dim=2)  # [B, Q, res]
        Gamma = self._weight * (
            1.0 + eu / cs2 + 0.5 * eu * eu / cs2 / cs2 - 0.5 * uv / cs2
        )

        return Gamma

    def collision(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor,
        g: torch.Tensor =  None,
        pressure: torch.Tensor = None,
        dfai: torch.Tensor = None,
        dprho: torch.Tensor =  None,
        KBC_case: str = None
    ) -> List[torch.Tensor]:
        """
        Args:
            f (torch.Tensor): f before streaming [B, Q, res]
            rho (torch.Tensor): density [B, 1, res]
            vel (torch.Tensor): velocity [B, dim, res]
            flags (torch.Tensor): flags [B, 1, res]
            force (torch.Tensor): force [B, dim, res]
            KBC_case: str = [None, 'A', 'B', 'C', 'D'], where None is LBGK case, 'A/B/C/D' is different KBC cases

        Returns:
            List[torch.Tensor]: f,g after streaming [B, Q, res]
        """
        dim = 2
        tau_f = self._tau_f
        tau_g = self._tau_g
        c = dx / dt
        cs2 = c * c / 3.0
        RT = cs2

        feq = self.get_feq_(dx=dx, dt=dt, rho=rho, vel=vel, force=None)
        geq = self.get_geq_(dx=dx, dt=dt, rho=rho, vel=vel, force=None, pressure=pressure, feq=feq)

        Gamma_u = self.compute_Gamma(dx=dx, dt=dt, vel=vel)

        # collision_f = (1.0 - 1.0 / tau_f) * f + feq / tau_f
        collision_f = dt * (1.0 - 0.5 / tau_f) * Gamma_u / RT * (
            (self._e * c - vel.unsqueeze(1)) * (-dfai.unsqueeze(1))
        ).sum(dim=2) * dt + (1.0 - 1.0 / tau_f) * f + feq / tau_f

        collision_g = dt * (1.0 - 0.5 / tau_g) * (
            Gamma_u * (
                (self._e * c - vel.unsqueeze(1)) * (force.unsqueeze(1))
            ).sum(dim=2) + \
            (Gamma_u - self._weight) * (
                (self._e * c - vel.unsqueeze(1)) * (-dprho.unsqueeze(1))
            ).sum(dim=2)
        ) * dt + (1.0 - 1.0 / tau_g) * g + geq / tau_g

        f_new = torch.where(
            flags == int(CellType.FLUID),
            collision_f,
            f
        )

        g_new = torch.where(
            flags == int(CellType.FLUID),
            collision_g,
            g
        )

        return [f_new, g_new]
