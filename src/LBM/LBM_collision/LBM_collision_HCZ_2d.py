import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from src.LBM.LBM_collision import LBMCollisionMRT2d
from src.LBM.utils import CellType, KBCType


class LBMCollisionHCZ2d(LBMCollisionMRT2d):
    rank = 2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(LBMCollisionHCZ2d, self).__init__(*args, **kwargs)

    def capillary_process(
        self,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor,
        dt: float = 1.0,
        dx: float = 1.0,
        g: torch.Tensor = None,
        density: torch.Tensor = None,
        pressure: torch.Tensor = None,
        H2: torch.Tensor = None,
        phi: torch.Tensor = None,
    ):
        dim = 2
        pad = (1, 1, 1, 1)

        c = dx / dt
        cs2 = c * c / 3.0
        RT = cs2

        # Choose your mode for different boundaries
        # prho = torch.where(
        #     flags == int(CellType.OBSTACLE),
        #     F.pad(
        #         pressure[..., 1:-1, 1:-1] - RT * density[..., 1:-1, 1:-1],
        #         pad=pad,
        #         mode="constant",
        #         value=0
        #     ),
        #     F.pad(
        #         pressure[..., 1:-1, 1:-1],
        #         pad=pad,
        #         mode="replicate"
        #     ) - RT * density,
        # )
        # fai = torch.where(
        #     flags == int(CellType.OBSTACLE),
        #     F.pad(
        #         self.equation_of_states(dx=dx, dt=dt, rho=rho)[..., 1:-1, 1:-1] - RT * density[..., 1:-1, 1:-1],
        #         pad=pad,
        #         mode="constant",
        #         value=0
        #     ),
        #     F.pad(
        #         self.equation_of_states(dx=dx, dt=dt, rho=rho)[..., 1:-1, 1:-1],
        #         pad=pad,
        #         mode="replicate"
        #     ) - RT * density,
        # )

        prho = F.pad(
            (pressure - RT * density)[..., 1:-1, 1:-1], pad=pad, mode="replicate"
        )
        fai = F.pad(
            (self.equation_of_states(dx=dx, dt=dt, rho=rho) - rho * RT)[
                ..., 1:-1, 1:-1
            ],
            pad=pad,
            mode="replicate",
        )

        # ===========================
        #      Contact Angle
        # ===========================
        # 1. neg x
        hlp_CA = torch.abs(rho[..., 2:, 1] - rho[..., :-2, 1])
        rho[..., 1:-1, 0] = torch.where(
            flags[..., 1:-1, 0] == int(CellType.OBSTACLE),
            rho[..., 1:-1, 2] + torch.tan(math.pi / 2.0 - self._contact_angle) * hlp_CA,
            rho[..., 1:-1, 0],
        )
        # 2. pos x
        hlp_CA = torch.abs(rho[..., 2:, -2] - rho[..., :-2, -2])
        rho[..., 1:-1, -1] = torch.where(
            flags[..., 1:-1, -1] == int(CellType.OBSTACLE),
            rho[..., 1:-1, -3]
            + torch.tan(math.pi / 2.0 - self._contact_angle) * hlp_CA,
            rho[..., 1:-1, -1],
        )
        # 3. neg y
        hlp_CA = torch.abs(rho[..., 1, 2:] - rho[..., 1, :-2])
        rho[..., 0, 1:-1] = torch.where(
            flags[..., 0, 1:-1] == int(CellType.OBSTACLE),
            rho[..., 2, 1:-1] + torch.tan(math.pi / 2.0 - self._contact_angle) * hlp_CA,
            rho[..., 0, 1:-1],
        )
        # 4.pos y
        hlp_CA = torch.abs(rho[..., -2, 2:] - rho[..., -2, :-2])
        rho[..., -1, 1:-1] = torch.where(
            flags[..., -1, 1:-1] == int(CellType.OBSTACLE),
            rho[..., -3, 1:-1]
            + torch.tan(math.pi / 2.0 - self._contact_angle) * hlp_CA,
            rho[..., -1, 1:-1],
        )

        # 5. edge points
        rho[..., 0, 0] = 0.5 * (rho[..., 1, 0] + rho[..., 0, 1])
        rho[..., -1, 0] = 0.5 * (rho[..., -2, 0] + rho[..., -1, 1])
        rho[..., 0, -1] = 0.5 * (rho[..., 0, -2] + rho[..., 1, -1])
        rho[..., -1, -1] = 0.5 * (rho[..., -2, -2] + rho[..., -2, -2])

        density = self._density_gas + (self._density_liquid - self._density_gas) * (
            (rho - self._rho_gas) / (self._rho_liquid - self._rho_gas)
        )

        # ===========================
        #      Laplacian of density
        # ===========================
        laplacian_density = self.get_laplacian(input_=density, dx=dx, flags=flags)

        # ===========================
        #      Get your forces
        # ===========================
        force = (
            self._kappa
            * density
            * LBMCollisionHCZ2d.get_grad(input_=laplacian_density, dx=dx, flags=flags)
        )
        force += self._gravity * density
        if H2 is not None:
            mu0 = 4 * math.pi * 1e-7
            k = 0.33
            chi = k * (1.0 - self.smooth_phi(phi=phi, eps=0.1 * dx))
            force += (
                -0.5
                * mu0
                * H2
                * LBMCollisionHCZ2d.get_grad(input_=chi, dx=dx, flags=flags)
            )
        dfai = LBMCollisionHCZ2d.get_grad(input_=fai, dx=dx, flags=flags)
        dprho = LBMCollisionHCZ2d.get_grad(input_=prho, dx=dx, flags=flags)

        # ===========================
        #      Get your real macro-varaibles (besides from velocity)
        # ===========================
        macro_vel = (
            (((g.unsqueeze(2) * self._e).sum(dim=1) * c + 0.5 * dt * RT * force))
            / RT
            / density
        )  # [B, dim, res]
        vel = torch.where(
            (flags == int(CellType.FLUID)).repeat(1, dim, *([1] * dim)), macro_vel, vel
        )

        macro_pressure = g.sum(dim=1).unsqueeze(1) - 0.5 * dt * (vel * dprho).sum(
            dim=1
        ).unsqueeze(
            1
        )  # [B, 1, res]
        pressure = torch.where(flags == int(CellType.FLUID), macro_pressure, pressure)

        return [rho, vel, density, pressure, force, dfai, dprho]

    def smooth_phi(self, phi: torch.Tensor, eps: float) -> torch.Tensor:
        result = (phi > eps) * 1.0 + (torch.abs(phi) <= eps) * (
            0.5 + (0.5 / eps) * phi + (0.5 / np.pi) * torch.sin((np.pi / eps) * phi)
        )
        return result

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
        density: torch.Tensor,
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
            f (torch.Tensor): f before streaming [B, Q, res]
            rho (torch.Tensor): density [B, 1, res]
            vel (torch.Tensor): velocity [B, dim, res]
            density (torch.Tensor): density [B, 1, res]
            flags (torch.Tensor): flags [B, 1, res]
            force (torch.Tensor): force [B, dim, res]
            KBC_type: int = [None, 'A', 'B', 'C', 'D'], where None is LBGK case, 'A/B/C/D' is different KBC cases

        Returns:
            torch.Tensor: f after streaming [B, Q, res]
        """
        tau_f = self._tau_f
        tau_g = self._tau_g
        c = dx / dt
        cs2 = c * c / 3.0
        RT = cs2

        feq = self.get_feq_(dx=dx, dt=dt, rho=rho, vel=vel, force=None)
        geq = self.get_geq_(
            dx=dx,
            dt=dt,
            rho=rho,
            vel=vel,
            density=density,
            force=None,
            pressure=pressure,
            feq=feq,
        )

        Gamma_u = self.compute_Gamma(dx=dx, dt=dt, vel=vel)

        collision_g = g + 1.0 / tau_g * (geq - g)

        if KBC_type is not None:
            ds = self.get_s_by_KBC(
                dx=dx, dt=dt, f=g, rho=rho, vel=vel, KBC_type=KBC_type
            ) - self.get_s_by_KBC(
                dx=dx, dt=dt, f=geq, rho=rho, vel=vel, KBC_type=KBC_type
            )
            dh = (g - geq) - ds

            beta = 0.5 / tau_g
            gamma = 1.0 / beta - (2.0 - 1.0 / beta) * (ds * dh / geq).sum(
                dim=1
            ).unsqueeze(1) / (dh * dh / geq).sum(dim=1).unsqueeze(1)
            collision_g = g + beta * (-2.0 * ds - gamma * dh)

        collision_f = (
            f
            + 1.0 / tau_f * (feq - f)
            + (
                dt
                * (1.0 - 0.5 / tau_f)
                * Gamma_u
                / RT
                * ((self._e * c - vel.unsqueeze(1)) * (-dfai.unsqueeze(1))).sum(dim=2)
                * dt
            )
        )

        collision_g = collision_g + (
            (1.0 - 0.5 / tau_g)
            * (
                Gamma_u
                * ((self._e * c - vel.unsqueeze(1)) * (force.unsqueeze(1))).sum(dim=2)
                + (Gamma_u - self._weight)
                * ((self._e * c - vel.unsqueeze(1)) * (-dprho.unsqueeze(1))).sum(dim=2)
            )
            * dt
        )

        f_new = torch.where(flags == int(CellType.FLUID), collision_f, f)

        g_new = torch.where(flags == int(CellType.FLUID), collision_g, g)

        return [f_new, g_new]
