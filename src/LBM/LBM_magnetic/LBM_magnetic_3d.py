import math
import numpy as np
from typing import List
import torch
import torch.nn.functional as F

from src.LBM.LBM_magnetic import AbstractLBMMagnetic
from src.LBM.LBM_collision import LBMCollisionHCZ3d
from src.LBM.utils import CellType, get_staggered_x, get_staggered_y, get_staggered_z


class LBMMagnetic3d(AbstractLBMMagnetic):
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
        super(LBMMagnetic3d, self).__init__(*args, **kwargs)
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
    
    def get_heq_(self, psi: torch.Tensor) -> torch.Tensor:
        """
        poisson equation solver requires a different feq solver,
        which can be referred by Chai et al, 2007. (e.q. 2.2)
        https://www.sciencedirect.com/science/article/pii/S0307904X07001722
        """
        Q = self._Q

        heq = psi * self._weight  # [B, Q, *res]
        heq[:, 0:1, ...] = heq[:, 0:1, ...] - psi

        return heq
    
    def smooth_phi(self, phi: torch.Tensor, eps: float) -> torch.Tensor:
        result = (phi > eps) * 1.0 + (torch.abs(phi) <= eps) * \
                  (0.5 +
                   (0.5 / eps) * phi +
                   (0.5 / np.pi) * torch.sin((np.pi / eps) * phi)
                   )
        return result
    
    def get_H_int(
        self,
        dt: float,
        dx: float,
        phi: torch.Tensor,
        flags: torch.Tensor,
        H_ext_mac: List[torch.Tensor],
        h: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        A poisson equation will be solved carefully.
        The solver pipeline is referred by Chai et al, 2007.
        https://www.sciencedirect.com/science/article/pii/S0307904X07001722

        This is not a N-S solver and thus no advection,
        which implies a streaming step is dropped.
        Args:
            H_ext (Staggered3dGrid): input H_ext [B, dim, res]
            h (torch.Tensor): [B, Q, res]

        Returns:
            torch.Tensor: induced H_int: [B, dim, res]
        """
        dim = 3

        # macro variables computing
        tau = self._tau
        weight_bar = self._weight
        c = dx / dt
        cs2 = c * c / 3.0
        k = 0.33

        psi = h[:, 1:, ...].sum(dim=1).unsqueeze(1) / (
            1.0 - self._weight[:, 0:1, ...]
        )  # [B, 1, *res]
        
        H_int = -LBMCollisionHCZ3d.get_grad(input_=psi, dx=dx)

        # collision step
        heq = self.get_heq_(psi=psi)
        phi_mac_x, phi_mac_y, phi_mac_z = get_staggered_x(phi), get_staggered_y(phi), get_staggered_z(phi)
        H_ext_mac_x, H_ext_mac_y, H_ext_mac_z = H_ext_mac
        chi_mac_x = k * (self.smooth_phi(phi=phi_mac_x, eps=0.1 * dx))
        chi_mac_y = k * (self.smooth_phi(phi=phi_mac_y, eps=0.1 * dx))
        chi_mac_z = k * (self.smooth_phi(phi=phi_mac_z, eps=0.1 * dx))
        chi_H_ext_mac_x = chi_mac_x * H_ext_mac_x
        chi_H_ext_mac_y = chi_mac_y * H_ext_mac_y
        chi_H_ext_mac_z = chi_mac_z * H_ext_mac_z
        rhs = (
            (chi_H_ext_mac_x[..., 1:] - chi_H_ext_mac_x[..., :-1]) +
            (chi_H_ext_mac_y[..., 1:, :] - chi_H_ext_mac_y[..., :-1, :]) +
            (chi_H_ext_mac_z[..., 1:, :, :] - chi_H_ext_mac_z[..., :-1, :, :])
        ) / dx
        # only count where there is fluid
        rhs = torch.where(
            flags == int(CellType.FLUID), rhs, torch.zeros_like(rhs)
        )
        add_h = dt * weight_bar * rhs * (cs2 * (0.5 - tau) * dt)
        new_h = (1.0 - 1.0 / tau) * h + (1.0 / tau) * heq + add_h

        H_int[(flags == int(CellType.OBSTACLE)).repeat(1, dim, *([1] * dim))] = 0

        return [H_int, new_h]