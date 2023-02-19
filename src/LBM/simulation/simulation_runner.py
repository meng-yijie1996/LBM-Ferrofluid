from src.LBM.LBM_propagation import (
    LBMPropagation2d,
    LBMPropagation3d
)

from src.LBM.LBM_macro_compute import (
    LBMMacroCompute2d,
    LBMMacroCompute3d
)

from src.LBM.LBM_collision import (
    LBMCollision2d,
    LBMCollision3d,
    LBMCollisionMRT2d,
    LBMCollisionHCZ2d,
    LBMCollisionHCZ3d
)

from src.LBM.LBM_magnetic import (
    LBMMagnetic2d,
    LBMMagnetic3d,
)

from src.LBM.simulation import SimulationParameters


class SimulationRunner(object):
    def __init__(
            self,
            parameters: SimulationParameters,
    ):
        self.parameters = parameters

    def create_propagation(self):
        if self.parameters.is_2d():
            return LBMPropagation2d()
        else:
            return LBMPropagation3d()

    def create_macro_compute(self):
        if self.parameters.is_2d():
            return LBMMacroCompute2d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            return LBMMacroCompute3d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
    
    def create_collision(self):
        if self.parameters.is_2d():
            return LBMCollision2d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                kappa=self.parameters.kappa,
                tau_f=self.parameters.tau_f,
                tau_g=self.parameters.tau_g,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            return LBMCollision3d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                kappa=self.parameters.kappa,
                tau_f=self.parameters.tau_f,
                tau_g=self.parameters.tau_g,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
    
    def create_collision_MRT(self):
        if self.parameters.is_2d():
            return LBMCollisionMRT2d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                kappa=self.parameters.kappa,
                tau_f=self.parameters.tau_f,
                tau_g=self.parameters.tau_g,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            pass
    
    def create_collision_HCZ(self):
        if self.parameters.is_2d():
            return LBMCollisionHCZ2d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                kappa=self.parameters.kappa,
                tau_f=self.parameters.tau_f,
                tau_g=self.parameters.tau_g,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            return LBMCollisionHCZ3d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                kappa=self.parameters.kappa,
                tau_f=self.parameters.tau_f,
                tau_g=self.parameters.tau_g,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
    
    def create_LBM_magnetic(self):
        if self.parameters.is_2d():
            return LBMMagnetic2d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                kappa=self.parameters.kappa,
                tau_f=self.parameters.tau_f,
                tau_g=self.parameters.tau_g,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            return LBMMagnetic3d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                kappa=self.parameters.kappa,
                tau_f=self.parameters.tau_f,
                tau_g=self.parameters.tau_g,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
    
    def step(self):
        self.parameters.step()
