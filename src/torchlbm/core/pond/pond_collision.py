"""PonD BGK collision for the two-population (f, g) model.

Purely local: each node relaxes toward its co-moving equilibria, so no gauge
transform is needed here. The relaxation rates carry dt so the realised viscosity
stays fixed even though PonD re-chooses dt every step. Solid cells are left frozen.
"""
from typing import List

import torch
import torch.nn as nn

from torchlbm.compressible_node_data import CompressibleNodeData
from torchlbm.core.pond.pond_equilibrium import PondEquilibrium
from torchlbm.core.pond.pond_lattice_gauge import POND_LATTICE_TEMPERATURE


class PondCollisionModule(nn.Module):
    """BGK relaxation of f (momentum) and g (energy) toward equilibrium."""

    def __init__(
        self,
        lattice_velocities: List[List[float]],
        lattice_weights: List[float],
        viscosity: float,
        cp: float,
        cv: float,
        thermal_conductivity: float,
        dimension: int,
        lattice_temperature: float = POND_LATTICE_TEMPERATURE,
    ) -> None:
        super(PondCollisionModule, self).__init__()
        self.viscosity = viscosity
        self.cp = cp
        self.thermal_conductivity = thermal_conductivity
        self.equilibrium = PondEquilibrium(
            lattice_velocities, lattice_weights, cv=cv, dimension=dimension,
            lattice_temperature=lattice_temperature,
        )

    def relaxation_frequencies(self, rho, temperature, dt):
        # Chapman-Enskog with dt folded in => omega = 2 T dt / (2 nu + T dt), so
        # nu_eff = nu regardless of the (per-step) dt. At Pr = 1, omega_g == omega_f.
        t_dt = temperature * dt
        nu = self.viscosity / rho                            # kinematic viscosity (mu is dynamic)
        alpha = self.thermal_conductivity / (self.cp * rho)  # thermal diffusivity
        omega_f = 2.0 * t_dt / (2.0 * nu + t_dt)
        omega_g = 2.0 * t_dt / (2.0 * alpha + t_dt)
        return omega_f, omega_g

    def forward(self, node_data: CompressibleNodeData, dt: float = 1.0) -> CompressibleNodeData:
        rho = node_data.moments.density
        temperature = node_data.moments.temperature

        # Co-moving equilibria (also stashed as the "new" populations for the advection step).
        f_eq, g_eq = self.equilibrium.equilibria(rho, temperature)
        node_data.distributions.vel_new_population = f_eq
        node_data.distributions.temp_new_population = g_eq

        omega_vel, omega_temp = self.relaxation_frequencies(rho, temperature, dt)
        omega_vel = omega_vel.unsqueeze(0)
        omega_temp = omega_temp.unsqueeze(0)

        # f* = f + omega (f_eq - f), likewise for g.
        f = node_data.distributions.vel_old_population
        g = node_data.distributions.temp_old_population
        f_star = f + omega_vel * (f_eq - f)
        g_star = g + omega_temp * (g_eq - g)

        # Solid (bounce-back) nodes do not collide.
        if node_data.bounce_back_mask is not None:
            solid = (node_data.bounce_back_mask > 0).unsqueeze(0)
            f_star = torch.where(solid, f, f_star)
            g_star = torch.where(solid, g, g_star)

        node_data.distributions.vel_old_population = f_star
        node_data.distributions.temp_old_population = g_star
        return node_data
