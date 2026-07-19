"""PonD gauge: the co-moving reference frame lambda = {u, T} and its moment closure.

The lattice velocities c_i are fixed, but PonD rescales+shifts them per node into
"peculiar" velocities v_i = sqrt(theta) c_i + u (theta = T/T_L). Moments of the
f-populations in that frame recover (rho, u, T).
"""
from typing import List, Tuple

import torch
import torch.nn as nn

POND_LATTICE_TEMPERATURE = 1.0   # T_L: 1.0 for D2Q16, 1/3 for D2Q9T


class PondLatticeGauge(nn.Module):
    """Holds the fixed lattice (c_i, W_i) and builds the co-moving v_i / moments."""

    def __init__(
        self,
        lattice_velocities: List[List[float]],
        lattice_weights: List[float],
        dimension: int,
        lattice_temperature: float = POND_LATTICE_TEMPERATURE,
        density_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.lattice_temperature = lattice_temperature
        self.density_floor = density_floor
        self.register_buffer("c", torch.tensor(lattice_velocities, dtype=torch.get_default_dtype()))
        self.register_buffer("w", torch.tensor(lattice_weights, dtype=torch.get_default_dtype()))

    def theta(self, temperature: torch.Tensor) -> torch.Tensor:
        return temperature / self.lattice_temperature   # theta = T / T_L

    def peculiar_velocities(self, temperature: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # v_i = sqrt(theta) * c_i + u  (tailored, co-moving velocity set)
        a = torch.sqrt(self.theta(temperature))
        v = self.c[:, :, None, None, None] * a[None, None] + u[:, None]
        return v

    def departure_offsets(
        self, temperature: torch.Tensor, u: torch.Tensor, lattice_distance: float = 1.0, dt: float = 1.0
    ) -> torch.Tensor:
        # Semi-Lagrangian shift v_i * dt/dx (where each population came from last step).
        return self.peculiar_velocities(temperature, u) * (dt / lattice_distance)

    def gauge_moments(
        self, f: torch.Tensor, temperature: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Recover (rho, u, T) from the f-populations expressed in frame {u, T}.
        v = self.peculiar_velocities(temperature, u)
        rho = torch.sum(f, dim=0)                                  # rho = sum_i f_i
        rho = rho.clamp(min=self.density_floor)                   # guard 1/rho at strong shocks
        momentum = torch.einsum("dQxyz,Qxyz->dxyz", v, f)         # rho u = sum_i f_i v_i
        u_new = momentum / rho
        v_sq = torch.sum(v * v, dim=0)
        second = torch.einsum("Qxyz,Qxyz->xyz", v_sq, f)          # sum_i f_i |v_i|^2
        u_new_sq = torch.sum(u_new * u_new, dim=0)
        t_new = (second / rho - u_new_sq) / self.dimension        # T = (<|v|^2> - |u|^2) / D
        return rho, u_new, t_new
