"""PonD two-population equilibria in the co-moving gauge (trivial, u-independent).

Because the frame already carries u (baked into v_i), the equilibria are algebraic:
    f_eq_i = rho * W_i                       (mass/momentum)
    g_eq_i = (2 Cv - D) * T * f_eq_i         (energy)
Note g_eq vanishes when Cv = D/2, i.e. gamma = (D+2)/D = 2 -- the "gamma trap".
"""
from typing import List, Tuple

import torch
import torch.nn as nn

from torchlbm.core.pond.pond_lattice_gauge import POND_LATTICE_TEMPERATURE


class PondEquilibrium(nn.Module):
    """Density-only f equilibrium and energy g equilibrium in the co-moving gauge."""

    def __init__(
        self,
        lattice_velocities: List[List[float]],
        lattice_weights: List[float],
        cv: float,
        dimension: int,
        lattice_temperature: float = POND_LATTICE_TEMPERATURE,
    ) -> None:
        super(PondEquilibrium, self).__init__()
        self.cv = cv
        self.dimension = dimension
        self.lattice_temperature = lattice_temperature
        self.register_buffer("c", torch.tensor(lattice_velocities, dtype=torch.get_default_dtype()))
        self.register_buffer("w", torch.tensor(lattice_weights, dtype=torch.get_default_dtype()))

    def f_equilibrium(self, density: torch.Tensor) -> torch.Tensor:
        return self.w[:, None, None, None] * density[None]        # f_eq_i = rho W_i

    def g_equilibrium(
        self, density: torch.Tensor, temperature: torch.Tensor
    ) -> torch.Tensor:
        # g_eq_i = (2 Cv - D) T * f_eq_i  (no u term; u lives in v_i)
        return (2.0 * self.cv - self.dimension) * temperature[None] * self.f_equilibrium(density)

    def equilibria(
        self, density: torch.Tensor, temperature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.f_equilibrium(density), self.g_equilibrium(density, temperature)
