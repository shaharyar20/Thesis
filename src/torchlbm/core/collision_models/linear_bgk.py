from typing import List

import torch.nn as nn
import torch
from dataclasses import dataclass

from torchlbm.node_data import NodeData


class EquilibriumCalculationModule(nn.Module):
    """PyTorch module that implements the calculation of the equilibrium distribution based on a linear BGK operator.
    It calculates the equilibrium for the Navier-Stokes equations.
    It is implemented as a PyTorch module to allow composing algorithms based on consecutively applied modules.

    Args:
        nn (nn.Module): PyTorch module which is used ase base class.
    """

    def __init__(self, lattice_velocities: List[List[float]], lattice_weights: List[float]) -> None:
        """Initializer for the equilibrium calculation module.

        Args:
            lattice_velocities (List[List[float]]): The lattice velocities of the used velocity set.
            lattice_weights (List[float]): The lattice weights of the used velocity set.
        """
        super(EquilibriumCalculationModule, self).__init__()
        self.lattice_velocities = torch.tensor(lattice_velocities)
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)
        self.lattice_weights = torch.tensor(lattice_weights)
        self.register_buffer("lattice_weights_const", self.lattice_weights)

    def forward(self, node_data: NodeData) -> NodeData:
        """The forward pass of the equilibrium calculation module. The equilibrium distribution for the Navier-Stokes equations are calculated.
        It gets as input macroscopic quantities, the moments of the velocity distribution (density and velocity) and calculates the
        equilibrium distribution based on them.

        Args:
            node_data (NodeData): The node_data object is a pure data container that contains the storage heavy macroscopic and microscopix field data.

        Returns:
            NodeData: The modified node_data object where the new_population is overwritten with the equilibrium distribution.
        """
        macroscopic_velocity = node_data.moments.velocity + node_data.moments.forcing_velocity
        projected_discrete_velocities = torch.einsum(
            "dQ,dNML->QNML",
            self.lattice_velocities_const,
            macroscopic_velocity,
        )
        macroscopic_velocity_magnitude = torch.linalg.norm(
            macroscopic_velocity,
            ord=2,
            dim=0,
        )
        node_data.distributions.new_population = (
            node_data.moments.density.unsqueeze(0)
            * self.lattice_weights_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * (1 + 3 * projected_discrete_velocities + 9 / 2 * projected_discrete_velocities**2 - 3 / 2 * macroscopic_velocity_magnitude.unsqueeze(0) ** 2)
        )

        return node_data
