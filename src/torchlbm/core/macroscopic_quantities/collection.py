from typing import List

import torch
import torch.nn as nn
from dataclasses import dataclass

from torchlbm.core.macroscopic_quantities.density import DensityCalculationModule
from torchlbm.core.macroscopic_quantities.velocities import (
    MacroscopicVelocityCalculationModule,
)
from torchlbm.node_data import NodeData


class MacroscopicQuantityCalculationModule(nn.Module):
    """The MacroscopicQuantityCalculationModule calculation module provides functionality to calculate macroscopic quantities
    based on the discretized velocity distribution. It is implemented as a PyTorch module to allow composing algorithms
    based on consecutively applied modules.

    Args:
        nn (nn.Module): The base class. The MacroscopicQuantityCalculationModule is implemented as a PyTorch module.
                                        This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, lattice_velocities: List[List[float]]) -> None:
        """The initializer.

        Args:
            lattice_velocities (List[List[float]]): The lattice velocities as a two-dimensional list. It can thus be converted to tensors of arbitrary type.
        """
        super(MacroscopicQuantityCalculationModule, self).__init__()
        self.lattice_velocities = torch.tensor(lattice_velocities)
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)

    def forward(self, node_data: NodeData) -> NodeData:
        """The forward pass of the module which calculates the macroscopic quantities based on the population.
        Macroscopic quantities are for example density and velocity, which represent the moments of the discretized velocity distribution.

        Args:
            discrete_velocities (torch.Tensor): The discretized velocity distribution. It is a (L x Nx x Ny x Nz) tensor,
                                                where L denotes the number of lattice velocities, and Nx, Ny, and Nz the number of cells
                                                in x-, y-, and z-direction, respectively.

        Returns:
            List[torch.Tensor]: The list of tensors of macroscopic quantities.
                                Each component (of type torch.Tensor) represents the field of a macroscopic quantity.
                                For vector-based quantities we have chosen a struct-of array based memory layout.
                                Thus, a three-dimensional vector (velocity for example) has the layout (3 x Nx x Ny x Nz),
                                where 3 denotes the dimension and Nx, Ny, and Nz refer to the number of cells in x-, y-, and z-direction.
        """
        node_data.moments.density = torch.sum(node_data.distributions.old_population, dim=0)
        node_data.moments.velocity = torch.einsum(
            "dQ,QNML->dNML", self.lattice_velocities_const, node_data.distributions.old_population
        ) / node_data.moments.density.unsqueeze(0)
        return node_data
