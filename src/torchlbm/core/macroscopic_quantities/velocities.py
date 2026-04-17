from typing import List

import torch.nn as nn
import torch


class MacroscopicVelocityCalculationModule(nn.Module):
    """The MacroscopicVelocityCalculationModule calculation module provides functionality to calculate the macroscopic velocity-vector field
    based on the discretized velocity distribution. It is implemented as a PyTorch module to allow composing algorithms
    based on consecutively applied modules.

    Args:
        nn (nn.Module): The base class. The MacroscopicVelocityCalculationModule is implemented as a PyTorch module.
                         This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, lattice_velocities: List[List[float]]) -> None:
        """The initializer.

        Args:
            lattice_velocities (List[List[float]]): The lattice velocities as a two-dimensional list. It can thus be converted to tensors of arbitrary type.
        """
        super(MacroscopicVelocityCalculationModule, self).__init__()
        self.lattice_velocities = torch.tensor(lattice_velocities)
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)

    def forward(self, discrete_velocities: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """The forward pass of the module which calculates the macroscopic velocity-vector field based on the population.

        Args:
            discrete_velocities (torch.Tensor): The discretized velocity distribution. It is a (L x Nx x Ny x Nz) tensor where,
                                                L denotes the number of lattice velocities, and Nx, Ny, and Nz the number of cells
                                                in x-, y-, and z-direction, respectively.
            density (torch.Tensor): The density field as a tensor. It has the shape (Nx x Ny x Nz),
                                    where Nx, Ny, and Nz represent the number of total cells in x-, y-, and z-direction, respectively.

        Returns:
            torch.Tensor: The velocity-vector field as tensor of shape (3 x Nx x Ny x Nz),
                          where 3 denotes the dimension and Nx, Ny, and Nz refer to the number of cells in x-, y-, and z-direction.
                          Note, this follows the structure-of-array approach.
        """

        macroscopic_velocities = torch.einsum("dQ,QNML->dNML", self.lattice_velocities_const, discrete_velocities) / density.unsqueeze(0)
        return macroscopic_velocities
