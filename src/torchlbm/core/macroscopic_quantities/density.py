import torch
import torch.nn as nn


class DensityCalculationModule(nn.Module):
    """The PyTorch module used to calculate the density based on the discretized velocity distribution.

    Args:
        nn (nn.Modules): The base class. The DensityCalculationModule is implemented as a PyTorch module.
                         This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self) -> None:
        """The initializer."""
        super(DensityCalculationModule, self).__init__()

    def forward(self, discrete_velocities: torch.Tensor) -> torch.Tensor:
        """The forward pass of the module. It calculates the density based on the discrete velocity distribution.
        The density is the zeroth moment of the distribtion.

        Args:
            discrete_velocities (torch.Tensor): The discretized velocity distribution. It is a (L x Nx x Ny x Nz) tensor,
                                                where L denotes the number of lattice velocities, and Nx, Ny, and Nz the number of cells
                                                in x-, y-, and z-direction, respectively.

        Returns:
            torch.Tensor: The density field as a tensor. It has the shape (Nx x Ny x Nz),
                          where Nx, Ny, and Nz represent the number of total cells in x-, y-, and z-direction, respectively.
        """
        return torch.sum(discrete_velocities, dim=0)
