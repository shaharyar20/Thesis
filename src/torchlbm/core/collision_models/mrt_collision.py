from typing import List

import torch.nn as nn
import torch

from torchlbm.node_data import NodeData


class MRTCollisionModule(nn.Module):
    """A TorchLBM module that performs the multi relaxation time collision step of a Lattice-Boltzmann algorithm

    Args:
        nn (nn.Module): The collision operator is implemented as a PyTorch module. This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(
        self, relaxation_omega: float, my_population_to_momentum_transform: List[List[float]], my_momentum_to_population_transform: List[List[float]]
    ) -> None:
        """The initializer of the two relaxation time collosion module.

        Args:
            relaxation_omega (float): The relaxation parameter.
            my_population_to_momentum_transform (List[List[float]]): The matrix to transform to the momentum space.
                                                                     It is a property of the underlying velocity set.
            my_momentum_to_population_transform (List[List[float]]): The matrix to transform from the momentum space.
                                                                     It is a property of the underlying velocity set.
        """
        super(MRTCollisionModule, self).__init__()
        self.relaxation_omega = relaxation_omega
        self.my_population_to_momentum_transform = torch.Tensor(my_population_to_momentum_transform)
        self.register_buffer("population_to_momentum_transform_const", self.my_population_to_momentum_transform)
        self.my_momentum_to_population_transform = torch.Tensor(my_momentum_to_population_transform)
        self.register_buffer("momentum_to_population_transform_const", self.my_momentum_to_population_transform)
        self.relaxation = torch.full((9,), self.relaxation_omega).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("relaxation_const", self.relaxation)

    def forward(self, node_data: NodeData) -> torch.Tensor:
        """The forward pass of the collision module. Gets as input the discretized velocity distribution of the start of the timestept,
        and the equilibrium distribution calculated based on it. It returns the post-collision distribution.

        Args:
            discrete_velocities (torch.Tensor): The discretized velocity distribution at the beginning of the timestep. It is a (L x Nx x Ny x Nz) tensor,
            where L denotes the number of lattice velocities, and Nx, Ny, and Nz the number of cells in x-, y-, and z-direction, respectively.
            equilibrium_discrete_velocities (torch.Tensor): The equilibirium distribution towards which relaxation is performed.
                                                            It is a (L x Nx x Ny x Nz) tensor, where L denotes the number of lattice velocities,
                                                            and Nx, Ny, and Nz the number of cells in x-, y-, and z-direction, respectively.

        Returns:
            torch.Tensor: The discretized velocity distribution after collision.
        """
        moment_populations = torch.einsum("iQ,QNML->iNML", self.population_to_momentum_transform_const, node_data.distributions.old_population)
        moment_equilibrium_populations = torch.einsum("iQ,QNML->iNML", self.population_to_momentum_transform_const, node_data.distributions.new_population)
        collide = self.relaxation_const * (moment_populations - moment_equilibrium_populations)
        discrete_velocities_post_collision = torch.einsum("iQ,QNML->iNML", self.momentum_to_population_transform_const, moment_populations - collide)

        return discrete_velocities_post_collision
