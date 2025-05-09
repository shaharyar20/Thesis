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
        self, 
        free_parameters: List, 
        my_population_to_momentum_transform: List[List[float]], 
        my_momentum_to_population_transform: List[List[float]], 
        number_of_discrete_velocities: int,
        free_parameter_indices: List[int],
        viscosity_indices: List[int],
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
        self.my_population_to_momentum_transform = torch.Tensor(my_population_to_momentum_transform)
        self.register_buffer("population_to_momentum_transform_const", self.my_population_to_momentum_transform)
        self.my_momentum_to_population_transform = torch.Tensor(my_momentum_to_population_transform)
        self.register_buffer("momentum_to_population_transform_const", self.my_momentum_to_population_transform)

        self.free_parameters = torch.Tensor(free_parameters)
        self.relaxation_vector = torch.ones(number_of_discrete_velocities)
        self.free_parameters_indices = free_parameter_indices
        self.viscosity_indices = viscosity_indices
        self.relaxation_vector[self.free_parameters_indices] = self.free_parameters
        self.register_buffer("relaxation_vector_const", self.relaxation_vector)


    def forward(self, old_population: torch.Tensor, new_population: torch.Tensor, relaxation_omega: torch.Tensor, collision_source_term: torch.Tensor) -> torch.Tensor:
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
        self.relaxation_vector_const[self.viscosity_indices] = relaxation_omega
        moment_populations = torch.einsum("iQ,QNML->iNML", self.population_to_momentum_transform_const, old_population)
        moment_equilibrium_populations = torch.einsum("iQ,QNML->iNML", self.population_to_momentum_transform_const, new_population)
        collide = self.relaxation_vector_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (moment_populations - moment_equilibrium_populations)

        if collision_source_term is not None:
            moment_collision_source_term = torch.einsum("iQ,QNML->iNML", self.population_to_momentum_transform_const, collision_source_term)
            collide -= (1.0 - 0.5 * self.relaxation_vector_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) * moment_collision_source_term

        discrete_velocities_post_collision = torch.einsum("iQ,QNML->iNML", self.momentum_to_population_transform_const, moment_populations - collide)

        return discrete_velocities_post_collision
