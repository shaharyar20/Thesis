from typing import List

import torch.nn as nn
import torch

from torchlbm.node_data import NodeData


class TRTCollisionModule(nn.Module):
    """A Torch LBM module that performs the single relaxation time collision step of a Lattice-Boltzmann algorithm

    Args:
        nn (nn.Module): The collision operator is implemented as a PyTorch module. This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, relaxation_omega: float, my_opposite_lattice_indices: List[int], magic_parameter: float = 1.0 / 4.0) -> None:
        """The initializer of the two relaxation time collosion module.

        Args:
            relaxation_omega (float): The relaxation parameter.
            my_opposite_lattice_indices (List[int]): The opposite lattice indices of the velocity set of the simulation.
            magic_parameter (float, optional): The parameter of the two relaxation time scheme. Defaults to 1.0/4.0.
        """
        super(TRTCollisionModule, self).__init__()
        self.relaxation_omega = relaxation_omega
        self.my_opposite_lattice_indices = torch.tensor(my_opposite_lattice_indices)
        self.register_buffer("my_opposite_lattice_indices_const", self.my_opposite_lattice_indices)
        self.magic_parameter = magic_parameter

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
        symmetric_omega = self.relaxation_omega
        antisymmetric_omega = 1.0 / (self.magic_parameter / (1.0 / symmetric_omega - 0.5) + 0.5)

        symmetric_discrete_velocities = node_data.distributions.old_population + node_data.distributions.old_population[self.my_opposite_lattice_indices_const]
        antisymmetric_discrete_velocities = (
            node_data.distributions.old_population - node_data.distributions.old_population[self.my_opposite_lattice_indices_const]
        )
        symmetric_equilibrium_discrete_velocities = (
            node_data.distributions.new_population + node_data.distributions.new_population[self.my_opposite_lattice_indices_const]
        )
        antisymmetric_equilibrium_discrete_velocities = (
            node_data.distributions.new_population - node_data.distributions.new_population[self.my_opposite_lattice_indices_const]
        )

        discrete_velocities_post_collision = (
            node_data.distributions.old_population
            - symmetric_omega * 0.5 * (symmetric_discrete_velocities - symmetric_equilibrium_discrete_velocities)
            - antisymmetric_omega * 0.5 * (antisymmetric_discrete_velocities - antisymmetric_equilibrium_discrete_velocities)
        )

        return discrete_velocities_post_collision
