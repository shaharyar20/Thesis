import torch
import torch.nn as nn
from dataclasses import dataclass

from torchlbm.node_data import NodeData


class CollisionModule(nn.Module):
    """A Torch LBM module that performs the single relaxation time collision step of a Lattice-Boltzmann algorithm

    Args:
        nn (nn.Module): The collision operator is implemented as a PyTorch module. This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, relaxation_omega: float) -> None:
        """The initializer of the collision module.

        Args:
            relaxation_omega (float): The relaxation frequency.
        """
        super(CollisionModule, self).__init__()
        self.relaxation_omega = relaxation_omega

    def forward(self, node_data: NodeData) -> torch.Tensor:
        """The forward pass of the collision modules. Gets as input the discretized velocity distribution of the start of the timestept,
        and the equilibrium distribution calculated based on it. It returns the post-collision distribution.

        Args:
            node_data (NodeData): The node_data object is a pure data container that contains the storage heavy macroscopic and microscopix field data.

        Returns:
            torch.Tensor: The discretized velocity distribution after collision.
        """
        discrete_velocities_post_collision = (
            1.0 - self.relaxation_omega
        ) * node_data.distributions.old_population + self.relaxation_omega * node_data.distributions.new_population
        return discrete_velocities_post_collision
