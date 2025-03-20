from typing import List
import torch.nn as nn
import torch
from dataclasses import dataclass

from torchlbm.node_data import NodeData


class BounceBackBoundaryUpdate(nn.Module):
    """Implements the bounce back boundary update to model rigid bodies.

    Args:
        nn (nn.Module): The bounce back boundary update operator is implemented as a PyTorch module.
                        This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, opposite_lattice_indices: List[int]) -> None:
        """Initializer for the bounce back boundary update. Initializes all relevant members. Usually created using a factory function.

        Args:
            opposite_lattice_indices (List[int]): The opposite lattice indices that are used to perform the bounce back.
        """
        super(BounceBackBoundaryUpdate, self).__init__()
        self.opposite_lattice_indices: List[int] = opposite_lattice_indices

    def forward(self, node_data: NodeData) -> NodeData:
        """The forward pass of the module that performs the actual bounce back update.

        Args:
            node_data (NodeData): The node data that contains the storage intensive fields for macroscopic and microscopic quantities.
                                  On the discrete velocities are used. Theay are a (L, Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote
                                  the total number of cells of the computational domain.
                                  L is the number of discrete velocities of the underlying velocity set.

        Returns:
            NodeData: The node data which is updated according to the bounce back algorithm.
        """
        discrete_velocities = node_data.distributions.old_population
        if node_data.bounce_back_mask is not None:
            discrete_velocities = torch.where(node_data.bounce_back_mask > 0, discrete_velocities[self.opposite_lattice_indices], discrete_velocities)
        node_data.distributions.old_population = discrete_velocities
        return node_data
