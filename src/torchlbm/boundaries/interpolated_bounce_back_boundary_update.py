from typing import List
import torch.nn as nn
import torch
from dataclasses import dataclass

from torchlbm.node_data import NodeData


class InterpolatedBounceBackBoundaryUpdate(nn.Module):
    """Implements the bounce back boundary update to model rigid bodies.

    Args:
        nn (nn.Module): The bounce back boundary update operator is implemented as a PyTorch module.
                        This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, opposite_lattice_indices: List[int], ibb_factors1, ibb_factors2, boundary_indices, opp_boundary_indices, additional_indices) -> None:
        """Initializer for the bounce back boundary update. Initializes all relevant members. Usually created using a factory function.

        Args:
            opposite_lattice_indices (List[int]): The opposite lattice indices that are used to perform the bounce back.
        """
        super(InterpolatedBounceBackBoundaryUpdate, self).__init__()
        self.opposite_lattice_indices: List[int] = opposite_lattice_indices
        self.ibb_factors1 = ibb_factors1
        self.register_buffer("ibb_factors1_const", ibb_factors1)
        self.ibb_factors2 = ibb_factors2
        self.register_buffer("ibb_factors2_const", ibb_factors2)
        self.boundary_indices = boundary_indices
        self.register_buffer("boundary_indices_const", boundary_indices)
        self.opp_boundary_indices = opp_boundary_indices
        self.register_buffer("opp_boundary_indices_const", opp_boundary_indices)
        self.additional_indices = additional_indices
        self.register_buffer("additional_indices_const", additional_indices)
        

    def forward(self, old_population: torch.Tensor, density: torch.Tensor, velocity: torch.Tensor, bounce_back_mask: torch.Tensor) -> torch.Tensor:
        """The forward pass of the module that performs the actual bounce back update.

        Args:
            node_data (NodeData): The node data that contains the storage intensive fields for macroscopic and microscopic quantities.
                                  On the discrete velocities are used. Theay are a (L, Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote
                                  the total number of cells of the computational domain.
                                  L is the number of discrete velocities of the underlying velocity set.

        Returns:
            NodeData: The node data which is updated according to the bounce back algorithm.
        """
        
        discrete_velocities = old_population.clone()
        discrete_velocities[
            self.boundary_indices_const[:,0], self.boundary_indices_const[:,1], self.boundary_indices_const[:,2], self.boundary_indices_const[:,3]
            ] = self.ibb_factors1_const * discrete_velocities[
                self.opp_boundary_indices_const[:,0], self.opp_boundary_indices_const[:,1], self.opp_boundary_indices_const[:,2], self.opp_boundary_indices_const[:,3]
            ] + self.ibb_factors2_const * discrete_velocities[
                self.additional_indices_const[:,0], self.additional_indices_const[:,1], self.additional_indices_const[:,2], self.additional_indices_const[:,3]
            ]
        
        return discrete_velocities
