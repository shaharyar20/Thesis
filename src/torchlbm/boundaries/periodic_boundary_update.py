from typing import List
import torch.nn as nn
import torch
from dataclasses import dataclass
from torchlbm.node_data import NodeData


class PeriodicBoundaryUpdate(nn.Module):
    """Performs the periodic boundary update.

    Args:
        nn (nn.Module): The periodic boundary update operator is implemented as a PyTorch module.
                        This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(
        self,
        num_halo_cells: int,
        access_indices: List[List[int]],
        dimension: int,
        is_i_periodic: bool,
        is_j_periodic: bool,
        is_k_periodic: bool,
    ) -> None:
        """The initializer of the the periodic boundary update. It is usually called from a factory function.

        Args:
            num_halo_cells (int): The number of halo cells in each spatial dimension.
            access_indices (List[List[int]]): Important indices used to acces the cells that are involved in the
                                              periodic boundary update.
            dimension (int): The spatial dimension as an integer.
            is_i_periodic (bool): Indicator whether the x direction is periodic.
            is_j_periodic (bool): Indicator whether the y direction is periodic.
            is_k_periodic (bool): Indicator whether the z direction is periodic.
        """
        super(PeriodicBoundaryUpdate, self).__init__()
        self.num_halo_cells: int = num_halo_cells
        self.access_indices: List[List[int]] = access_indices
        self.dimension: int = dimension
        self.is_i_periodic: bool = is_i_periodic
        self.is_j_periodic: bool = is_j_periodic
        self.is_k_periodic: bool = is_k_periodic

    def forward(self, node_data: NodeData) -> NodeData:
        """Performs the periodic boundary update as the forward pass of the PyTorch module.

        Args:
            node_data (NodeData): The node data that contains the storage intensive fields for macroscopic and microscopic quantities.
                                  Only the discrete velocity population for which the halo update is performed is used.
                                  It is a (L, Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells
                                  of the computational domain.
                                  L is the number of discrete velocities of the underlying velocity set.

        Returns:
            NodeData: The node data with applied periodic boundary conditions.
        """

        x = node_data.distributions.old_population
        halo = self.num_halo_cells

        if self.dimension == 3:
            # --- Face Updates ---
            if self.is_i_periodic:
                x[:, :halo, :, :] = x[:, -2 * halo : -halo, :, :]  # Left halo from right interior
                x[:, -halo:, :, :] = x[:, halo : 2 * halo, :, :]  # Right halo from left interior

            if self.is_j_periodic:
                x[:, :, :halo, :] = x[:, :, -2 * halo : -halo, :]  # Bottom halo from top interior
                x[:, :, -halo:, :] = x[:, :, halo : 2 * halo, :]  # Top halo from bottom interior

            if self.is_k_periodic:
                x[:, :, :, :halo] = x[:, :, :, -2 * halo : -halo]  # Front halo from back interior
                x[:, :, :, -halo:] = x[:, :, :, halo : 2 * halo]  # Back halo from front interior

            # --- Edge Updates ---
            if self.is_i_periodic and self.is_j_periodic:
                x[:, :halo, :halo, :] = x[:, -2 * halo : -halo, -2 * halo : -halo, :]
                x[:, -halo:, :halo, :] = x[:, halo : 2 * halo, -2 * halo : -halo, :]
                x[:, :halo, -halo:, :] = x[:, -2 * halo : -halo, halo : 2 * halo, :]
                x[:, -halo:, -halo:, :] = x[:, halo : 2 * halo, halo : 2 * halo, :]

            if self.is_i_periodic and self.is_k_periodic:
                x[:, :halo, :, :halo] = x[:, -2 * halo : -halo, :, -2 * halo : -halo]
                x[:, -halo:, :, :halo] = x[:, halo : 2 * halo, :, -2 * halo : -halo]
                x[:, :halo, :, -halo:] = x[:, -2 * halo : -halo, :, halo : 2 * halo]
                x[:, -halo:, :, -halo:] = x[:, halo : 2 * halo, :, halo : 2 * halo]

            if self.is_j_periodic and self.is_k_periodic:
                x[:, :, :halo, :halo] = x[:, :, -2 * halo : -halo, -2 * halo : -halo]
                x[:, :, -halo:, :halo] = x[:, :, halo : 2 * halo, -2 * halo : -halo]
                x[:, :, :halo, -halo:] = x[:, :, -2 * halo : -halo, halo : 2 * halo]
                x[:, :, -halo:, -halo:] = x[:, :, halo : 2 * halo, halo : 2 * halo]

            # --- Corner Updates ---
            if self.is_i_periodic and self.is_j_periodic and self.is_k_periodic:
                x[:, :halo, :halo, :halo] = x[:, -2 * halo : -halo, -2 * halo : -halo, -2 * halo : -halo]
                x[:, -halo:, :halo, :halo] = x[:, halo : 2 * halo, -2 * halo : -halo, -2 * halo : -halo]
                x[:, :halo, -halo:, :halo] = x[:, -2 * halo : -halo, halo : 2 * halo, -2 * halo : -halo]
                x[:, -halo:, -halo:, :halo] = x[:, halo : 2 * halo, halo : 2 * halo, -2 * halo : -halo]

                x[:, :halo, :halo, -halo:] = x[:, -2 * halo : -halo, -2 * halo : -halo, halo : 2 * halo]
                x[:, -halo:, :halo, -halo:] = x[:, halo : 2 * halo, -2 * halo : -halo, halo : 2 * halo]
                x[:, :halo, -halo:, -halo:] = x[:, -2 * halo : -halo, halo : 2 * halo, halo : 2 * halo]
                x[:, -halo:, -halo:, -halo:] = x[:, halo : 2 * halo, halo : 2 * halo, halo : 2 * halo]

        if self.dimension == 2:
            # --- Edge Updates ---
            if self.is_i_periodic:
                x[:, :halo, :, :] = x[:, -2 * halo : -halo, :, :]  # Left halo from right interior
                x[:, -halo:, :, :] = x[:, halo : 2 * halo, :, :]  # Right halo from left interior

            if self.is_j_periodic:
                x[:, :, :halo, :] = x[:, :, -2 * halo : -halo, :]  # Bottom halo from top interior
                x[:, :, -halo:, :] = x[:, :, halo : 2 * halo, :]  # Top halo from bottom interior

            # --- Corner Updates ---
            if self.is_i_periodic and self.is_j_periodic:
                x[:, :halo, :halo, :] = x[:, -2 * halo : -halo, -2 * halo : -halo, :]  # Bottom-left from top-right
                x[:, -halo:, :halo, :] = x[:, halo : 2 * halo, -2 * halo : -halo, :]  # Bottom-right from top-left
                x[:, :halo, -halo:, :] = x[:, -2 * halo : -halo, halo : 2 * halo, :]  # Top-left from bottom-right
                x[:, -halo:, -halo:, :] = x[:, halo : 2 * halo, halo : 2 * halo, :]  # Top-right from bottom-left

        if self.dimension == 1:
            if self.is_i_periodic:
                x[:, :halo, :, :] = x[:, -2 * halo : -halo, :, :]  # Left halo from right interior
                x[:, -halo:, :, :] = x[:, halo : 2 * halo, :, :]  # Right halo from left interior

        node_data.distributions.old_population = x

        return node_data
