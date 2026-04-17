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

    def get_halo_array_single_field(self, array: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor:
        """Returns the halo cells of a specified array. The i-, j-, and k-parameters are either 0, 1, 2. This means:
        0: Lower part of the spatial direction.
        1: Middle part of the spatial direction.
        2: Upper part of the spatial direction.

        Args:
            array (torch.Tensor): The array for which the halo cells are returned. It has the dimension (Tx, Ty, Tz),
                                  where Tx, Ty, and Tz denote the total number of cells of the computational domain.
            i (int): Indicating the part of the block in x-direction.
            j (int): Indicating the part of the block in y-direction.
            k (int): Indicating the part of the block in z-direction.

        Returns:
            torch.Tensor: Part of the input array which corresponds to the halo cells addressed by i, j, and k.
        """
        shifted_i = i - 1
        shifted_j = j - 1
        shifted_k = k - 1
        return array[
            self.access_indices[0][i] - shifted_i * self.num_halo_cells : self.access_indices[0][i + 1] - shifted_i * self.num_halo_cells,
            self.access_indices[1][j] - shifted_j * self.num_halo_cells if self.dimension != 1 else 0 : (
                self.access_indices[1][j + 1] - shifted_j * self.num_halo_cells if self.dimension != 1 else 1
            ),
            self.access_indices[2][k] - shifted_k * self.num_halo_cells if self.dimension == 3 else 0 : (
                self.access_indices[2][k + 1] - shifted_k * self.num_halo_cells if self.dimension == 3 else 1
            ),
        ]

    def exchange_halo_in_array_single_field(self, array: torch.Tensor, halo_cells: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor:
        """Exchanges the halo cells in a scalar-valued field. The i-, j-, and k-parameters are either 0, 1, 2. This means:
        0: Lower part of the spatial direction.
        1: Middle part of the spatial direction.
        2: Upper part of the spatial direction.

        Args:
            array (torch.Tensor): The array for which the part addressed by i, j, and k will be filled. It has the dimension (Tx, Ty, Tz),
                                  where Tx, Ty, and Tz denote the total number of cells of the computational domain.
            halo_cells (torch.Tensor): The values with which the scalar-valued array will be filled.
            i (int): Indicating the part of the block in x-direction.
            j (int): Indicating the part of the block in y-direction.
            k (int): Indicating the part of the block in z-direction.

        Returns:
            torch.Tensor: The scalar-valued field with exchanged halos. It has the dimension (Tx, Ty, Tz),
                          where Tx, Ty, and Tz denote the total number of cells of the computational domain.
        """
        array[
            self.access_indices[0][i] : self.access_indices[0][i + 1],
            self.access_indices[1][j] if self.dimension != 1 else 0 : self.access_indices[1][j + 1] if self.dimension != 1 else 1,
            self.access_indices[2][k] if self.dimension == 3 else 0 : self.access_indices[2][k + 1] if self.dimension == 3 else 1,
        ] = halo_cells
        return array

    def get_halo_array_multi_field(self, array: torch.Tensor, i: int, j: int, k: int):
        """Returns the halo cells of a specified vector-valued array. The i-, j-, and k-parameters are either 0, 1, 2. This means:
        0: Lower part of the spatial direction.
        1: Middle part of the spatial direction.
        2: Upper part of the spatial direction.

        Args:
            array (torch.Tensor): The array for which the halo cells are returned. It has the dimension (n, Tx, Ty, Tz),
                                  where Tx, Ty, and Tz denote the total number of cells of the computational domain.
                                  n is the number of vector components.
            i (int): Indicating the part of the block in x-direction.
            j (int): Indicating the part of the block in y-direction.
            k (int): Indicating the part of the block in z-direction.

        Returns:
            torch.Tensor: Part of the input array which corresponds to the halo cells addressed by i, j, and k.
        """
        shifted_i = i - 1
        shifted_j = j - 1
        shifted_k = k - 1
        return array[
            :,
            self.access_indices[0][i] - shifted_i * self.num_halo_cells : self.access_indices[0][i + 1] - shifted_i * self.num_halo_cells,
            self.access_indices[1][j] - shifted_j * self.num_halo_cells if self.dimension != 1 else 0 : (
                self.access_indices[1][j + 1] - shifted_j * self.num_halo_cells if self.dimension != 1 else 1
            ),
            self.access_indices[2][k] - shifted_k * self.num_halo_cells if self.dimension == 3 else 0 : (
                self.access_indices[2][k + 1] - shifted_k * self.num_halo_cells if self.dimension == 3 else 1
            ),
        ]

    def exchange_halo_in_array_multi_field(self, array: torch.Tensor, halo_cells: torch.Tensor, i: int, j: int, k: int):
        """Exchanges the halo cells in a vector-valued field. The i-, j-, and k-parameters are either 0, 1, 2. This means:
        0: Lower part of the spatial direction.
        1: Middle part of the spatial direction.
        2: Upper part of the spatial direction.

        Args:
            array (torch.Tensor): The array for which the part addressed by i, j, and k will be filled. It has the dimension (n, Tx, Ty, Tz),
                                  where Tx, Ty, and Tz denote the total number of cells of the computational domain.
                                  n is the number of vector components.
            halo_cells (torch.Tensor): The values with which the scalar-valued array will be filled.
            i (int): Indicating the part of the block in x-direction.
            j (int): Indicating the part of the block in y-direction.
            k (int): Indicating the part of the block in z-direction.

        Returns:
            torch.Tensor: The scalar-valued field with exchanged halos. It has the dimension (n, Tx, Ty, Tz),
                          where Tx, Ty, and Tz denote the total number of cells of the computational domain.
                          n is the number of vector components.
        """
        array[
            :,
            self.access_indices[0][i] : self.access_indices[0][i + 1],
            self.access_indices[1][j] if self.dimension != 1 else 0 : self.access_indices[1][j + 1] if self.dimension != 1 else 1,
            self.access_indices[2][k] if self.dimension == 3 else 0 : self.access_indices[2][k + 1] if self.dimension == 3 else 1,
        ] = halo_cells
        return array

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
        discrete_velocities = node_data.distributions.old_population

        x_range = [0, 1, 2]
        y_range = [0, 1, 2] if self.dimension != 1 else [1]
        z_range = [0, 1, 2] if self.dimension == 3 else [1]

        for i in x_range:
            for j in y_range:
                for k in z_range:
                    if (i != 1 and self.is_i_periodic) or (j != 1 and self.is_j_periodic) or (k != 1 and self.is_k_periodic):
                        distribution_halo_cells = self.get_halo_array_multi_field(discrete_velocities, i, j, k)
                        # print(distribution_halo_cells.shape)
                        discrete_velocities = self.exchange_halo_in_array_multi_field(
                            discrete_velocities,
                            distribution_halo_cells,
                            -1 * (i - 1) + 1,
                            -1 * (j - 1) + 1,
                            -1 * (k - 1) + 1,
                        )
        node_data.distributions.old_population = discrete_velocities

        return node_data
