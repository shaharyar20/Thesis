import torch
import torch.nn as nn
from dataclasses import dataclass

# from torchlbm.node_data import NodeData\
from torchlbm.thermal_node_data import ThermalNodeData


class ThermalZeroGradientBoundaryUpdate(nn.Module):
    """Class that performs the zero gradient boundary update. Can for example be used to model outlets. Howeever, reflections might occor.

    Args:
        nn (nn.Module): The zero gradient wall boundary update is implemented as a PyTorch module in order to ensure easy algorithm assembly.
    """

    def __init__(
        self,
        is_east_zero_gradient: bool,
        is_west_zero_gradient: bool,
        is_north_zero_gradient: bool,
        is_south_zero_gradient: bool,
        is_top_zero_gradient: bool,
        is_bottom_zero_gradient: bool,
        num_halo_cells: int,
        dimension: int,
    ) -> None:
        """The constructor for the zero gradient boundary update. Usually called from a factory function.

        Args:
            is_east_zero_gradient (bool): Inidicates whether the east boundary of the computational domain is a zero gradient boundary.
            is_west_zero_gradient (bool): Inidicates whether the west boundary of the computational domain is a zero gradient boundary.
            is_north_zero_gradient (bool): Inidicates whether the north boundary of the computational domain is a zero gradient boundary.
            is_south_zero_gradient (bool): Inidicates whether the south boundary of the computational domain is a zero gradient boundary.
            is_top_zero_gradient (bool): Inidicates whether the top boundary of the computational domain is a zero gradient boundary.
            is_bottom_zero_gradient (bool): Inidicates whether the bottom boundary of the computational domain is a zero gradient boundary.
            num_halo_cells (int): The number of halo cells in each spatial dimension.
            dimension (int): The spatial dimension.
        """
        super(ThermalZeroGradientBoundaryUpdate, self).__init__()

        self.is_east_zero_gradient = is_east_zero_gradient
        self.is_west_zero_gradient = is_west_zero_gradient
        self.is_north_zero_gradient = is_north_zero_gradient
        self.is_south_zero_gradient = is_south_zero_gradient
        self.is_top_zero_gradient = is_top_zero_gradient
        self.is_bottom_zero_gradient = is_bottom_zero_gradient
        self.num_halo_cells = num_halo_cells
        self.dimension = dimension

    def update_east_zero_gradient(self, population: torch.Tensor) -> torch.Tensor:
        """Applies the zero-gradient boundary condition to the east boundary of the computational domain. Called from the forward pass
        of the PyTorch module that implements the overall zero-gradient update functionality.

        Args:
            population (torch.Tensor): The discretised velocity distribution for which the zero-gradient update should be performed.
                                       It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The discretised velocity distribution where the zero-gradient boundary condition has already been applied.
                          It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.
        """
        population[:, -self.num_halo_cells :, :, :] = population[:, -2 * self.num_halo_cells : -self.num_halo_cells, :, :]
        return population

    def update_west_zero_gradient(self, population: torch.Tensor) -> torch.Tensor:
        """Applies the zero-gradient boundary condition to the west boundary of the computational domain. Called from the forward pass
        of the PyTorch module that implements the overall zero-gradient update functionality.

        Args:
            population (torch.Tensor): The discretised velocity distribution for which the zero-gradient update should be performed.
                                       It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The discretised velocity distribution where the zero-gradient boundary condition has already been applied.
                          It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.
        """
        population[:, : self.num_halo_cells, :, :] = population[:, self.num_halo_cells : 2 * self.num_halo_cells, :, :]
        return population

    def update_north_zero_gradient(self, population: torch.Tensor) -> torch.Tensor:
        """Applies the zero-gradient boundary condition to the north boundary of the computational domain. Called from the forward pass
        of the PyTorch module that implements the overall zero-gradient update functionality.

        Args:
            population (torch.Tensor): The discretised velocity distribution for which the zero-gradient update should be performed.
                                       It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The discretised velocity distribution where the zero-gradient boundary condition has already been applied.
                          It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.
        """
        population[:, :, -self.num_halo_cells :, :] = population[:, :, -2 * self.num_halo_cells : -self.num_halo_cells, :]
        return population

    def update_south_zero_gradient(self, population: torch.Tensor) -> torch.Tensor:
        """Applies the zero-gradient boundary condition to the south boundary of the computational domain. Called from the forward pass
        of the PyTorch module that implements the overall zero-gradient update functionality.

        Args:
            population (torch.Tensor): The discretised velocity distribution for which the zero-gradient update should be performed.
                                       It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The discretised velocity distribution where the zero-gradient boundary condition has already been applied.
                          It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.
        """
        population[:, :, : self.num_halo_cells, :] = population[:, :, self.num_halo_cells : 2 * self.num_halo_cells, :]
        return population

    def update_top_zero_gradient(self, population: torch.Tensor) -> torch.Tensor:
        """Applies the zero-gradient boundary condition to the top boundary of the computational domain. Called from the forward pass
        of the PyTorch module that implements the overall zero-gradient update functionality.

        Args:
            population (torch.Tensor): The discretised velocity distribution for which the zero-gradient update should be performed.
                                       It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The discretised velocity distribution where the zero-gradient boundary condition has already been applied.
                          It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.
        """
        population[:, :, :, -self.num_halo_cells :] = population[:, :, :, -2 * self.num_halo_cells : -self.num_halo_cells]
        return population

    def update_bottom_zero_gradient(self, population: torch.Tensor) -> torch.Tensor:
        """Applies the zero-gradient boundary condition to the bottom boundary of the computational domain. Called from the forward pass
        of the PyTorch module that implements the overall zero-gradient update functionality.

        Args:
            population (torch.Tensor): The discretised velocity distribution for which the zero-gradient update should be performed.
                                       It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The discretised velocity distribution where the zero-gradient boundary condition has already been applied.
                          It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.
        """
        population[:, :, :, : self.num_halo_cells] = population[:, :, :, self.num_halo_cells : 2 * self.num_halo_cells]
        return population

    def forward(self, node_data: ThermalNodeData) -> ThermalNodeData:
        """The forward pass of the module which performs the zero gradient boundary update.

        Args:
            node_data (NodeData): The node data that contains the storage intensive fields for macroscopic and microscopic quantities.

        Returns:
            NodeData: The node data that contains the storage intensive fields for macroscopic and microscopic quantities.
                      The zero-gradient update has already been performed.
        """
        if self.is_east_zero_gradient:
            node_data.distributions.vel_old_population = self.update_east_zero_gradient(node_data.distributions.vel_old_population)
            node_data.distributions.temp_old_population = self.update_east_zero_gradient(node_data.distributions.temp_old_population)
        if self.is_west_zero_gradient:
            node_data.distributions.vel_old_population = self.update_west_zero_gradient(node_data.distributions.vel_old_population)
            node_data.distributions.temp_old_population = self.update_west_zero_gradient(node_data.distributions.temp_old_population)

        if self.is_north_zero_gradient:
            if self.dimension != 1:
                node_data.distributions.vel_old_population = self.update_north_zero_gradient(node_data.distributions.vel_old_population)
                node_data.distributions.temp_old_population = self.update_north_zero_gradient(node_data.distributions.temp_old_population)
        if self.is_south_zero_gradient:
            if self.dimension != 1:
                node_data.distributions.vel_old_population = self.update_south_zero_gradient(node_data.distributions.vel_old_population)
                node_data.distributions.temp_old_population = self.update_south_zero_gradient(node_data.distributions.temp_old_population)

        # if self.is_top_zero_gradient:
        if self.dimension == 3:
            node_data.distributions.vel_old_population = self.update_top_zero_gradient(node_data.distributions.vel_old_population)
            node_data.distributions.temp_old_population = self.update_top_zero_gradient(node_data.distributions.temp_old_population)
        # if self.is_bottom_zero_gradient:
        if self.dimension == 3:
            node_data.distributions.vel_old_population = self.update_bottom_zero_gradient(node_data.distributions.vel_old_population)
            node_data.distributions.temp_old_population = self.update_bottom_zero_gradient(node_data.distributions.temp_old_population)

        return node_data
