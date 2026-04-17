from typing import List

import torch
import torch.nn as nn
from dataclasses import dataclass

# from torchlbm.node_data import NodeData
from torchlbm.thermal_node_data import ThermalNodeData


class ThermalWallBoundaryUpdate(nn.Module):
    """The class that applies boundary condtions for a moving wall. In case a zero wall velocity is chosen,
    it models a rigid and non-moving wall.

    Args:
        nn (nn.Module): The wall boundary update operator is implemented as a PyTorch module.
                        This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(
        self,
        is_east_wall: bool,
        east_wall_velocity: torch.Tensor,
        east_wall_temperature: torch.Tensor,
        is_west_wall: bool,
        west_wall_velocity: torch.Tensor,
        west_wall_temperature: torch.Tensor,
        is_north_wall: bool,
        north_wall_velocity: torch.Tensor,
        north_wall_temperature: torch.Tensor,
        is_south_wall: bool,
        south_wall_velocity: torch.Tensor,
        south_wall_temperature: torch.Tensor,
        is_top_wall: bool,
        top_wall_velocity: torch.Tensor,
        top_wall_temperature: torch.Tensor,
        is_bottom_wall: bool,
        bottom_wall_velocity: torch.Tensor,
        bottom_wall_temperature: torch.Tensor,
        num_halo_cells: int,
        access_indices: List[List[int]],
        dimension: int,
        lattice_velocity: torch.Tensor,
        lattice_velocity_integers: List[List[int]],
        lattice_weights: torch.Tensor,
        east_velocities: List[int],
        west_velocities: List[int],
        north_velocities: List[int],
        south_velocities: List[int],
        top_velocities: List[int],
        bottom_velocities: List[int],
        opposite_lattice_indices: List[int],
        shifted_velx: float,
        shifted_vely: float,
    ) -> None:
        """The initializer for the wall boundary update class. It is usually called from a factory function.

        Args:
            is_east_wall (bool): Inidicator whether the east part of the computational domain is a wall.
            east_wall_velocity (torch.Tensor): The wall velocity of the east part of the computational domain.
            is_west_wall (bool): Inidicator whether the west part of the computational domain is a wall.
            west_wall_velocity (torch.Tensor): The wall velocity of the west part of the computational domain.
            is_north_wall (bool): Inidicator whether the north part of the computational domain is a wall.
            north_wall_velocity (torch.Tensor): The wall velocity of the north part of the computational domain.
            is_south_wall (bool): Inidicator whether the south part of the computational domain is a wall.
            south_wall_velocity (torch.Tensor): The wall velocity of the south part of the computational domain.
            is_top_wall (bool): Inidicator whether the top part of the computational domain is a wall.
            top_wall_velocity (torch.Tensor): The wall velocity of the top part of the computational domain.
            is_bottom_wall (bool): Inidicator whether the bottom part of the computational domain is a wall.
            bottom_wall_velocity (torch.Tensor): The wall velocity of the bottom part of the computational domain.
            num_halo_cells (int): The number of halo cells.
            access_indices (List[List[int]]): Indices that are used to acces the right cells for the wall boundary update.
            dimension (int): The spatial dimension.
            lattice_velocity (torch.Tensor): The velocity directions of the underlying velocity set.
            lattice_weights (List[float]): The weights of the underlying velocity set.
            east_velocities (List[int]): The indices of the directions of the underlying velocity set that have components in the east direction.
            west_velocities (List[int]): The indices of the directions of the underlying velocity set that have components in the west direction.
            north_velocities (List[int]): The indices of the directions of the underlying velocity set that have components in the north direction.
            south_velocities (List[int]): The indices of the directions of the underlying velocity set that have components in the south direction.
            top_velocities (List[int]): The indices of the directions of the underlying velocity set that have components in the top direction.
            bottom_velocities (List[int]): The indices of the directions of the underlying velocity set that have components in the bottom direction.
            opposite_lattice_indices (List[int]): The opposite lattice indices. They are a property of the underlying velocity set.
        """
        super(ThermalWallBoundaryUpdate, self).__init__()

        self.is_east_wall = is_east_wall
        self._east_wall_velocity = east_wall_velocity
        self.register_buffer("east_wall_velocity_const", self._east_wall_velocity)
        self._east_wall_temperature = east_wall_temperature
        self.is_west_wall = is_west_wall
        self._west_wall_velocity = west_wall_velocity
        self.register_buffer("west_wall_velocity_const", self._west_wall_velocity)
        self._west_wall_temperature = west_wall_temperature
        self.is_north_wall = is_north_wall
        self._north_wall_velocity = north_wall_velocity
        self.register_buffer("north_wall_velocity_const", self._north_wall_velocity)
        self._north_wall_temperature = north_wall_temperature
        self.is_south_wall = is_south_wall
        self._south_wall_velocity = south_wall_velocity
        self.register_buffer("south_wall_velocity_const", self._south_wall_velocity)
        self._south_wall_temperature = south_wall_temperature
        self.is_top_wall = is_top_wall
        self._top_wall_velocity = top_wall_velocity
        self.register_buffer("top_wall_velocity_const", self._top_wall_velocity)
        self._top_wall_temperature = top_wall_temperature
        self.is_bottom_wall = is_bottom_wall
        self._bottom_wall_velocity = bottom_wall_velocity
        self.register_buffer("bottom_wall_velocity_const", self._bottom_wall_velocity)
        self._bottom_wall_temperature = bottom_wall_temperature
        self.num_halo_cells = num_halo_cells
        self.access_indices = access_indices
        self.dimension = dimension

        self._lattice_velocity = lattice_velocity
        self.shifted_velx = shifted_velx
        self.shifted_vely = shifted_vely
        self._lattice_velocity[0] += self.shifted_velx
        self._lattice_velocity[1] += self.shifted_vely
        print("lattice_velocity", self._lattice_velocity)
        self.register_buffer("lattice_velocity_const", self._lattice_velocity)
        self.lattice_velocity_integers = lattice_velocity_integers
        self._lattice_weights = lattice_weights
        self.register_buffer("lattice_weights_const", self._lattice_weights)
        self.east_velocities = east_velocities
        self.west_velocities = west_velocities
        self.north_velocities = north_velocities
        self.south_velocities = south_velocities
        self.top_velocities = top_velocities
        self.bottom_velocities = bottom_velocities
        self.opposite_lattice_indices = opposite_lattice_indices

    # fmt: off
    def update_east_wall(self, vel_population: torch.Tensor, temp_population: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Update the the east boundaries with a wall-boundary condition.

        Args:
            population (torch.Tensor): The population that should be updated. It is a (L, Tx, Ty, Tz) tensor,
                                       where Tx, Ty, and Tz denote the total number of cells of the computational domain.
                                       L is the number of discrete velocities of the underlying velocity set.
            density (torch.Tensor): The density field of the block that should be updated. Used to impose wall velocities.
                                    It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The updated population. It is a (L, Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells
                          of the computational domain.
                          L is the number of discrete velocities of the underlying velocity set.
        """
        for east_index in self.east_velocities:
            opposite_index = self.opposite_lattice_indices[east_index]
            f_slice = vel_population[
                east_index,
                self.access_indices[0][2] - self.num_halo_cells + self.lattice_velocity_integers[0][east_index]:
                self.access_indices[0][3] - self.num_halo_cells + self.lattice_velocity_integers[0][east_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][east_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][east_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][east_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][east_index] if self.dimension == 3 else 1,
            ]
            density_slice = density[
                self.access_indices[0][2] - self.num_halo_cells:
                self.access_indices[0][3] - self.num_halo_cells,
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.dot(self.east_wall_velocity_const, self.lattice_velocity_const[:, east_index])
            weight = self.lattice_weights_const[east_index]
            vel_population[
                opposite_index,
                self.access_indices[0][2] - self.num_halo_cells:
                self.access_indices[0][3] - self.num_halo_cells,
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ] = f_slice - 6.0 * weight * density_slice * wall_velocity_projection
            g_slice = temp_population[
                east_index,
                self.access_indices[0][2] - self.num_halo_cells + self.lattice_velocity_integers[0][east_index]:
                self.access_indices[0][3] - self.num_halo_cells + self.lattice_velocity_integers[0][east_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][east_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][east_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][east_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][east_index] if self.dimension == 3 else 1,
            ]
            temp_population[
                opposite_index,
                self.access_indices[0][2] - self.num_halo_cells:
                self.access_indices[0][3] - self.num_halo_cells,
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            # ] = -g_slice + 2 * weight * self._east_wall_temperature
            ] = -g_slice + 4 * weight * self._east_wall_temperature * density_slice
        return vel_population, temp_population

    def update_west_wall(self, vel_population: torch.Tensor, temp_population: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Update the the west boundaries with a wall-boundary condition.

        Args:
            population (torch.Tensor): The population that should be updated. It is a (L, Tx, Ty, Tz) tensor,
                                       where Tx, Ty, and Tz denote the total number of cells of the computational domain.
                                       L is the number of discrete velocities of the underlying velocity set.
            density (torch.Tensor): The density field of the block that should be updated. Used to impose wall velocities.
                                    It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The updated population. It is a (L, Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells
                          of the computational domain.
                          L is the number of discrete velocities of the underlying velocity set.
        """
        for west_index in self.west_velocities:
            opposite_index = self.opposite_lattice_indices[west_index]
            f_slice = vel_population[
                west_index,
                self.access_indices[0][0] + self.num_halo_cells + self.lattice_velocity_integers[0][west_index]:
                self.access_indices[0][1] + self.num_halo_cells + self.lattice_velocity_integers[0][west_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][west_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][west_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][west_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][west_index] if self.dimension == 3 else 1,
            ]
            density_slice = density[
                self.access_indices[0][0] + self.num_halo_cells:
                self.access_indices[0][1] + self.num_halo_cells,
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.dot(self.west_wall_velocity_const, self.lattice_velocity_const[:, west_index])
            weight = self.lattice_weights_const[west_index]
            vel_population[
                opposite_index,
                self.access_indices[0][0] + self.num_halo_cells:
                self.access_indices[0][1] + self.num_halo_cells,
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ] = f_slice - 6.0 * weight * density_slice * wall_velocity_projection
            g_slice = temp_population[
                west_index,
                self.access_indices[0][0] + self.num_halo_cells + self.lattice_velocity_integers[0][west_index]:
                self.access_indices[0][1] + self.num_halo_cells + self.lattice_velocity_integers[0][west_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][west_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][west_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][west_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][west_index] if self.dimension == 3 else 1,
            ]
            temp_population[
                opposite_index,
                self.access_indices[0][0] + self.num_halo_cells:
                self.access_indices[0][1] + self.num_halo_cells,
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            # ] = -g_slice + 2 * weight * self._west_wall_temperature
            ] = -g_slice + 4 * weight * self._west_wall_temperature * density_slice
            # print(self._west_wall_temperature)
        return vel_population, temp_population

    def update_north_wall(self, vel_population: torch.Tensor, temp_population: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Update the the north boundaries with a wall-boundary condition.

        Args:
            population (torch.Tensor): The population that should be updated. It is a (L, Tx, Ty, Tz) tensor,
                                       where Tx, Ty, and Tz denote the total number of cells of the computational domain.
                                       L is the number of discrete velocities of the underlying velocity set.
            density (torch.Tensor): The density field of the block that should be updated. Used to impose wall velocities.
                                    It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The updated population. It is a (L, Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells
                          of the computational domain.
                          L is the number of discrete velocities of the underlying velocity set.
        """
        for north_index in self.north_velocities:
            opposite_index = self.opposite_lattice_indices[north_index]
            f_slice = vel_population[
                north_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][north_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][north_index],
                self.access_indices[1][2] - self.num_halo_cells + self.lattice_velocity_integers[1][north_index] if self.dimension != 1 else 0:
                self.access_indices[1][3] - self.num_halo_cells + self.lattice_velocity_integers[1][north_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][north_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][north_index] if self.dimension == 3 else 1,
            ]
            density_slice = density[
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][2] - self.num_halo_cells if self.dimension != 1 else 0:
                self.access_indices[1][3] - self.num_halo_cells if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.dot(self.north_wall_velocity_const, self.lattice_velocity_const[:, north_index])
            weight = self.lattice_weights_const[north_index]
            vel_population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][2] - self.num_halo_cells if self.dimension != 1 else 0:
                self.access_indices[1][3] - self.num_halo_cells if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ] = f_slice - 6.0 * weight * density_slice * wall_velocity_projection
            g_slice = temp_population[
                north_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][north_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][north_index],
                self.access_indices[1][2] - self.num_halo_cells + self.lattice_velocity_integers[1][north_index] if self.dimension != 1 else 0:
                self.access_indices[1][3] - self.num_halo_cells + self.lattice_velocity_integers[1][north_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][north_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][north_index] if self.dimension == 3 else 1,
            ]
            temp_population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][2] - self.num_halo_cells if self.dimension != 1 else 0:
                self.access_indices[1][3] - self.num_halo_cells if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ] = -g_slice + 2 * weight * self._north_wall_temperature
        return vel_population, temp_population

    def update_south_wall(self, vel_population: torch.Tensor, temp_population: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Update the the south boundaries with a wall-boundary condition.

        Args:
            population (torch.Tensor): The population that should be updated. It is a (L, Tx, Ty, Tz) tensor,
                                       where Tx, Ty, and Tz denote the total number of cells of the computational domain.
                                       L is the number of discrete velocities of the underlying velocity set.
            density (torch.Tensor): The density field of the block that should be updated. Used to impose wall velocities.
                                    It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The updated population. It is a (L, Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells
                          of the computational domain.
                          L is the number of discrete velocities of the underlying velocity set.
        """
        for south_index in self.south_velocities:
            opposite_index = self.opposite_lattice_indices[south_index]
            f_slice = vel_population[
                south_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][south_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][south_index],
                self.access_indices[1][0] + self.num_halo_cells + self.lattice_velocity_integers[1][south_index] if self.dimension != 1 else 0:
                self.access_indices[1][1] + self.num_halo_cells + self.lattice_velocity_integers[1][south_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][south_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][south_index] if self.dimension == 3 else 1,
            ]
            density_slice = density[
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][0] + self.num_halo_cells if self.dimension != 1 else 0:
                self.access_indices[1][1] + self.num_halo_cells if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.dot(self.south_wall_velocity_const, self.lattice_velocity_const[:, south_index])
            weight = self.lattice_weights_const[south_index]
            vel_population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][0] + self.num_halo_cells if self.dimension != 1 else 0:
                self.access_indices[1][1] + self.num_halo_cells if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ] = f_slice - 6.0 * weight * density_slice * wall_velocity_projection
            g_slice = temp_population[
                south_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][south_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][south_index],
                self.access_indices[1][0] + self.num_halo_cells + self.lattice_velocity_integers[1][south_index] if self.dimension != 1 else 0:
                self.access_indices[1][1] + self.num_halo_cells + self.lattice_velocity_integers[1][south_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][south_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][south_index] if self.dimension == 3 else 1,
            ]
            temp_population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][0] + self.num_halo_cells if self.dimension != 1 else 0:
                self.access_indices[1][1] + self.num_halo_cells if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ] = -g_slice + 2 * weight * self._south_wall_temperature
        return vel_population, temp_population

    def update_top_wall(self, vel_population: torch.Tensor, temp_population: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Update the the top boundaries with a wall-boundary condition.

        Args:
            population (torch.Tensor): The population that should be updated. It is a (L, Tx, Ty, Tz) tensor,
                                       where Tx, Ty, and Tz denote the total number of cells of the computational domain.
                                       L is the number of discrete velocities of the underlying velocity set.
            density (torch.Tensor): The density field of the block that should be updated. Used to impose wall velocities.
                                    It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The updated population. It is a (L, Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells
                          of the computational domain.
                          L is the number of discrete velocities of the underlying velocity set.
        """
        for top_index in self.top_velocities:
            opposite_index = self.opposite_lattice_indices[top_index]
            f_slice = vel_population[
                top_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][top_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][top_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][top_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][top_index] if self.dimension != 1 else 1,
                self.access_indices[2][2] - self.num_halo_cells + self.lattice_velocity_integers[2][top_index] if self.dimension == 3 else 0:
                self.access_indices[2][3] - self.num_halo_cells + self.lattice_velocity_integers[2][top_index] if self.dimension == 3 else 1,
            ]
            density_slice = density[
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][2] - self.num_halo_cells if self.dimension == 3 else 0:
                self.access_indices[2][3] - self.num_halo_cells if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.dot(self.top_wall_velocity_const, self.lattice_velocity_const[:, top_index])
            weight = self.lattice_weights_const[top_index]
            vel_population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][2] - self.num_halo_cells if self.dimension == 3 else 0:
                self.access_indices[2][3] - self.num_halo_cells if self.dimension == 3 else 1,
            ] = f_slice - 6.0 * weight * density_slice * wall_velocity_projection
            g_slice = temp_population[
                top_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][top_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][top_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][top_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][top_index] if self.dimension != 1 else 1,
                self.access_indices[2][2] - self.num_halo_cells + self.lattice_velocity_integers[2][top_index] if self.dimension == 3 else 0:
                self.access_indices[2][3] - self.num_halo_cells + self.lattice_velocity_integers[2][top_index] if self.dimension == 3 else 1,
            ]
            temp_population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][2] - self.num_halo_cells if self.dimension == 3 else 0:
                self.access_indices[2][3] - self.num_halo_cells if self.dimension == 3 else 1,
            ] = -g_slice + 2 * weight * self._top_wall_temperature
        return vel_population, temp_population

    def update_bottom_wall(self, vel_population: torch.Tensor, temp_population: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Update the the bottom boundaries with a wall-boundary condition.

        Args:
            population (torch.Tensor): The population that should be updated. It is a (L, Tx, Ty, Tz) tensor,
                                       where Tx, Ty, and Tz denote the total number of cells of the computational domain.
                                       L is the number of discrete velocities of the underlying velocity set.
            density (torch.Tensor): The density field of the block that should be updated. Used to impose wall velocities.
                                    It is a (Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells of the computational domain.

        Returns:
            torch.Tensor: The updated population. It is a (L, Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote the total number of cells
                          of the computational domain.
                          L is the number of discrete velocities of the underlying velocity set.
        """
        for bottom_index in self.bottom_velocities:
            opposite_index = self.opposite_lattice_indices[bottom_index]
            f_slice = vel_population[
                bottom_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][bottom_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][bottom_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][bottom_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][bottom_index] if self.dimension != 1 else 1,
                self.access_indices[2][0] + self.num_halo_cells + self.lattice_velocity_integers[2][bottom_index] if self.dimension == 3 else 0:
                self.access_indices[2][1] + self.num_halo_cells + self.lattice_velocity_integers[2][bottom_index] if self.dimension == 3 else 1,
            ]
            density_slice = density[
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][0] + self.num_halo_cells if self.dimension == 3 else 0:
                self.access_indices[2][1] + self.num_halo_cells if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.dot(self.bottom_wall_velocity_const, self.lattice_velocity_const[:, bottom_index])
            weight = self.lattice_weights_const[bottom_index]
            vel_population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][0] + self.num_halo_cells if self.dimension == 3 else 0:
                self.access_indices[2][1] + self.num_halo_cells if self.dimension == 3 else 1,
            ] = f_slice - 6.0 * weight * density_slice * wall_velocity_projection
            g_slice = temp_population[
                bottom_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][bottom_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][bottom_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][bottom_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][bottom_index] if self.dimension != 1 else 1,
                self.access_indices[2][0] + self.num_halo_cells + self.lattice_velocity_integers[2][bottom_index] if self.dimension == 3 else 0:
                self.access_indices[2][1] + self.num_halo_cells + self.lattice_velocity_integers[2][bottom_index] if self.dimension == 3 else 1,
            ]
            temp_population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][0] + self.num_halo_cells if self.dimension == 3 else 0:
                self.access_indices[2][1] + self.num_halo_cells if self.dimension == 3 else 1,
            ] = -g_slice + 2 * weight * self._bottom_wall_temperature
        return vel_population, temp_population
    # fmt: on

    def forward(self, node_data: ThermalNodeData) -> ThermalNodeData:
        """Performs the wall boundary update as the forwards pass of the PyTorch module.

        Args:
            node_data (NodeData): The node data that contains the storage intensive fields for macroscopic and microscopic quantities.

        Returns:
            NodeData: The node data that contains the storage intensive fields for macroscopic and microscopic quantities which are already
                      updated according to the wall boundary condition.
        """
        if self.is_east_wall:
            node_data.distributions.vel_old_population, node_data.distributions.temp_old_population = self.update_east_wall(node_data.distributions.vel_old_population, node_data.distributions.temp_old_population, node_data.moments.density)
        if self.is_west_wall:
            node_data.distributions.vel_old_population, node_data.distributions.temp_old_population = self.update_west_wall(node_data.distributions.vel_old_population, node_data.distributions.temp_old_population, node_data.moments.density)

        if self.is_north_wall:
            node_data.distributions.vel_old_population, node_data.distributions.temp_old_population = self.update_north_wall(node_data.distributions.vel_old_population, node_data.distributions.temp_old_population, node_data.moments.density)
        if self.is_south_wall:
            node_data.distributions.vel_old_population, node_data.distributions.temp_old_population = self.update_south_wall(node_data.distributions.vel_old_population, node_data.distributions.temp_old_population, node_data.moments.density)

        if self.is_top_wall:
            node_data.distributions.vel_old_population, node_data.distributions.temp_old_population = self.update_top_wall(node_data.distributions.vel_old_population, node_data.distributions.temp_old_population, node_data.moments.density)
        if self.is_bottom_wall:
            node_data.distributions.vel_old_population, node_data.distributions.temp_old_population = self.update_bottom_wall(node_data.distributions.vel_old_population, node_data.distributions.temp_old_population, node_data.moments.density)

        return node_data
