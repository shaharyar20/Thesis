from typing import List

import torch
import torch.nn as nn
from dataclasses import dataclass

from torchlbm.node_data import NodeData


class EquilibriumBoundaryUpdate(nn.Module):
    """The class that applies boundary condtions for a moving wall. In case a zero wall velocity is chosen,
    it models a rigid and non-moving wall.

    Args:
        nn (nn.Module): The wall boundary update operator is implemented as a PyTorch module.
                        This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(
        self,
        is_east_equilibrium: bool,
        east_equilibrium_velocity: torch.Tensor,
        east_equilibrium_density: float,
        is_west_equilibrium: bool,
        west_equilibrium_velocity: torch.Tensor,
        west_equilibrium_density: float,
        is_north_equilibrium: bool,
        north_equilibrium_velocity: torch.Tensor,
        north_equilibrium_density: float,
        is_south_equilibrium: bool,
        south_equilibrium_velocity: torch.Tensor,
        south_equilibrium_density: float,
        is_top_equilibrium: bool,
        top_equilibrium_velocity: torch.Tensor,
        top_equilibrium_density: float,
        is_bottom_equilibrium: bool,
        bottom_equilibrium_velocity: torch.Tensor,
        bottom_equilibrium_density: float,
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
        super(EquilibriumBoundaryUpdate, self).__init__()

        self.is_east_equilibrium = is_east_equilibrium
        self.east_equilibrium_velocity = east_equilibrium_velocity
        self.register_buffer("east_equilibrium_velocity_const", self.east_equilibrium_velocity)
        self.east_equilibrium_density = east_equilibrium_density
        self.is_west_equilibrium = is_west_equilibrium
        self.west_equilibrium_velocity = west_equilibrium_velocity
        self.register_buffer("west_equilibrium_velocity_const", self.west_equilibrium_velocity)
        self.west_equilibrium_density = west_equilibrium_density
        self.is_north_equilibrium = is_north_equilibrium
        self.north_equilibrium_velocity = north_equilibrium_velocity
        self.register_buffer("north_equilibrium_velocity_const", self.north_equilibrium_velocity)
        self.north_equilibrium_density = north_equilibrium_density
        self.is_south_equilibrium = is_south_equilibrium
        self.south_equilibrium_velocity = south_equilibrium_velocity
        self.register_buffer("south_equilibrium_velocity_const", self.south_equilibrium_velocity)
        self.south_equilibrium_density = south_equilibrium_density
        self.is_top_equilibrium = is_top_equilibrium
        self.top_equilibrium_velocity = top_equilibrium_velocity
        self.register_buffer("top_equilibrium_velocity_const", self.top_equilibrium_velocity)
        self.top_equilibrium_density = top_equilibrium_density
        self.is_bottom_equilibrium = is_bottom_equilibrium
        self.bottom_equilibrium_velocity = bottom_equilibrium_velocity
        self.register_buffer("bottom_equilibrium_velocity_const", self.bottom_equilibrium_velocity)
        self.bottom_equilibrium_density = bottom_equilibrium_density

        self.num_halo_cells = num_halo_cells
        self.access_indices = access_indices
        self.dimension = dimension

        self._lattice_velocity = lattice_velocity
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

    def calculate_equilibrium(self, density: float, velocity: torch.Tensor) -> torch.Tensor:
        projected_discrete_velocities = torch.einsum(
            "dQ,d->Q",
            self.lattice_velocity_const,
            velocity,
        )
        macroscopic_velocity_magnitude = torch.linalg.norm(
            velocity,
            ord=2,
            dim=0,
        )
        equilibrium_population = (
            density
            * self.lattice_weights_const
            * (1 + 3 * projected_discrete_velocities + 9 / 2 * projected_discrete_velocities**2 - 3 / 2 * macroscopic_velocity_magnitude.unsqueeze(0) ** 2)
        )
        return equilibrium_population



    # fmt: off
    def update_east_wall(self, population: torch.Tensor) -> torch.Tensor:
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
        population[
            :,
            self.access_indices[0][2] - self.num_halo_cells:
            self.access_indices[0][3] - self.num_halo_cells,
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][1] if self.dimension == 3 else 0:
            self.access_indices[2][2] if self.dimension == 3 else 1,
        ] = self.calculate_equilibrium( self.east_equilibrium_density, self.east_equilibrium_velocity_const).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return population

    def update_west_wall(self, population: torch.Tensor) -> torch.Tensor:
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
        population[
            :,
            self.access_indices[0][0] + self.num_halo_cells:
            self.access_indices[0][1] + self.num_halo_cells,
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][1] if self.dimension == 3 else 0:
            self.access_indices[2][2] if self.dimension == 3 else 1,
        ] = self.calculate_equilibrium( self.west_equilibrium_density, self.west_equilibrium_velocity_const).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return population
        

    def update_north_wall(self, population: torch.Tensor) -> torch.Tensor:
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
        
        population[
            :,
            self.access_indices[0][1]:
            self.access_indices[0][2],
            self.access_indices[1][2] - self.num_halo_cells if self.dimension != 1 else 0:
            self.access_indices[1][3] - self.num_halo_cells if self.dimension != 1 else 1,
            self.access_indices[2][1] if self.dimension == 3 else 0:
            self.access_indices[2][2] if self.dimension == 3 else 1,
        ] = self.calculate_equilibrium( self.north_equilibrium_density, self.north_equilibrium_velocity_const).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return population

    def update_south_wall(self, population: torch.Tensor) -> torch.Tensor:
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
        population[
            :,
            self.access_indices[0][1]:
            self.access_indices[0][2],
            self.access_indices[1][0] + self.num_halo_cells if self.dimension != 1 else 0:
            self.access_indices[1][1] + self.num_halo_cells if self.dimension != 1 else 1,
            self.access_indices[2][1] if self.dimension == 3 else 0:
            self.access_indices[2][2] if self.dimension == 3 else 1,
        ] = self.calculate_equilibrium( self.south_equilibrium_density, self.south_equilibrium_velocity_const).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return population

    def update_top_wall(self, population: torch.Tensor) -> torch.Tensor:
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
        
        population[
            :,
            self.access_indices[0][1]:
            self.access_indices[0][2],
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][2] - self.num_halo_cells if self.dimension == 3 else 0:
            self.access_indices[2][3] - self.num_halo_cells if self.dimension == 3 else 1,
        ] = self.calculate_equilibrium( self.top_equilibrium_density, self.top_equilibrium_velocity_const).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return population

    def update_bottom_wall(self, population: torch.Tensor) -> torch.Tensor:
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
        
        population[
            :,
            self.access_indices[0][1]:
            self.access_indices[0][2],
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][0] + self.num_halo_cells if self.dimension == 3 else 0:
            self.access_indices[2][1] + self.num_halo_cells if self.dimension == 3 else 1,
        ] = self.calculate_equilibrium( self.bottom_equilibrium_density, self.bottom_equilibrium_velocity_const).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return population
    # fmt: on

    def forward(self, old_population: torch.Tensor, density: torch.Tensor, velocity: torch.Tensor, bounce_back_mask: torch.Tensor) -> torch.Tensor:
        """Performs the wall boundary update as the forwards pass of the PyTorch module.

        Args:
            node_data (NodeData): The node data that contains the storage intensive fields for macroscopic and microscopic quantities.

        Returns:
            NodeData: The node data that contains the storage intensive fields for macroscopic and microscopic quantities which are already
                      updated according to the wall boundary condition.
        """
        population = old_population.clone()
        if self.is_east_equilibrium:
            population = self.update_east_wall(population)
        if self.is_west_equilibrium:
            population = self.update_west_wall(population)

        if self.is_north_equilibrium:
            population = self.update_north_wall(population)
        if self.is_south_equilibrium:
            population = self.update_south_wall(population)

        if self.is_top_equilibrium:
            population = self.update_top_wall(population)
        if self.is_bottom_equilibrium:
            population = self.update_bottom_wall(population)

        return population
