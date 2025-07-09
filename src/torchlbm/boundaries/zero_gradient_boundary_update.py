from typing import List

import torch
import torch.nn as nn
from dataclasses import dataclass

from torchlbm.node_data import NodeData


class ZeroGradientBoundaryUpdate(nn.Module):
    """The class that applies boundary condtions for a moving wall. In case a zero wall velocity is chosen,
    it models a rigid and non-moving wall.

    Args:
        nn (nn.Module): The wall boundary update operator is implemented as a PyTorch module.
                        This allows easily composing algorithms based on consecutive modules.
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
        access_indices: List[List[int]],
        dimension: int,
        lattice_velocity: torch.Tensor,
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
            is_east_zero_gradient (bool): Inidicator whether the east part of the computational domain is a wall.
            is_west_zero_gradient (bool): Inidicator whether the west part of the computational domain is a wall.
            is_north_zero_gradient (bool): Inidicator whether the north part of the computational domain is a wall.
            is_south_zero_gradient (bool): Inidicator whether the south part of the computational domain is a wall.
            is_top_zero_gradient (bool): Inidicator whether the top part of the computational domain is a wall.
            is_bottom_zero_gradient (bool): Inidicator whether the bottom part of the computational domain is a wall.
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
        super(ZeroGradientBoundaryUpdate, self).__init__()

        self.is_east_zero_gradient = is_east_zero_gradient
        self.is_west_zero_gradient = is_west_zero_gradient
        self.is_north_zero_gradient = is_north_zero_gradient
        self.is_south_zero_gradient = is_south_zero_gradient
        self.is_top_zero_gradient = is_top_zero_gradient
        self.is_bottom_zero_gradient = is_bottom_zero_gradient
        self.num_halo_cells = num_halo_cells
        self.access_indices = access_indices
        self.dimension = dimension

        self.lattice_velocity = lattice_velocity
        self.register_buffer("lattice_velocity_const", self.lattice_velocity)
        self.lattice_velocity_integers = lattice_velocity.clone().detach().int()
        self.lattice_velocity_integers = self.lattice_velocity_integers.tolist()
        self.lattice_weights = lattice_weights
        self.east_velocities = east_velocities
        self.west_velocities = west_velocities
        self.north_velocities = north_velocities
        self.south_velocities = south_velocities
        self.top_velocities = top_velocities
        self.bottom_velocities = bottom_velocities
        self.opposite_lattice_indices = opposite_lattice_indices

    # fmt: off
    def update_east_zero_gradient(self, population: torch.Tensor, velocity: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
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
        density_slice = density[
            self.access_indices[0][2] - self.num_halo_cells:
            self.access_indices[0][3] - self.num_halo_cells,
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][1] if self.dimension == 3 else 0:
            self.access_indices[2][2] if self.dimension == 3 else 1,
        ]
        velocity_slice = velocity[
            :,
            self.access_indices[0][2] - self.num_halo_cells:
            self.access_indices[0][3] - self.num_halo_cells,
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][1] if self.dimension == 3 else 0:
            self.access_indices[2][2] if self.dimension == 3 else 1,
        ]
        square_velocity_projection = torch.einsum("dNML,dNML->NML", velocity_slice, velocity_slice)
        for east_index in self.east_velocities:
            opposite_index = self.opposite_lattice_indices[east_index]
            f_slice = population[
                east_index,
                self.access_indices[0][2] - self.num_halo_cells + self.lattice_velocity_integers[0][east_index]:
                self.access_indices[0][3] - self.num_halo_cells + self.lattice_velocity_integers[0][east_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][east_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][east_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][east_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][east_index] if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.einsum("dNML,d->NML", velocity_slice, self.lattice_velocity_const[:, east_index])
            weight = self.lattice_weights[east_index]
            population[
                opposite_index,
                self.access_indices[0][2] - self.num_halo_cells:
                self.access_indices[0][3] - self.num_halo_cells,
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ] = - f_slice + 2.0 * weight * density_slice * (1.0 + 0.5 * 9.0 * wall_velocity_projection**2 - 0.5 * 3.0 * square_velocity_projection)
        # population[:, -self.num_halo_cells :, :, :] = population[:, -2 * self.num_halo_cells : -self.num_halo_cells, :, :]
        return population

    def update_west_zero_gradient(self, population: torch.Tensor, velocity: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
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
        density_slice = density[
            self.access_indices[0][0] + self.num_halo_cells:
            self.access_indices[0][1] + self.num_halo_cells,
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][1] if self.dimension == 3 else 0:
            self.access_indices[2][2] if self.dimension == 3 else 1,
        ]
        velocity_slice = velocity[
            :,
            self.access_indices[0][0] + self.num_halo_cells:
            self.access_indices[0][1] + self.num_halo_cells,
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][1] if self.dimension == 3 else 0:
            self.access_indices[2][2] if self.dimension == 3 else 1,
        ]
        square_velocity_projection = torch.einsum("dNML,dNML->NML", velocity_slice, velocity_slice)
        for west_index in self.west_velocities:
            opposite_index = self.opposite_lattice_indices[west_index]
            f_slice = population[
                west_index,
                self.access_indices[0][0] + self.num_halo_cells + self.lattice_velocity_integers[0][west_index]:
                self.access_indices[0][1] + self.num_halo_cells + self.lattice_velocity_integers[0][west_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][west_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][west_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][west_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][west_index] if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.einsum("dNML,d->NML", velocity_slice, self.lattice_velocity_const[:, west_index])
            weight = self.lattice_weights[west_index]
            population[
                opposite_index,
                self.access_indices[0][0] + self.num_halo_cells:
                self.access_indices[0][1] + self.num_halo_cells,
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ] = - f_slice + 2.0 * weight * density_slice * (1.0 + 0.5 * 9.0 * wall_velocity_projection**2 - 0.5 * 3.0 * square_velocity_projection)
        # population[:, : self.num_halo_cells, :, :] = population[:, self.num_halo_cells : 2 * self.num_halo_cells, :, :]
        return population

    def update_north_zero_gradient(self, population: torch.Tensor, velocity: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
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
        # density_slice = density[
        #     self.access_indices[0][1]:
        #     self.access_indices[0][2],
        #     self.access_indices[1][2] - self.num_halo_cells if self.dimension != 1 else 0:
        #     self.access_indices[1][3] - self.num_halo_cells if self.dimension != 1 else 1,
        #     self.access_indices[2][1] if self.dimension == 3 else 0:
        #     self.access_indices[2][2] if self.dimension == 3 else 1,
        # ]
        # velocity_slice = velocity[
        #     :,
        #     self.access_indices[0][1]:
        #     self.access_indices[0][2],
        #     self.access_indices[1][2] - self.num_halo_cells if self.dimension != 1 else 0:
        #     self.access_indices[1][3] - self.num_halo_cells if self.dimension != 1 else 1,
        #     self.access_indices[2][1] if self.dimension == 3 else 0:
        #     self.access_indices[2][2] if self.dimension == 3 else 1,
        # ]
        # square_velocity_projection = torch.einsum("dNML,dNML->NML", velocity_slice, velocity_slice)
        # for north_index in self.north_velocities:
        #     opposite_index = self.opposite_lattice_indices[north_index]
        #     f_slice = population[
        #         north_index,
        #         self.access_indices[0][1] + self.lattice_velocity_integers[0][north_index]:
        #         self.access_indices[0][2] + self.lattice_velocity_integers[0][north_index],
        #         self.access_indices[1][2] - self.num_halo_cells + self.lattice_velocity_integers[1][north_index] if self.dimension != 1 else 0:
        #         self.access_indices[1][3] - self.num_halo_cells + self.lattice_velocity_integers[1][north_index] if self.dimension != 1 else 1,
        #         self.access_indices[2][1] + self.lattice_velocity_integers[2][north_index] if self.dimension == 3 else 0:
        #         self.access_indices[2][2] + self.lattice_velocity_integers[2][north_index] if self.dimension == 3 else 1,
        #     ]
        #     wall_velocity_projection = torch.einsum("dNML,d->NML", velocity_slice, self.lattice_velocity_const[:, north_index])
        #     weight = self.lattice_weights[north_index]
        #     population[
        #         opposite_index,
        #         self.access_indices[0][1]:
        #         self.access_indices[0][2],
        #         self.access_indices[1][2] - self.num_halo_cells if self.dimension != 1 else 0:
        #         self.access_indices[1][3] - self.num_halo_cells if self.dimension != 1 else 1,
        #         self.access_indices[2][1] if self.dimension == 3 else 0:
        #         self.access_indices[2][2] if self.dimension == 3 else 1,
        #     ] = - f_slice + 2.0 * weight * density_slice * (1.0 + 0.5 * 9.0 * wall_velocity_projection**2 - 0.5 * 3.0 * square_velocity_projection)
        population[:, :, -self.num_halo_cells :, :] = population[:, :, -2 * self.num_halo_cells : -self.num_halo_cells, :]
        return population

    def update_south_zero_gradient(self, population: torch.Tensor, velocity: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
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
        density_slice = density[
            self.access_indices[0][1]:
            self.access_indices[0][2],
            self.access_indices[1][0] + self.num_halo_cells if self.dimension != 1 else 0:
            self.access_indices[1][1] + self.num_halo_cells if self.dimension != 1 else 1,
            self.access_indices[2][1] if self.dimension == 3 else 0:
            self.access_indices[2][2] if self.dimension == 3 else 1,
        ]
        velocity_slice = velocity[
            :,
            self.access_indices[0][1]:
            self.access_indices[0][2],
            self.access_indices[1][0] + self.num_halo_cells if self.dimension != 1 else 0:
            self.access_indices[1][1] + self.num_halo_cells if self.dimension != 1 else 1,
            self.access_indices[2][1] if self.dimension == 3 else 0:
            self.access_indices[2][2] if self.dimension == 3 else 1,
        ]
        square_velocity_projection = torch.einsum("dNML,dNML->NML", velocity_slice, velocity_slice)
        for south_index in self.south_velocities:
            opposite_index = self.opposite_lattice_indices[south_index]
            f_slice = population[
                south_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][south_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][south_index],
                self.access_indices[1][0] + self.num_halo_cells + self.lattice_velocity_integers[1][south_index] if self.dimension != 1 else 0:
                self.access_indices[1][1] + self.num_halo_cells + self.lattice_velocity_integers[1][south_index] if self.dimension != 1 else 1,
                self.access_indices[2][1] + self.lattice_velocity_integers[2][south_index] if self.dimension == 3 else 0:
                self.access_indices[2][2] + self.lattice_velocity_integers[2][south_index] if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.einsum("dNML,d->NML", velocity_slice, self.lattice_velocity_const[:, south_index])
            weight = self.lattice_weights[south_index]
            population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][0] + self.num_halo_cells if self.dimension != 1 else 0:
                self.access_indices[1][1] + self.num_halo_cells if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0:
                self.access_indices[2][2] if self.dimension == 3 else 1,
            ] = - f_slice + 2.0 * weight * density_slice * (1.0 + 0.5 * 9.0 * wall_velocity_projection**2 - 0.5 * 3.0 * square_velocity_projection)
        return population

    def update_top_zero_gradient(self, population: torch.Tensor, velocity: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
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
        density_slice = density[
            self.access_indices[0][1]:
            self.access_indices[0][2],
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][2] - self.num_halo_cells if self.dimension == 3 else 0:
            self.access_indices[2][3] - self.num_halo_cells if self.dimension == 3 else 1,
        ]
        velocity_slice = velocity[
            :,
            self.access_indices[0][1]:
            self.access_indices[0][2],
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][2] - self.num_halo_cells if self.dimension == 3 else 0:
            self.access_indices[2][3] - self.num_halo_cells if self.dimension == 3 else 1,
        ]
        square_velocity_projection = torch.einsum("dNML,dNML->NML", velocity_slice, velocity_slice)
        for top_index in self.top_velocities:
            opposite_index = self.opposite_lattice_indices[top_index]
            f_slice = population[
                top_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][top_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][top_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][top_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][top_index] if self.dimension != 1 else 1,
                self.access_indices[2][2] - self.num_halo_cells + self.lattice_velocity_integers[2][top_index] if self.dimension == 3 else 0:
                self.access_indices[2][3] - self.num_halo_cells + self.lattice_velocity_integers[2][top_index] if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.einsum("dNML,d->NML", velocity_slice, self.lattice_velocity_const[:, top_index])
            weight = self.lattice_weights[top_index]
            population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][2] - self.num_halo_cells if self.dimension == 3 else 0:
                self.access_indices[2][3] - self.num_halo_cells if self.dimension == 3 else 1,
            ] = - f_slice + 2.0 * weight * density_slice * (1.0 + 0.5 * 9.0 * wall_velocity_projection**2 - 0.5 * 3.0 * square_velocity_projection)
        return population

    def update_bottom_zero_gradient(self, population: torch.Tensor, velocity: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
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
        density_slice = density[
            self.access_indices[0][1]:
            self.access_indices[0][2],
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][0] + self.num_halo_cells if self.dimension == 3 else 0:
            self.access_indices[2][1] + self.num_halo_cells if self.dimension == 3 else 1,
        ]
        velocity_slice = velocity[
            :,
            self.access_indices[0][1]:
            self.access_indices[0][2],
            self.access_indices[1][1] if self.dimension != 1 else 0:
            self.access_indices[1][2] if self.dimension != 1 else 1,
            self.access_indices[2][0] + self.num_halo_cells if self.dimension == 3 else 0:
            self.access_indices[2][1] + self.num_halo_cells if self.dimension == 3 else 1,
        ]
        square_velocity_projection = torch.einsum("dNML,dNML->NML", velocity_slice, velocity_slice)
        for bottom_index in self.bottom_velocities:
            opposite_index = self.opposite_lattice_indices[bottom_index]
            f_slice = population[
                bottom_index,
                self.access_indices[0][1] + self.lattice_velocity_integers[0][bottom_index]:
                self.access_indices[0][2] + self.lattice_velocity_integers[0][bottom_index],
                self.access_indices[1][1] + self.lattice_velocity_integers[1][bottom_index] if self.dimension != 1 else 0:
                self.access_indices[1][2] + self.lattice_velocity_integers[1][bottom_index] if self.dimension != 1 else 1,
                self.access_indices[2][0] + self.num_halo_cells + self.lattice_velocity_integers[2][bottom_index] if self.dimension == 3 else 0:
                self.access_indices[2][1] + self.num_halo_cells + self.lattice_velocity_integers[2][bottom_index] if self.dimension == 3 else 1,
            ]
            wall_velocity_projection = torch.einsum("dNML,d->NML", velocity_slice, self.lattice_velocity_const[:, bottom_index])
            weight = self.lattice_weights[bottom_index]
            population[
                opposite_index,
                self.access_indices[0][1]:
                self.access_indices[0][2],
                self.access_indices[1][1] if self.dimension != 1 else 0:
                self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][0] + self.num_halo_cells if self.dimension == 3 else 0:
                self.access_indices[2][1] + self.num_halo_cells if self.dimension == 3 else 1,
            ] = - f_slice + 2.0 * weight * density_slice * (1.0 + 0.5 * 9.0 * wall_velocity_projection**2 - 0.5 * 3.0 * square_velocity_projection)
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
        if self.is_east_zero_gradient:
            population = self.update_east_zero_gradient(
                population, velocity, density
            )
        if self.is_west_zero_gradient:
            population = self.update_west_zero_gradient(
                population, velocity, density
            )

        if self.is_north_zero_gradient:
            population = self.update_north_zero_gradient(
                population, velocity, density
            )
        if self.is_south_zero_gradient:
            population = self.update_south_zero_gradient(
                population, velocity, density
            )

        if self.is_top_zero_gradient:
            population = self.update_top_zero_gradient(
                population, velocity, density
            )
        if self.is_bottom_zero_gradient:
            population = self.update_bottom_zero_gradient(
                population, velocity, density
            )

        return population
