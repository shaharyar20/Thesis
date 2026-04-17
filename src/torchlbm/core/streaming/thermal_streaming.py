from typing import List

import torch
import torch.nn as nn
from dataclasses import dataclass

# from torchlbm.node_data import NodeData
from torchlbm.thermal_node_data import ThermalNodeData


class ThermalStreamingModule(nn.Module):
    """The PyTorch module used to perform the streaming operation of the Lattice-Boltzmann method.

    Args:
        nn (nn.Modules): The base class. The StreamingModule is implemented as a PyTorch module.
                         This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, number_of_discrete_velocities: int, lattice_velocities: List[List[float]], num_total_cells: List[int]) -> None:
        """The initializer.

        Args:
            number_of_discrete_velocities (int): The number of velocities of the velocity set used to discretize the velocity distribution.
            lattice_velocities (List[List[float]]): The velocities of the velocity set as a two-dimension List which can be converted
                                                    to a tensor of arbitrary type.
        """
        super(ThermalStreamingModule, self).__init__()
        self.number_of_discrete_velocities = number_of_discrete_velocities
        self.lattice_velocities = torch.tensor(lattice_velocities).clone().detach().int()
        self.lattice_velocities = self.lattice_velocities.tolist()
        self.num_total_cells = num_total_cells

    def forward(self, node_data: ThermalNodeData) -> ThermalNodeData:
        """The forward pass which performs the streaming operation.

        Args:
            node_data (NodeData): The node data that contains the storage intensive fields for macroscopic and microscopic quantities.
                                  Only the discrete_velocities are used. They are a (L x Nx x Ny x Nz) tensor,
                                  where L denotes the number of lattice velocities, and Nx, Ny, and Nz the number of cells
                                  in x-, y-, and z-direction, respectively.

        Returns:
            NodeData: The node data after streaming
        """
        # discrete_velocities = node_data.distributions.old_population
        # for i in range(self.number_of_discrete_velocities):
        #     discrete_velocities[i, :, :, :] = torch.roll(
        #         discrete_velocities[i, :, :, :],
        #         shifts=[self.lattice_velocities[0][i],
        #                 self.lattice_velocities[1][i],
        #                 self.lattice_velocities[2][i]],
        #         dims=[0, 1, 2],
        #     )
        # node_data.distributions.old_population = discrete_velocities
        # return node_data
        for i in range(self.number_of_discrete_velocities):
            node_data.distributions.vel_new_population[
                i,
                0 + int(self.lattice_velocities[0][i] > 0) : self.num_total_cells[0] - int(self.lattice_velocities[0][i] < 0),
                0 + int(self.lattice_velocities[1][i] > 0) : self.num_total_cells[1] - int(self.lattice_velocities[1][i] < 0),
                0 + int(self.lattice_velocities[2][i] > 0) : self.num_total_cells[2] - int(self.lattice_velocities[2][i] < 0),
            ] = node_data.distributions.vel_old_population[
                i,
                0 + int(self.lattice_velocities[0][i] < 0) : self.num_total_cells[0] - int(self.lattice_velocities[0][i] > 0),
                0 + int(self.lattice_velocities[1][i] < 0) : self.num_total_cells[1] - int(self.lattice_velocities[1][i] > 0),
                0 + int(self.lattice_velocities[2][i] < 0) : self.num_total_cells[2] - int(self.lattice_velocities[2][i] > 0),
            ]
            node_data.distributions.temp_new_population[
                i,
                0 + int(self.lattice_velocities[0][i] > 0) : self.num_total_cells[0] - int(self.lattice_velocities[0][i] < 0),
                0 + int(self.lattice_velocities[1][i] > 0) : self.num_total_cells[1] - int(self.lattice_velocities[1][i] < 0),
                0 + int(self.lattice_velocities[2][i] > 0) : self.num_total_cells[2] - int(self.lattice_velocities[2][i] < 0),
            ] = node_data.distributions.temp_old_population[
                i,
                0 + int(self.lattice_velocities[0][i] < 0) : self.num_total_cells[0] - int(self.lattice_velocities[0][i] > 0),
                0 + int(self.lattice_velocities[1][i] < 0) : self.num_total_cells[1] - int(self.lattice_velocities[1][i] > 0),
                0 + int(self.lattice_velocities[2][i] < 0) : self.num_total_cells[2] - int(self.lattice_velocities[2][i] > 0),
            ]

        node_data.distributions.vel_old_population = node_data.distributions.vel_new_population
        node_data.distributions.temp_old_population = node_data.distributions.temp_new_population
        return node_data
