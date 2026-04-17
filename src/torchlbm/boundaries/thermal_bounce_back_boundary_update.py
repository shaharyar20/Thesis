from typing import List
import torch.nn as nn
import torch
from dataclasses import dataclass

# from torchlbm.node_data import NodeData
from torchlbm.thermal_node_data import ThermalNodeData


class ThermalBounceBackBoundaryUpdate(nn.Module):
    """Implements the bounce back boundary update to model rigid bodies.

    Args:
        nn (nn.Module): The bounce back boundary update operator is implemented as a PyTorch module.
                        This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, opposite_lattice_indices: List[int], lattice_weights, bounce_back_temperature) -> None:
        """Initializer for the bounce back boundary update. Initializes all relevant members. Usually created using a factory function.

        Args:
            opposite_lattice_indices (List[int]): The opposite lattice indices that are used to perform the bounce back.
        """
        super(ThermalBounceBackBoundaryUpdate, self).__init__()
        self.opposite_lattice_indices: List[int] = opposite_lattice_indices
        self.bounce_back_temperature = bounce_back_temperature
        self.lattice_weights = lattice_weights
        self.register_buffer("lattice_weights_const", self.lattice_weights)

    def forward(self, node_data: ThermalNodeData) -> ThermalNodeData:
        """The forward pass of the module that performs the actual bounce back update.

        Args:
            node_data (NodeData): The node data that contains the storage intensive fields for macroscopic and microscopic quantities.
                                  On the discrete velocities are used. Theay are a (L, Tx, Ty, Tz) tensor, where Tx, Ty, and Tz denote
                                  the total number of cells of the computational domain.
                                  L is the number of discrete velocities of the underlying velocity set.

        Returns:
            NodeData: The node data which is updated according to the bounce back algorithm.
        """
        # print("Before bounce back: ", node_data.distributions.temp_old_population.shape)
        discrete_velocities = node_data.distributions.vel_old_population
        discrete_temperatures = node_data.distributions.temp_old_population
        if node_data.bounce_back_mask is not None:
            # print((-discrete_temperatures[self.opposite_lattice_indices]).shape)
            # print((2*self.lattice_weights_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*self.bounce_back_temperature).shape)
            discrete_velocities = torch.where(node_data.bounce_back_mask > 0, discrete_velocities[self.opposite_lattice_indices], discrete_velocities)
            discrete_temperatures = torch.where(
                node_data.bounce_back_mask > 0,
                -discrete_temperatures[self.opposite_lattice_indices] + 2*self.lattice_weights_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*self.bounce_back_temperature, 
                # -discrete_temperatures[self.opposite_lattice_indices] + 4*node_data.moments.density*self.lattice_weights_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*self.bounce_back_temperature,
                # -discrete_temperatures[self.opposite_lattice_indices] + 2*node_data.distributions.temp_new_population,
                discrete_temperatures
            )
            # print(self.lattice_weights_const.shape)
            # print(self.bounce_back_temperature.shape)
        node_data.distributions.vel_old_population = discrete_velocities
        node_data.distributions.temp_old_population = discrete_temperatures
        # print("After bounce back: ", node_data.distributions.temp_old_population.shape)
        # print(node_data.distributions.temp_old_population[10])
        return node_data
