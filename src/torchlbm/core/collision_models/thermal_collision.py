import torch
import torch.nn as nn
from dataclasses import dataclass

# from torchlbm.node_data import NodeData
from torchlbm.thermal_node_data import ThermalNodeData


class ThermalCollisionModule(nn.Module):
    """A Torch LBM module that performs the single relaxation time collision step of a Lattice-Boltzmann algorithm

    Args:
        nn (nn.Module): The collision operator is implemented as a PyTorch module. This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, relaxation_omega_vel: float, relaxation_omega_temp: float) -> None:
        """The initializer of the collision module.

        Args:
            relaxation_omega (float): The relaxation frequency.
        """
        super(ThermalCollisionModule, self).__init__()
        self.relaxation_omega_vel = relaxation_omega_vel
        self.relaxation_omega_temp = relaxation_omega_temp

    def forward(self, node_data: ThermalNodeData) -> ThermalNodeData:
        """The forward pass of the collision modules. Gets as input the discretized velocity distribution of the start of the timestept,
        and the equilibrium distribution calculated based on it. It returns the post-collision distribution.

        Args:
            node_data (NodeData): The node_data object is a pure data container that contains the storage heavy macroscopic and microscopix field data.

        Returns:
            torch.Tensor: The discretized velocity distribution after collision.
        """

        node_data.distributions.vel_old_population = torch.where(
            node_data.bounce_back_mask > 0,
            node_data.distributions.vel_old_population,
            (1.0 - self.relaxation_omega_vel) * node_data.distributions.vel_old_population + self.relaxation_omega_vel * node_data.distributions.vel_new_population
        )

        node_data.distributions.temp_old_population = torch.where(
            node_data.bounce_back_mask > 0,
            node_data.distributions.temp_old_population,
            (1.0 - self.relaxation_omega_temp) * node_data.distributions.temp_old_population + self.relaxation_omega_temp * node_data.distributions.temp_new_population
        )

        # discrete_velocities_post_collision = (
        #     1.0 - self.relaxation_omega_vel
        # ) * node_data.distributions.vel_old_population + self.relaxation_omega_vel * node_data.distributions.vel_new_population
        # discrete_temperatures_post_collision = (
        #     1.0 - self.relaxation_omega_temp
        # ) * node_data.distributions.temp_old_population + self.relaxation_omega_temp * node_data.distributions.temp_new_population
        return node_data
