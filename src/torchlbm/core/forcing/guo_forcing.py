import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import List

from torchlbm.node_data import NodeData


class GuoForcingModule(nn.Module):
    """PyTorch module that implements the calculation of a volume force based on the Shan-Chem scheme.
    It is implemented as a PyTorch module to allow composing algorithms based on consecutively applied modules.

    Args:
        nn (nn.Module): PyTorch module which is used ase base class.
    """

    def __init__(
        self,
        force_vector: torch.tensor,
        lattice_velocities: List[List[float]],
        lattice_weights: List[float],
    ) -> None:
        """Initializer of the module.

        Args:
            forcing_active (bool): Indicated whether the forcing scheme is active.
            tau (float): The relaxation time of the LBM scheme.
            force_vector (torch.tensor): The constant force vector applied to the whole domain. It has the dimension 3, where three is the number
                                         of spatial dimensions.
        """
        super(GuoForcingModule, self).__init__()
        self.force_vector = force_vector
        self.register_buffer("force_vector_const", self.force_vector)
        self.lattice_velocities = torch.tensor(lattice_velocities)
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)
        self.lattice_weights = torch.tensor(lattice_weights)
        self.register_buffer("lattice_weights_const", self.lattice_weights)

    def forward(self, volume_force_field: torch.Tensor, density: torch.Tensor, velocity: torch.Tensor, relaxation_omega: torch.Tensor) -> List[torch.Tensor]:
        """The forward passt calculation the equilibrium macroscopic velocities for the volume force.

        Args:
            node_data (NodeData): The node_data object is a pure data container that contains the storage heavy macroscopic and microscopix field data.

        Returns:
            torch.Tensor: The equilibrium velocitiy that was calculated based on the force.
        """
        volume_force_field = volume_force_field + self.force_vector_const
        equilibrium_macroscopic_velocities = volume_force_field / (density * relaxation_omega)

        equilibrium_macroscopic_velocities = 0.5 * volume_force_field / density
        macroscopic_velocity = velocity + equilibrium_macroscopic_velocities

        projected_discrete_velocities = torch.einsum(
            "dQ,dNML->QNML",
            self.lattice_velocities_const,
            macroscopic_velocity,
        )
        unprojected_discrete_velocities = 9 * projected_discrete_velocities * self.lattice_velocities_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        difference_velocities = 3 * (self.lattice_velocities_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) - macroscopic_velocity.unsqueeze(1))

        weighted_discrete_velocities = self.lattice_weights_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (
                difference_velocities + unprojected_discrete_velocities)

        collision_source_term = torch.einsum(
            "dQNML,dNML->QNML",
            weighted_discrete_velocities,
            volume_force_field
        )
        return equilibrium_macroscopic_velocities, volume_force_field, collision_source_term
