import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import List

from torchlbm.node_data import NodeData


class ShanChenForcingModule(nn.Module):
    """PyTorch module that implements the calculation of a volume force based on the Shan-Chem scheme.
    It is implemented as a PyTorch module to allow composing algorithms based on consecutively applied modules.

    Args:
        nn (nn.Module): PyTorch module which is used ase base class.
    """

    def __init__(
        self,
        forcing_active: bool,
        tau: float,
        force_vector: torch.tensor,
    ) -> None:
        """Initializer of the module.

        Args:
            forcing_active (bool): Indicated whether the forcing scheme is active.
            tau (float): The relaxation time of the LBM scheme.
            force_vector (torch.tensor): The constant force vector applied to the whole domain. It has the dimension 3, where three is the number
                                         of spatial dimensions.
        """
        super(ShanChenForcingModule, self).__init__()
        self.forcing_active = forcing_active
        self.tau = tau
        self.force_vector = force_vector
        self.register_buffer("force_vector_const", self.force_vector)

    def forward(self, volume_force_field: torch.Tensor, density: torch.Tensor) -> List[torch.Tensor]:
        """The forward passt calculation the equilibrium macroscopic velocities for the volume force.

        Args:
            node_data (NodeData): The node_data object is a pure data container that contains the storage heavy macroscopic and microscopix field data.

        Returns:
            torch.Tensor: The equilibrium velocitiy that was calculated based on the force.
        """
        volume_force_field = volume_force_field + self.force_vector_const
        equilibrium_macroscopic_velocities = self.tau * volume_force_field / density
        return equilibrium_macroscopic_velocities, volume_force_field
