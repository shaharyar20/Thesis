from typing import List
import torch
import torch.nn as nn

from torchlbm.node_data import NodeData


class ShanChenPseudopotentialMultiphaseModule(nn.Module):
    """PyTorch module that implements the Shan-Chen pseudopotential method for multiphase simulations.
    It is implemented as a PyTorch module to allow composing algorithms based on consecutively applied modules.

    Args:
        nn (nn.Module): PyTorch module which is used ase base class.
    """

    def __init__(
        self,
        lattice_velocities: List[List[float]],
        lattice_weights: List[float],
        access_indices: List[List[int]],
        tau: float,
        dimension: int,
    ) -> None:
        """Constructor of the module. The constructor is usually called from a factory function.

        Args:
            lattice_velocities (List[List[float]]): The velocity directions of the underlying velocity set.
            lattice_weights (List[float]): The weights of the underlying velocity set.
            access_indices (List[List[int]]): Important indices that are used to access correct array positions.
            tau (float): The relaxation time.
            dimension (int): The spatial dimension of the simulated problem.
        """
        super(ShanChenPseudopotentialMultiphaseModule, self).__init__()
        # self.number_of_discrete_velocities = number_of_discrete_velocities
        self.ref_density = 1.0
        self.lattice_velocities = torch.tensor(lattice_velocities)
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)
        self.lattice_weights = torch.tensor(lattice_weights)
        self.register_buffer("lattice_weights_const", self.lattice_weights)
        self.access_indices: List[List[int]] = access_indices
        self.G = -4.7
        self.tau = tau
        self.dimension = dimension

    def forward(self, node_data: NodeData) -> torch.Tensor:
        """The main functionality of the module as the forward pass of the module.
        It performs the calculation of the pseudopotential forces.

        Args:
            node_data (NodeData): The node data that contains all storage intense field information.

        Returns:
            torch.Tensor: The calculated forces.
        """
        pseudopotential = self.ref_density * (1.0 - torch.exp(-node_data.moments.density / self.ref_density))
        shanchen_force = torch.zeros_like(node_data.moments.velocity)

        for k in range(9):
            shifted_i = int(self.lattice_velocities_const[0, k])
            shifted_j = int(self.lattice_velocities_const[1, k])
            shifted_k = int(self.lattice_velocities_const[2, k])
            pseudopotential_shift = pseudopotential[
                self.access_indices[0][1] + shifted_i : self.access_indices[0][2] + shifted_i,
                self.access_indices[1][1] + shifted_j if self.dimension != 1 else 0 : self.access_indices[1][2] + shifted_j if self.dimension != 1 else 1,
                self.access_indices[2][1] + shifted_k if self.dimension == 3 else 0 : self.access_indices[2][2] + shifted_k if self.dimension == 3 else 1,
            ].unsqueeze(0)

            shanchen_force[
                :,
                self.access_indices[0][1] : self.access_indices[0][2],
                self.access_indices[1][1] if self.dimension != 1 else 0 : self.access_indices[1][2] if self.dimension != 1 else 1,
                self.access_indices[2][1] if self.dimension == 3 else 0 : self.access_indices[2][2] if self.dimension == 3 else 1,
            ] += (
                self.lattice_weights_const[k] * self.lattice_velocities_const[:, k].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * pseudopotential_shift
            )

        shanchen_force *= -pseudopotential * self.G

        node_data.moments.volume_force_field += shanchen_force
        equilibrium_macroscopic_velocities = self.tau * (node_data.moments.volume_force_field) / node_data.moments.density

        return equilibrium_macroscopic_velocities
