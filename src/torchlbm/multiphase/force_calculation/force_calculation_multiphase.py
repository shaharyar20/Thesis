from typing import List
import torch
import torch.nn as nn


class ForceCalculationMultiphaseModule(nn.Module):
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
        dimension: int,
        interaction_strength: float,
        n_discrete_velocities: int,
        is_gravity_active: bool,
        gravity_value: float,
    ) -> None:
        """Constructor of the module. The constructor is usually called from a factory function.

        Args:
            lattice_velocities (List[List[float]]): The velocity directions of the underlying velocity set.
            lattice_weights (List[float]): The weights of the underlying velocity set.
            access_indices (List[List[int]]): Important indices that are used to access correct array positions.
            dimension (int): The spatial dimension of the simulated problem.
            interaction_strength (float): The pre-factor multiplied with pseudopotential of the nearest neighbors.
        """
        super(ForceCalculationMultiphaseModule, self).__init__()
        self.lattice_velocities = torch.tensor(lattice_velocities).clone().detach().int()
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)
        self.lattice_weights = torch.tensor(lattice_weights)
        self.register_buffer("lattice_weights_const", self.lattice_weights)
        self.access_indices: List[List[int]] = access_indices
        self.G = interaction_strength
        self.dimension = dimension
        self.n_discrete_velocities = n_discrete_velocities
        self.is_gravity_active = is_gravity_active
        self.gravity_vector = torch.tensor([0.0, -gravity_value, 0.0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        self.register_buffer("gravity_vector_const", self.gravity_vector)

    def forward(self, pseudopotential: torch.Tensor, volume_force_field: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """The main functionality of the module as the forward pass of the module.
        It performs the calculation of the pseudopotential forces.

        Args:
            pseudopotential (torch.Tensor): The pseudopotential calculated based on prescribed EOS.
            node_data (NodeData): The node data that contains all storage intense field information.

        Returns:
            node_data (NodeData): The modified node data object with the updated volume force field.
        """
        shanchen_force = torch.zeros_like(volume_force_field)

        for k in range(self.n_discrete_velocities):
            shifted_i = self.lattice_velocities_const[0, k]
            shifted_j = self.lattice_velocities_const[1, k]
            shifted_k = self.lattice_velocities_const[2, k]
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
        volume_force_field += shanchen_force

        if self.is_gravity_active:
            volume_force_field += (density - density.mean()) * self.gravity_vector_const

        return volume_force_field
