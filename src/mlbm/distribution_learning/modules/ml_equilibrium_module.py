from typing import List

import torch.nn as nn
import torch
from mlbm.distribution_learning.networks.lightning_module import SymmetryNetwork
from torchlbm.node_data import NodeData
from mlbm.distribution_learning.utils import plot_stats


class MLEquilibriumModule(nn.Module):
    def __init__(self, model: SymmetryNetwork, lattice_velocities: List[List[float]], lattice_weights: List[float]) -> None:
        super(MLEquilibriumModule, self).__init__()
        self.model = model
        self.model.eval()
        self.lattice_velocities = torch.tensor(lattice_velocities).clone().detach()
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)
        self.lattice_weights = torch.tensor(lattice_weights).clone().detach()
        self.register_buffer("lattice_weights_const", self.lattice_weights)

    # @torch.no_grad()
    def forward(self, node_data: NodeData) -> NodeData:
        Q, X, Y, Z = node_data.distributions.old_population.shape
        flatten_old_population = node_data.distributions.old_population.reshape(Q, X * Y * Z).transpose(0, 1)

        with torch.no_grad():
            nn_output = self.model.run_inference(flatten_old_population, normalize_with_density=True)

        nn_output = nn_output.transpose(0, 1).reshape(Q, X, Y, Z)

        macroscopic_velocity = node_data.moments.velocity + node_data.moments.forcing_velocity
        projected_discrete_velocities = torch.einsum(
            "dQ,dNML->QNML",
            self.lattice_velocities_const,
            macroscopic_velocity,
        )
        macroscopic_velocity_magnitude = torch.linalg.norm(
            macroscopic_velocity,
            ord=2,
            dim=0,
        )
        reference_result = (
            node_data.moments.density.unsqueeze(0)
            * self.lattice_weights_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * (1 + 3 * projected_discrete_velocities + 9 / 2 * projected_discrete_velocities**2 - 3 / 2 * macroscopic_velocity_magnitude.unsqueeze(0) ** 2)
        )

        error = torch.abs((nn_output - reference_result) / reference_result)
        print(f"Max error: {torch.max(error)}")
        print(f"Min error: {torch.min(error)}")

        node_data.distributions.new_population = nn_output
        return node_data
