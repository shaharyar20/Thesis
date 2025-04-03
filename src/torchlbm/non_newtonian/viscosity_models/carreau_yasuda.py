import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List

from torchlbm.node_data import NodeData
from torchlbm.unit_converter import UnitConverter


class CarreauYasudaModule(nn.Module):
    """A TorchLBM module that performs the Carreau-Yasuda Model step of a Lattice-Boltzmann algorithm for non-newtonian fluids

    Args:
        nn (nn.Module): The Carreau-Yasuda Model is implemented as a PyTorch module. This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(
        self,
        unit_converter: UnitConverter,
        lattice_velocities,
        number_of_discrete_velocities,
        viscosity_inf,
        viscosity_0,
        lam,
        n,
        a,
    ) -> None:
        """The initializer of the collision module.

        Args:
            relaxation_omega (float): The relaxation frequency.
        """
        super(CarreauYasudaModule, self).__init__()
        self.unit_converter: UnitConverter = unit_converter
        self.cs = unit_converter.cs
        self.viscosity_inf = viscosity_inf
        self.viscosity_0 = viscosity_0
        self.lam = lam
        self.n = n
        self.a = a
        self.exp = (n - 1.0) / a
        lattice_velocities = torch.tensor(lattice_velocities)
        self.kron = torch.empty([number_of_discrete_velocities, 3, 3])
        for i in range(number_of_discrete_velocities):
            self.kron[i] = torch.kron(lattice_velocities[:, i].unsqueeze(1), lattice_velocities[:, i].unsqueeze(0))

        self.register_buffer("kron_const", self.kron)

    def forward(self, node_data: NodeData) -> torch.Tensor:
        """The forward pass of the Carreau-Yasuda Model modules. Gets as input the discretized velocity distribution of the start of the timestept,
        and the equilibrium distribution calculated based on it. It returns the post-collision distribution.

        Args:
            node_data (NodeData): The node_data object is a pure data container that contains the storage heavy macroscopic and microscopix field data.

        Returns:
            torch.Tensor: The discretized velocity distribution after collision.
        """
        neq = node_data.distributions.old_population - node_data.distributions.new_population

        test = torch.einsum(
            "QNML,Qde->NMLde",
            neq,
            self.kron_const,
        )
        blub = torch.norm(test, dim=[-2, -1])
        shear_rate = node_data.relaxation_omega * blub / (self.cs**2 * node_data.moments.density)
        shear_rate = shear_rate / self.unit_converter.conversion_factor_time
        # print(shear_rate.shape)

        viscosity = self.viscosity_inf + (self.viscosity_0 - self.viscosity_inf) * (1 + (self.lam * shear_rate) ** self.a) ** self.exp

        relaxation_time = self.unit_converter.convert_kinematic_viscosity_to_relaxation_time_lattice_units(viscosity)
        # print(f"Max relaxation time: {torch.max(relaxation_time)}")
        # print(f"Min relaxation time: {torch.min(relaxation_time)}")
        new_relaxation_omega = 1.0 / relaxation_time

        # new_relaxation_omega = (2 * self.cs**2 * node_data.moments.density) / (2 * viscosity + self.cs**2 * node_data.moments.density)

        return new_relaxation_omega
