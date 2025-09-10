from typing import List, Tuple
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.types as Tensor
from torchlbm.node_data import NodeData
from torchlbm.core.collision_models.entropic_mrt_decompositions import decompose_s_d2q9_nat, decompose_s_d3q27_nat


class EntropicMRTCollisionModule(nn.Module):
    """A TorchLBM module that performs the multi relaxation time collision step of a Lattice-Boltzmann algorithm
    Args:
        nn (nn.Module): The collision operator is implemented as a PyTorch module. This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, lattice_velocities: List[List[float]], lattice_weights: List[List[float]], model: int) -> None:
        """The initializer of the two relaxation time collosion module.
        Args:
            relaxation_omega (float): The relaxation parameter.
            my_population_to_momentum_transform (List[List[float]]): The matrix to transform to the momentum space.
                                                                     It is a property of the underlying velocity set.
            my_momentum_to_population_transform (List[List[float]]): The matrix to transform from the momentum space.
                                                                     It is a property of the underlying velocity set.
        """
        super(EntropicMRTCollisionModule, self).__init__()
        # self.relaxation_omega = relaxation_omega
        # self.relaxation = torch.tensor([relaxation_omega])
        # self.register_buffer("relaxation_const", self.relaxation)
        self.lattice_velocities = torch.tensor(lattice_velocities).clone().detach()
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)
        self.model = model
        self.lattice_weights = torch.tensor(lattice_weights).clone().detach()
        self.register_buffer("lattice_weights_const", self.lattice_weights)

    def forward(self, old_population: torch.Tensor, new_population: torch.Tensor, relaxation_omega: torch.Tensor, collision_source_term: torch.Tensor) -> torch.Tensor:
        """The forward pass of the collision module. Gets as input the discretized velocity distribution of the start of the timestept,
        and the equilibrium distribution calculated based on it. It returns the post-collision distribution.
        Args:
            discrete_velocities (torch.Tensor): The discretized velocity distribution at the beginning of the timestep. It is a (L x Nx x Ny x Nz) tensor,
            where L denotes the number of lattice velocities, and Nx, Ny, and Nz the number of cells in x-, y-, and z-direction, respectively.
            equilibrium_discrete_velocities (torch.Tensor): The equilibirium distribution towards which relaxation is performed.
                                                            It is a (L x Nx x Ny x Nz) tensor, where L denotes the number of lattice velocities,
                                                            and Nx, Ny, and Nz the number of cells in x-, y-, and z-direction, respectively.
        Returns:
            torch.Tensor: The discretized velocity distribution after collision.
        """
        beta = 0.5 * relaxation_omega
        invbeta = 1.0 / beta

        f_i = old_population
        f_eq_i = new_population
        f_neq_i = f_i - f_eq_i

        # Alternative implementation: KBC only stabilizes the collision for voxels, where entropy decreases for linear BGK
        # Karlin 2014, eq. (3)
        if self.model == 2:
            delta_s = decompose_s_d2q9_nat(f_neq_i, self.lattice_velocities_const)
        elif self.model == 3:
            delta_s = decompose_s_d3q27_nat(f_neq_i, self.lattice_velocities_const)
        else:
            raise ValueError("Unsupported dimension: {}".format(self.model))

        delta_h = f_i - f_eq_i - delta_s
        scalar_s = torch.sum(delta_s * delta_h / f_eq_i, dim=0)
        scalar_h = torch.clamp(torch.sum(delta_h * delta_h / f_eq_i, dim=0), min=1e-10)  # avoid division by zero

        # Karlin 2014, eq. (9)
        gamma = invbeta - (2.0 - invbeta) * (scalar_s / (scalar_h))
        # node_data.distributions.population_like = delta_s
        # node_data.moments.gamma = gamma

        # Karlin 2014, eq. (10)
        f_post_i = f_i - beta * (2.0 * delta_s + gamma * delta_h)
        f_bgk_i = (1.0 - relaxation_omega) * f_i + relaxation_omega * f_eq_i

        mod = False
        if mod:
            H_tol = 1.0e-7

            H_i = -torch.sum(f_i * torch.log(f_i / self.lattice_weights_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), dim=0)
            H_bgk_i = -torch.sum(f_bgk_i * torch.log(f_bgk_i / self.lattice_weights_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), dim=0)

            f_post_i = torch.where(H_bgk_i > H_i * (1.0 - H_tol), f_post_i, f_bgk_i)
            # modified = torch.where(H_bgk_i > H_i * (1.0 - H_tol), 1, 0)
            # node_data.moments.density_like = modified

        return f_post_i

    def entropic_scalar(self, X: torch.Tensor, Y: torch.Tensor, f_eq_i: torch.Tensor) -> torch.Tensor:
        """Compute the entropic scalar product of x and y to approximate gamma in KBC.
        Args:
            X (torch.Tensor): The first input tensor.
            Y (torch.Tensor): The second input tensor.
            moment_equilibrium_populations (torch.Tensor): The equilibrium populations.
        Returns
            torch.Tensor of the shape X, Y, Z and sums over the first dimension (lattice dimension).
            Entropic scalar product of x, y, and feq.
        """
        # Karlin 2014, eq. (8)
        scalar = torch.sum(X * Y / f_eq_i, dim=0)
        return scalar

    def test_split_population(
        self,
        delta_h: torch.Tensor,
        delta_s: torch.Tensor,
        f_i: torch.Tensor,
        f_eq_i: torch.Tensor,
        beta: float,
        gamma: torch.Tensor,
        rho: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        test_split = torch.sum(delta_h * torch.log(1 + ((1 - beta * gamma) * delta_h - (2 * beta - 1) * delta_s) / f_eq_i), dim=0)

        return test_split
