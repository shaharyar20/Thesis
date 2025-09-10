import torch
import torch.nn as nn

from torchlbm.node_data import NodeData


class GNNCollisionModule(nn.Module):
    """A TorchLBM module that performs the single relaxation time collision step of a Lattice-Boltzmann algorithm using a graph neural network.

    Args:
        nn (nn.Module): The collision operator is implemented as a PyTorch module. This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, model, model_path) -> None:
        """The initializer of the collision module. This only works for relaxation omega = 1. and lattice time step = 1.

        Args:
            relaxation_omega (float): The relaxation frequency.
        """
        super(GNNCollisionModule, self).__init__()
        self.model = model.load_from_checkpoint(model_path, input_dim=1, output_dim=1, edge_dim=0)
        self.model = self.model.eval()
        self.model = torch.compile(self.model)
        

    # @torch.no_grad()
    def forward(self, old_population: torch.Tensor, new_population: torch.Tensor, relaxation_omega: torch.Tensor, collision_source_term: torch.Tensor) -> torch.Tensor:
        """The forward pass of the collision modules. Gets as input the discretized velocity distribution of the start of the timestep,
        and the equilibrium distribution calculated based on it. It returns the post-collision distribution.

        Args:
            discrete_velocities (torch.Tensor): The discretized velocity distribution at the beginning of the timestep. It is a (L x Nx x Ny x Nz) tensor,
            where L denotes the number of lattice velocities, and Nx, Ny, and Nz the number of cells in x-, y-, and z-direction, respectively.
            equilibrium_discrete_velocities (torch.Tensor): The equilibrium distribution towards which relaxation is performed.
                                                            It is a (L x Nx x Ny x Nz) tensor, where L denotes the number of lattice velocities,
                                                            and Nx, Ny, and Nz the number of cells in x-, y-, and z-direction, respectively.
            source_term (torch.Tensor): The source term resulting from forces in the domain. It is a (L x Nx x Ny x Nz) tensor,
            where L denotes the number of lattice velocities, and Nx, Ny, and Nz the number of cells in x-, y-, and z-direction, respectively.

        Returns:
            torch.Tensor: The discretized velocity distribution after collision.
        """
        # print(relaxation_omega.unsqueeze(-1).shape)
        # print(a)
        Q, X, Y, Z = old_population.shape
        old_population = old_population.reshape(Q, -1).transpose(0, 1).unsqueeze(-1)
        old_population = self.model(old_population, relaxation_omega.unsqueeze(-1)).squeeze(-1).transpose(0, 1).reshape(Q, X, Y, Z)

        return old_population
