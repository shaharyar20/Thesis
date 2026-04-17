import torch
import torch.nn as nn

from torchlbm.node_data import NodeData
# from mlbm.distribution_learning.networks.cons_sym_networks import *
# from mlbm.distribution_learning.networks.networks import *


class NNCollisionModule(nn.Module):
    """A TorchLBM module that performs the single relaxation time collision step of a Lattice-Boltzmann algorithm using a neural network.

    Args:
        nn (nn.Module): The collision operator is implemented as a PyTorch module. This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, model, model_path) -> None:
        """The initializer of the collision module. This only works for relaxation omega = 1. and lattice time step = 1.

        Args:
            relaxation_omega (float): The relaxation frequency.
        """
        super(NNCollisionModule, self).__init__()
        checkpoint = torch.load(model_path)
        hparams = checkpoint["hparams"]
        state_dict = checkpoint["state_dict"]
        self.model = model(hparams=hparams)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # self.model = torch.jit.script(self.model)

    @torch.no_grad()
    def forward(self, node_data: NodeData) -> torch.Tensor:
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
        Q, X, Y, Z = node_data.distributions.old_population.shape
        flattened_old_population = node_data.distributions.old_population.reshape(Q, X * Y * Z).transpose(0, 1)
        discrete_velocities_post_collision = self.model(flattened_old_population).transpose(0, 1).reshape(Q, X, Y, Z)
        # discrete_velocities_post_collision = (1.0 - self.lattice_time_step * self.relaxation_omega) * node_data.distributions.old_population + self.lattice_time_step * self.relaxation_omega * node_data.distributions.new_population + self.lattice_time_step * node_data.moments.collision_source_term
        return discrete_velocities_post_collision
