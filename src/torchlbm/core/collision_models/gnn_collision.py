import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from torchlbm.node_data import NodeData
# from mlbm.distribution_learning.networks.cons_sym_networks import *
# from mlbm.distribution_learning.networks.networks import *

# import networkx as nx
# from torch_geometric.utils.convert import to_networkx
# import matplotlib.pyplot as plt


class GNNCollisionModule(nn.Module):
    """A TorchLBM module that performs the single relaxation time collision step of a Lattice-Boltzmann algorithm using a graph neural network.

    Args:
        nn (nn.Module): The collision operator is implemented as a PyTorch module. This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, model, model_path, num_graphs, lattice_velocities) -> None:
        """The initializer of the collision module. This only works for relaxation omega = 1. and lattice time step = 1.

        Args:
            relaxation_omega (float): The relaxation frequency.
        """
        super(GNNCollisionModule, self).__init__()
        checkpoint = torch.load(model_path)
        hparams = checkpoint["hparams"]
        state_dict = checkpoint["state_dict"]
        self.model = model(hparams, 1, 1, 1)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # self.model = torch.jit.script(self.model)
        print(num_graphs)

        graph_edge_index = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
        lattice_weights = torch.tensor([1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.]).reshape(8, 1)
        self.edge_attr = lattice_weights.repeat(2*num_graphs, 1)
        edge_index = graph_edge_index.repeat(1, num_graphs)
        batch_offsets = torch.arange(num_graphs).repeat_interleave(graph_edge_index.size(1)) * 9
        batch_offsets = torch.stack((batch_offsets, batch_offsets))
        self.edge_index = edge_index + batch_offsets
        self.edge_index = self.edge_index.cuda()
        self.edge_attr = self.edge_attr.cuda()
        lattice_velocities = torch.tensor(lattice_velocities)
        self.pos = lattice_velocities.repeat(1, num_graphs).transpose(1, 0)
        self.pos = self.pos[:, :2]
        print(self.pos.shape)
        print(self.pos[:, :9])
        self.pos = self.pos.cuda()

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
        # print(node_data.distributions.old_population[:,5,5,0])
        normalized_old_population = node_data.distributions.old_population # / node_data.moments.density
        flattened_old_population = normalized_old_population.reshape(Q, X * Y * Z).transpose(0, 1).reshape(-1, 1)
        # print(flattened_old_population.shape)
        # graph_data = Data(x=flattened_old_population, edge_index=self.edge_index, edge_attr=self.edge_attr)
        # vis = to_networkx(graph_data)
        # plt.figure(1, figsize=(8, 8))
        # nx.draw_networkx(vis, cmap=plt.get_cmap('Set3'),node_color = graph_data.x.cpu().numpy(),node_size=70,edge_cmap=plt.get_cmap('Set3'),edge_color=graph_data.edge_attr.reshape(-1).cpu().numpy())
        # plt.savefig("graph.png")
        # print(graph_data.num_edge_features)
        # print(graph_data.num_nodes)
        # print(graph_data.num_edges)
        # print(graph_data.num_node_features)
        # print(graph_data.is_directed())
        # batch = Batch.from_data_list([graph_data])
        discrete_velocities_post_collision = self.model(flattened_old_population, self.edge_index, self.pos, self.edge_attr).reshape(-1, Q).transpose(0, 1).reshape(Q, X, Y, Z)#* node_data.moments.density
        # print(discrete_velocities_post_collision.shape)
        # print(discrete_velocities_post_collision[:, 100])
        # print(discrete_velocities_post_collision[:, 5, 5, 0])
        # discrete_velocities_post_collision = self.model(flattened_old_population).transpose(0,1 ).reshape(Q, X, Y, Z)
        # discrete_velocities_post_collision = (1.0 - self.lattice_time_step * self.relaxation_omega) * node_data.distributions.old_population + self.lattice_time_step * self.relaxation_omega * node_data.distributions.new_population + self.lattice_time_step * node_data.moments.collision_source_term
        return discrete_velocities_post_collision
