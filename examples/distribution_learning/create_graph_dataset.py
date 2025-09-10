import os
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import itertools
import pickle

from mlbm.distribution_learning.datasets.collision_2d_graph_dataset import Collision2DGraphDataset

from torchlbm.node_data import NodeData, Distributions, Moments
from torchlbm.core.lattices.d2q9 import D2Q9
from torchlbm.core.macroscopic_quantities.collection import MacroscopicQuantityCalculationModule

import sys

def get_tensor_memory_in_MB(tensor):
    return tensor.element_size() * tensor.numel() / (1024 ** 2)

def data_object_memory_in_MB(data):
    total = 0
    for key, item in data.items():
        if isinstance(item, torch.Tensor):
            total += get_tensor_memory_in_MB(item)
    return total

def pyg_dataset_memory_usage(dataset, num_samples=None):
    if num_samples is None or num_samples > len(dataset):
        num_samples = len(dataset)

    total = 0
    for i in range(num_samples):
        data = dataset[i]
        total += data_object_memory_in_MB(data)

    avg_per_sample = total / num_samples
    estimated_total_MB = avg_per_sample * len(dataset)

    return estimated_total_MB



dataset_path = 'data/entropic_10k_gs2_exp3_5_u05'
num_samples = 1000000
graph_structure = 2
omega_min = 0.8
omega_max = 1.9999
omega_exp = 3.5
normalize = False
rho_min = 0.95
rho_max = 1.05
u_abs_max = 0.05
sigma_max = 1e-3
random_seed = None

if __name__ == "__main__":

    graph_dataset = Collision2DGraphDataset(num_samples=num_samples, 
                                            graph_structure=graph_structure, 
                                            random_seed=random_seed, 
                                            normalize=normalize, 
                                            rho_min=rho_min, 
                                            rho_max=rho_max, 
                                            u_abs_max=u_abs_max, 
                                            sigma_max=sigma_max, 
                                            omega_min=omega_min, 
                                            omega_max=omega_max,
                                            omega_exp=omega_exp)

    # Estimate the size of the dataset
    # estimated_size = estimate_item_size(graph_dataset)

    estimated_size = pyg_dataset_memory_usage(graph_dataset)
    print(f"Estimated size of the dataset: {estimated_size:.2f} MB")

    graph_dataset.save(dataset_path)

    loaded_graph_dataset = Collision2DGraphDataset.load(dataset_path)

    flag = True
    for i in range(len(graph_dataset.lattice_graphs)):
        if torch.equal(graph_dataset.lattice_graphs[i].x, loaded_graph_dataset.lattice_graphs[i].x) == False:
            flag = False
            break
        if torch.equal(graph_dataset.lattice_graphs[i].edge_index, loaded_graph_dataset.lattice_graphs[i].edge_index) == False:
            flag = False
            break
        if torch.equal(graph_dataset.lattice_graphs[i].y, loaded_graph_dataset.lattice_graphs[i].y) == False:
            flag = False
            break
        if torch.equal(graph_dataset.lattice_graphs[i].pos, loaded_graph_dataset.lattice_graphs[i].pos) == False:
            flag = False
            break
        if torch.equal(graph_dataset.lattice_graphs[i].omega, loaded_graph_dataset.lattice_graphs[i].omega) == False:
            flag = False
            break
        if torch.equal(graph_dataset.lattice_graphs[i].weights, loaded_graph_dataset.lattice_graphs[i].weights) == False:
            flag = False
            break
    print(flag) # Should print True
    print(graph_dataset.normalize == loaded_graph_dataset.normalize)  # Should print True
    if graph_dataset.normalize:
        print(torch.equal(graph_dataset.density, loaded_graph_dataset.density))  # Should print True

    loader = DataLoader(loaded_graph_dataset, batch_size=len(loaded_graph_dataset))
    lattice_graphs = next(iter(loader))

    relaxation_omega = lattice_graphs.omega.reshape(-1,9).transpose(1,0)
    relaxation_omega = relaxation_omega[0, :].numpy()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(relaxation_omega, bins=100, density=True)
    ax.set_title('Distribution of relaxation frequencies')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path, 'omega_distribution.pdf'))

    # print(lattice_graphs.x[27:36,1])

    pre_collision = lattice_graphs.x.reshape(-1,9).transpose(1,0).unsqueeze(-1).unsqueeze(-1)
    post_collision = lattice_graphs.y.reshape(-1,9).transpose(1,0).unsqueeze(-1).unsqueeze(-1)

    # print(pre_collision.shape)

    if loaded_graph_dataset.normalize:
        pre_collision *= loaded_graph_dataset.density
        post_collision *= loaded_graph_dataset.density

    lattice = D2Q9()
    macroscopic_module = MacroscopicQuantityCalculationModule(
            lattice_velocities=lattice.lattice_velocities(),
        )
    # node_data_pre = NodeData(
    #     distributions=Distributions(pre_collision, pre_collision),
    #     moments=Moments(torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor())
    # )

    density, velocity = macroscopic_module(pre_collision)

    # node_data_pre = macroscopic_module(node_data_pre)
    density = density.squeeze(-1).squeeze(-1)
    velocity = velocity.squeeze(-1).squeeze(-1)
    vel_mag = torch.norm(velocity, dim=0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

    # print(pre_collision.shape)

    ax1.boxplot(pre_collision.squeeze(-1).squeeze(-1).transpose(1,0).numpy(), tick_labels=[f'V{i}' for i in range(0, 9)])
    ax1.set_title('Box Plot of Discrete Velocities')
    # ax1.set_xlabel('Discrete Velocities')
    ax1.set_ylabel('Values')

    ax2.boxplot(density.numpy(), tick_labels=['Density'])
    ax2.set_title('Box Plot of Density')
    # ax2.set_xlabel('Density')
    ax2.set_ylabel('Values')

    ax3.boxplot(velocity[0].numpy(), positions = [0], tick_labels=['X-Velocity'])
    ax3.boxplot(velocity[1].numpy(), positions = [1], tick_labels=['Y-Velocity'])
    ax3.set_title('Box Plot of Velocity Magnitude')
    # ax3.set_xlabel('Velocity Magnitude')
    ax3.set_ylabel('Values')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path, 'stats.pdf'))
    # plt.show()

    pre_collision = pre_collision.squeeze(-1).squeeze(-1).numpy()
    post_collision = post_collision.squeeze(-1).squeeze(-1).numpy()

    # Plot histograms of pre-collision velocities in the same figure with 9 subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, j in itertools.product(range(3), range(3)):
        axs[i, j].hist(pre_collision[3*i+j], bins=100, density=True)
        axs[i, j].set_title(f'Population {3*i+j}')
        axs[i, j].set_xlabel('Value')
        axs[i, j].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path, 'pre_collision_histograms.pdf'))
    # plt.show()

    # Plot histograms of post-collision velocities in the same figure with 9 subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, j in itertools.product(range(3), range(3)):
        axs[i, j].hist(post_collision[3*i+j], bins=100, density=True)
        axs[i, j].set_title(f'Population {3*i+j}')
        axs[i, j].set_xlabel('Value')
        axs[i, j].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path, 'post_collision_histograms.pdf'))
    # plt.show()

    non_equilibrium = loaded_graph_dataset.non_equilibrium.squeeze(-1).squeeze(-1).numpy()
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, j in itertools.product(range(3), range(3)):
        axs[i, j].hist(non_equilibrium[3*i+j], bins=100, density=True)
        axs[i, j].set_title(f'Population {3*i+j}')
        axs[i, j].set_xlabel('Value')
        axs[i, j].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path, 'non_equilibrium_histograms.pdf'))
    # plt.show()

    single_loader = DataLoader(loaded_graph_dataset, batch_size=1)
    single_graph = next(iter(single_loader))

    # Plot this single graph using networkx
    vis = to_networkx(single_graph)
    node_labels = single_graph.x[:, 0].detach().numpy()
    pos = single_graph.pos.detach().numpy()


    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    nx.draw_networkx(vis, pos=pos, cmap=plt.get_cmap('Set3'), node_color=node_labels, node_size=70, linewidths=6)
    plt.savefig(os.path.join(dataset_path, 'graph_structure.pdf'))
    # plt.figure(1,figsize=(5,5)) 
    # nx.draw_networkx(vis, pos=pos, cmap=plt.get_cmap('Set3'),node_color = node_labels,node_size=70,linewidths=6)
    # plt.savefig(os.path.join(dataset_path, 'graph_structure.pdf'))
    # plt.show()


