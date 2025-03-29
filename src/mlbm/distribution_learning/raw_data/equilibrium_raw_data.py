import numpy as np
import torch
from scipy.special import lambertw
import random
from torch.utils.data import Dataset
import sys

from torchlbm.core.lattices.d2q9 import D2Q9
from torchlbm.node_data import NodeData, Distributions, Moments
from torchlbm.core.equilibrium.equilibrium import EquilibriumCalculationModule
from torchlbm.core.collision_models.srt_collision import SRTCollisionModule
from torchlbm.core.macroscopic_quantities.collection import MacroscopicQuantityCalculationModule
import torchlbm.standalone_operations.file_operations as file_o
from torchlbm.io_tools.statistics_output_writer import get_boxplot_figure

from typing import NamedTuple
import os
from pathlib import Path


def set_determinism(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(random_seed)
    np.random.default_rng(random_seed)
    random.seed(random_seed)


def remove_determinism():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def compute_rho_u(X, Y, Z, rho_min=0.95, rho_max=1.05, u_abs_min=0, u_abs_max=0.01):
    # y = np.random.rand(100000) * (rho_max - rho_min) + rho_min
    # rho = np.exp(lambertw(y).real)

    # rho = torch.rand((X, Y, Z)) * (rho_max - rho_min) + rho_min
    rho = torch.ones((X, Y, Z))
    u = torch.rand((3, X, Y, Z)) * (u_abs_max - u_abs_min) + u_abs_min
    theta = torch.rand((X, Y, Z)) * 2 * np.pi

    u[0] = u[0] * np.cos(theta)
    u[1] = u[1] * np.sin(theta)

    return rho, u


def compute_f_neq(discrete_velocities, X, Y, Z, sigma_min=0, sigma_max=1.5e-4):
    _, Q = discrete_velocities.shape

    # sigma = torch.rand((X, Y, Z)) * (sigma_max - sigma_min) + sigma_min
    sigma = torch.abs(torch.normal(torch.zeros((X, Y, Z)), sigma_max / 1.0))
    sigma = sigma.repeat(Q, 1, 1, 1)
    f_neq = torch.normal(torch.zeros_like(sigma), sigma)

    rho = f_neq.sum(dim=0)
    momentum = torch.einsum("dQ,QNML->dNML", discrete_velocities, f_neq)

    for i in range(Q):
        vel_times_mom = torch.einsum("d,dNML->NML", discrete_velocities[:, i], momentum)
        f_neq[i] = f_neq[i] - (1 / 9) * rho - (1 / 6) * vel_times_mom

    return f_neq


class RawDataSettings(NamedTuple):
    random_seed: int
    X: int
    Y: int
    Z: int
    relaxation_omega: float
    u_abs_max: float


def generate_and_write_raw_data(path: Path, settings: RawDataSettings):

    set_determinism(settings.random_seed)

    lattice = D2Q9()

    macroscopic_module = MacroscopicQuantityCalculationModule(
        lattice_velocities=lattice.lattice_velocities(),
    )

    equilibrium_module = EquilibriumCalculationModule(lattice.lattice_velocities(), lattice.lattice_weights())

    collision_module = SRTCollisionModule(settings.relaxation_omega)
    discrete_velocities = torch.tensor(lattice.lattice_velocities())

    rho, u = compute_rho_u(settings.X, settings.Y, settings.Z, u_abs_max=settings.u_abs_max)

    sample = NodeData(distributions=Distributions(torch.Tensor(), torch.Tensor()), moments=Moments(rho, u, torch.zeros_like(u), torch.zeros_like(u)))

    # Compute equilibrium state for pre state
    sample = equilibrium_module(sample)
    sample.distributions.old_population = sample.distributions.new_population.clone()

    # feq + fneq
    fneq = compute_f_neq(discrete_velocities, settings.X, settings.Y, settings.Z)
    sample.distributions.old_population += fneq

    # Compute Macroscopic Velocities
    sample = macroscopic_module(sample)

    # Compute equilibrium state for collision operator
    sample: NodeData = equilibrium_module(sample)

    # Perform collision
    # sample.distributions.new_population = collision_module(sample)
    n_samples = settings.X * settings.Y * settings.Z
    pre_equilibrium = sample.distributions.old_population.reshape(9, n_samples).transpose(0, 1)
    post_equilibrium = sample.distributions.new_population.reshape(9, n_samples).transpose(0, 1)
    density = sample.moments.density.flatten()
    velocity = sample.moments.velocity.reshape(3, n_samples).transpose(0, 1)

    folder_created = file_o.create_folder(path)
    print(f"Folder succesfully created: {folder_created}")

    print(f"Max neq: {torch.max(pre_equilibrium - post_equilibrium)}")
    print(f"Mean neq: {torch.mean(torch.abs(pre_equilibrium - post_equilibrium), dim=(0))}")
    print(f"Std neq: {torch.std(torch.abs(pre_equilibrium - post_equilibrium), dim=(0))}")

    import matplotlib.pyplot as plt

    fig = get_boxplot_figure(pre_equilibrium.transpose(1, 0))
    fig.savefig(os.path.join(path, "visualization_pre_collision.pdf"))
    plt.close(fig)
    fig = get_boxplot_figure(post_equilibrium.transpose(1, 0))
    fig.savefig(os.path.join(path, "visualization_post_collision.pdf"))
    plt.close(fig)

    torch.save(pre_equilibrium, os.path.join(path, "pre_equilibrium.pt"))
    torch.save(post_equilibrium, os.path.join(path, "post_equilibrium.pt"))
    torch.save(density, os.path.join(path, "density.pt"))
    torch.save(velocity, os.path.join(path, "velocity.pt"))
    torch.save(pre_equilibrium - post_equilibrium, os.path.join(path, "non_equilibrium_distribution.pt"))
    torch.save(torch.tensor(lattice.lattice_weights()), os.path.join(path, "lattice_weights.pt"))
    torch.save(torch.tensor(lattice.lattice_velocities()), os.path.join(path, "lattice_velocities.pt"))
    torch.save(torch.mean(pre_equilibrium, dim=0), os.path.join(path, "normalization_mean.pt"))
    torch.save(torch.std(pre_equilibrium, dim=0), os.path.join(path, "normalization_std.pt"))
    torch.save(torch.min(pre_equilibrium, dim=0).values, os.path.join(path, "normalization_min.pt"))
    torch.save(torch.max(pre_equilibrium, dim=0).values, os.path.join(path, "normalization_max.pt"))

    remove_determinism()
