import numpy as np
import torch
import random
from torch.utils.data import Dataset
import math
import pickle
import os

from torchlbm.core.lattices.d2q9 import D2Q9
from torchlbm.node_data import NodeData, Distributions, Moments
from torchlbm.core.collision_models.linear_bgk import EquilibriumCalculationModule
from torchlbm.core.collision_models.collision import CollisionModule
from torchlbm.core.macroscopic_quantities.collection import MacroscopicQuantityCalculationModule


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


def compute_rho_u(num_samples, rho_min=0.95, rho_max=1.05, u_abs_min=1e-15, u_abs_max=0.03):
    rho = torch.rand(num_samples, 1, 1) * (rho_max - rho_min) + rho_min
    u0 = torch.rand(num_samples, 1, 1) * (u_abs_max - u_abs_min) + u_abs_min
    theta = torch.rand(num_samples, 1, 1) * 2 * np.pi

    u_x = u0 * torch.cos(theta)
    u_y = u0 * torch.sin(theta)
    u_z = torch.zeros_like(u_x)
    u = torch.stack((u_x, u_y, u_z))

    return rho, u


def compute_f_neq(discrete_velocities, num_samples, sigma_min=1e-15, sigma_max=5e-4):
    _, Q = discrete_velocities.shape

    sigma = torch.rand(num_samples, 1, 1) * (sigma_max - sigma_min) + sigma_min
    sigma = sigma.repeat(Q, 1, 1, 1)
    f_neq = torch.normal(torch.zeros_like(sigma), sigma)

    rho = f_neq.sum(dim=0)
    momentum = torch.einsum("dQ,QNML->dNML", discrete_velocities, f_neq)

    for i in range(Q):
        vel_times_mom = torch.einsum("d,dNML->NML", discrete_velocities[:, i], momentum)
        f_neq[i] = f_neq[i] - (1 / 9) * rho - (1 / 6) * vel_times_mom

    return f_neq


class SyntheticCollisionDataset(Dataset):
    def __init__(self, num_samples=1000, relaxation_omega=1.0, random_seed=None, normalize=True, rho_min=0.95, rho_max=1.05, u_abs_min=1e-15, u_abs_max=0.03, data=None) -> None:
        super().__init__()

        if data is not None:
            father, self.pre_collision, self.post_collision = data
            self.normalize = father.normalize
            if father.normalize:
                self.density = father.density
            self.num_samples = len(self.pre_collision)
            return

        self.num_samples = num_samples
        self.relaxation_omega = relaxation_omega
        self.random_seed = random_seed
        self.normalize = normalize
        self.u_abs_max = u_abs_max

        if self.random_seed is not None:
            set_determinism(self.random_seed)

        lattice = D2Q9()

        macroscopic_module = MacroscopicQuantityCalculationModule(
            lattice_velocities=lattice.lattice_velocities(),
        )

        equilibrium_module = EquilibriumCalculationModule(
            lattice_velocities=lattice.lattice_velocities(),
            lattice_weights=lattice.lattice_weights(),
        )

        collision_module = CollisionModule(
            relaxation_omega=self.relaxation_omega,
        )

        discrete_velocities = torch.tensor(lattice.lattice_velocities())
        
        rho, u = compute_rho_u(num_samples=self.num_samples, rho_min=rho_min, rho_max=rho_max, u_abs_min=u_abs_min, u_abs_max=self.u_abs_max)

        sample = NodeData(distributions=Distributions(torch.Tensor(), torch.Tensor()), moments=Moments(rho, u, torch.zeros_like(u), torch.zeros_like(u)))

        sample = equilibrium_module(sample)
        sample.distributions.old_population = sample.distributions.new_population.clone()

        f_neq = compute_f_neq(discrete_velocities=discrete_velocities, num_samples=self.num_samples)
        sample.distributions.old_population += f_neq

        sample = macroscopic_module(sample)

        sample = equilibrium_module(sample)

        sample.distributions.new_population = collision_module(sample)

        self.pre_collision = sample.distributions.old_population
        self.post_collision = sample.distributions.new_population
        self.density = sample.moments.density

        if normalize:
            self.pre_collision /= sample.moments.density
            self.post_collision /= sample.moments.density

        self.pre_collision = sample.distributions.old_population.reshape(9, self.num_samples).transpose(0, 1)
        self.post_collision = sample.distributions.new_population.reshape(9, self.num_samples).transpose(0, 1)

        # if normalize:
        #     # means = torch.mean(self.pre_collision, dim=0)
        #     # sigmas = torch.std(self.pre_collision, dim=0)
        #     # self.min_for_norm = torch.min(self.pre_collision, dim=0).values
        #     # self.max_for_norm = torch.max(self.pre_collision, dim=0).values
        #     self.min_for_norm = torch.min(self.pre_collision)
        #     self.max_for_norm = torch.max(self.post_collision)

        #     self.pre_collision = (self.pre_collision - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)
        #     self.post_collision = (self.post_collision - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)

        if self.random_seed is not None:
            remove_determinism()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index < 0 or index >= self.num_samples:
            raise IndexError("Incorrect index")

        return self.pre_collision[index], self.post_collision[index]

    def splits(self, split={"train": 0.8, "val": 0.1, "test": 0.1}):
        n_train = int(self.num_samples * split["train"])
        n_val = int(self.num_samples * split["val"])

        train_data = self.pre_collision[0:n_train], self.post_collision[0:n_train]
        val_data = self.pre_collision[n_train : (n_train + n_val)], self.post_collision[n_train : (n_train + n_val)]
        test_data = self.pre_collision[(n_train + n_val) :], self.post_collision[(n_train + n_val) :]

        train_dataset = SyntheticCollisionDataset(data=(self,) + train_data)
        val_dataset = SyntheticCollisionDataset(data=(self,) + val_data)
        test_dataset = SyntheticCollisionDataset(data=(self,) + test_data)

        return train_dataset, val_dataset, test_dataset

    def denormalize(self, norm_data):
        if not self.normalize:
            raise Exception("This dataset wasn't normalized")

        return norm_data * (self.max_for_norm - self.min_for_norm) + self.min_for_norm

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        pre_collision_path = os.path.join(path, "pre_collision.pt")
        post_collision_path = os.path.join(path, "post_collision.pt")
        density_path = os.path.join(path, "density.pt")
        metadata_path = os.path.join(path, "metadata.pkl")

        torch.save(self.pre_collision, pre_collision_path)
        torch.save(self.post_collision, post_collision_path)
        if self.normalize:
            torch.save(self.density, density_path)
        with open(metadata_path, "wb") as f:
            metadata = {
                "normalize": self.normalize,
                # "max_for_norm": self.max_for_norm if self.normalize else None,
                # "min_for_norm": self.min_for_norm if self.normalize else None,
                "num_samples": self.num_samples,
            }
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, path):
        pre_collision_path = os.path.join(path, "pre_collision.pt")
        post_collision_path = os.path.join(path, "post_collision.pt")
        density_path = os.path.join(path, "density.pt")
        metadata_path = os.path.join(path, "metadata.pkl")

        pre_collision = torch.load(pre_collision_path)
        post_collision = torch.load(post_collision_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Create a "father" object to pass to the initializer
        father = cls()
        father.normalize = metadata["normalize"]
        if father.normalize: 
            density = torch.load(density_path)
            father.density = density
        # if father.normalize:
        #     father.max_for_norm = metadata["max_for_norm"]
        #     father.min_for_norm = metadata["min_for_norm"]

        data = (father, pre_collision, post_collision)
        return cls(data=data)
