import numpy as np
import torch
import lightning as L
from scipy.special import lambertw
import random
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset


from typing import NamedTuple
import os
from pathlib import Path


class SyntheticDataset(Dataset):
    def __init__(
        self,
        path: Path,
        normalize=False,
    ):

        self.dataX = torch.load(os.path.join(path, "pre_equilibrium.pt"))
        self.dataY = torch.load(os.path.join(path, "post_equilibrium.pt"))
        self.density = torch.load(os.path.join(path, "density.pt"))
        self.min_for_norm = torch.load(os.path.join(path, "normalization_min.pt"))
        self.max_for_norm = torch.load(os.path.join(path, "normalization_max.pt"))
        self.mean_for_norm = torch.load(os.path.join(path, "normalization_mean.pt"))
        self.std_for_norm = torch.load(os.path.join(path, "normalization_std.pt"))

        self.normalize = normalize
        self.n_samples = self.dataX.shape[0]

        if self.normalize:
            self.dataX = (self.dataX - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)
            self.dataY = (self.dataY - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)

            # self.dataX = self.dataX / self.density.unsqueeze(-1)
            # self.dataY = self.dataY / self.density.unsqueeze(-1)
            # self.min_for_norm = torch.min(self.dataX, dim=0).values
            # self.max_for_norm = torch.max(self.dataX, dim=0).values
            # self.dataX = (self.dataX - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)
            # self.dataY = (self.dataY - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_samples:
            raise IndexError("Wrong index")

        return self.dataX[idx], self.dataY[idx]

    def denormalize(self, norm_data):
        if not self.normalize:
            raise Exception("This dataset wasn't normalized")

        return norm_data * (self.max_for_norm - self.min_for_norm) + self.min_for_norm
        # return norm_data * self.density.unsqueeze(-1)


class SimulatioDataset(Dataset):
    def __init__(
        self,
        normalize=False,
    ):

        self.dataX = torch.load("/Users/jwinter/Research/00_Code/TorchLBM/cases/TaylorGreenVortex/pytorch_output/pre_equilibrium_0.00049020.pt")
        self.dataY = torch.load("/Users/jwinter/Research/00_Code/TorchLBM/cases/TaylorGreenVortex/pytorch_output/post_equilibrium_0.00049020.pt")
        self.density = torch.load("/Users/jwinter/Research/00_Code/TorchLBM/cases/TaylorGreenVortex/pytorch_output/density_equilibrium_0.00049020.pt")

        self.normalize = normalize
        self.n_samples = self.dataX.shape[0]

        if self.normalize:
            self.dataX = self.dataX / self.density.unsqueeze(-1)
            self.dataY = self.dataY / self.density.unsqueeze(-1)
            self.dataX = (self.dataX - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)
            self.dataY = (self.dataY - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)

            # self.min_for_norm = torch.min(self.dataX, dim=0).values
            # self.max_for_norm = torch.max(self.dataX, dim=0).values
            # self.dataX = (self.dataX - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)
            # self.dataY = (self.dataY - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)

        print(f"Mean pre equilibrium:\n{torch.mean(self.dataX, dim=0)}")
        print(f"Mean post equilibrium:\n{torch.mean(self.dataY, dim=0)}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_samples:
            raise IndexError("Wrong index")

        return self.dataX[idx], self.dataY[idx]

    def denormalize(self, norm_data):
        if not self.normalize:
            raise Exception("This dataset wasn't normalized")

        norm_data = norm_data * (self.max_for_norm - self.min_for_norm) + self.min_for_norm
        norm_data = norm_data * self.density.unsqueeze(-1)
        return norm_data


class EquilibriumDataModule(L.LightningDataModule):
    def __init__(self, path: Path, normalize: bool, batch_size: int = 32):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.normalize = normalize
        self.min_for_norm = torch.load(os.path.join(path, "normalization_min.pt"))
        self.max_for_norm = torch.load(os.path.join(path, "normalization_max.pt"))
        self.mean_for_norm = torch.load(os.path.join(path, "normalization_mean.pt"))
        self.std_for_norm = torch.load(os.path.join(path, "normalization_std.pt"))
        self.lattice_velocities = torch.load(os.path.join(path, "lattice_velocities.pt"))

    def setup(self, stage: str):
        print("\n\n\n\n\n\nLoad existing dataset\n\n\n\n\n\n")
        dataset = SyntheticDataset(self.path, self.normalize)
        # simulation_dataset = SimulatioDataset(self.normalize)

        num_val_samples = int(0.1 * len(dataset))
        print(f"Num_val_samples: {num_val_samples}")
        num_test_samples = int(0.1 * len(dataset))
        print(f"Num_test_samples: {num_test_samples}")
        num_train_samples = len(dataset) - num_val_samples - num_test_samples
        print(f"Num_train_samples: {num_train_samples}")

        self.train_set, self.val_set, self.test_set = random_split(
            dataset,
            [num_train_samples, num_val_samples, num_test_samples],
            generator=torch.Generator().manual_seed(42),
        )
        # self.val_set = simulation_dataset
        # self.test_set = simulation_dataset

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
