from typing import List
import os
from pathlib import Path
from enum import Enum

import torch
from torch.utils.data import Dataset

import torchlbm.standalone_operations.file_operations as file_o
from torchlbm.exceptions import TorchlbmError
from mlbm.distribution_learning.data_handler.data_handler import DataHandler, NormalizationOption


class EquilibriumDataset(Dataset):

    def __init__(self, data_path: Path, data_handler: DataHandler, normalization_option: NormalizationOption) -> None:

        self.data_path: Path = file_o.get_checked_path(data_path)
        self.normalization_option = normalization_option

        self.data_handler = data_handler

        self.data_x: torch.tensor = torch.load(self.data_path.joinpath("pre_equilibrium.pt"))
        if self.data_x.shape[-1] != self.data_handler.number_discrete_velocities:
            raise TorchlbmError(f"The member self.data_x of the dataset does not have the correct shape! Its shape is: {self.data_x.shape}")
        self.data_y: torch.tensor = torch.load(self.data_path.joinpath("post_equilibrium.pt"))
        if self.data_y.shape[-1] != self.data_handler.number_discrete_velocities:
            raise TorchlbmError(f"The member self.data_y of the dataset does not have the correct shape! Its shape is: {self.data_y.shape}")

        self.n_samples = self.data_x.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_samples:
            raise IndexError("Wrong index for accessing an element in the dataset!")

        return self.data_x[idx], self.data_y[idx]
