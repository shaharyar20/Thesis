from typing import List
import os
from pathlib import Path
from enum import Enum

import torch
from torch.utils.data import DataLoader, random_split

from lightning.pytorch.core.datamodule import LightningDataModule

import torchlbm.standalone_operations.file_operations as file_o
from torchlbm.exceptions import TorchlbmError
from mlbm.distribution_learning.data_handler.data_handler import DataHandler, NormalizationOption
from mlbm.distribution_learning.datasets.equilibrium_dataset import EquilibriumDataset
from mlbm.distribution_learning.data_handler.data_handler import DataHandler


class EquilibriumDataModule(LightningDataModule):

    def __init__(self, data_path: Path, data_handler: DataHandler, normalization_option: NormalizationOption) -> None:
        super().__init__()
        self.data_path: Path = file_o.get_checked_path(data_path)
        self.normalization_option = normalization_option

        self.data_handler = data_handler

        self.batch_size = 1000
        self.validation_data_portion = 0.1
        self.test_data_portion = 0.1

    def setup(self, stage: str):

        print("Setting up data based on the following data handler:")
        self.data_handler.log()
        dataset = EquilibriumDataset(
            data_path=self.data_path,
            data_handler=self.data_handler,
            normalization_option=self.normalization_option,
        )
        num_val_samples = int(self.validation_data_portion * len(dataset))
        num_test_samples = int(self.test_data_portion * len(dataset))
        num_train_samples = len(dataset) - num_val_samples - num_test_samples
        print(f"Num_train_samples: {num_train_samples}")
        print(f"Num_val_samples: {num_val_samples}")
        print(f"Num_test_samples: {num_test_samples}")

        self.train_set, self.val_set, self.test_set = random_split(
            dataset,
            [num_train_samples, num_val_samples, num_test_samples],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)
