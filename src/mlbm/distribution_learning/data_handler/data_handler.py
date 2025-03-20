from typing import List
import os
from pathlib import Path
from enum import Enum

import torch

import torchlbm.standalone_operations.file_operations as file_o
from torchlbm.exceptions import TorchlbmError


class NormalizationOption(Enum):
    none = 0
    zero_one = 1
    mean_std = 2
    density = 2


@torch.jit.script
class DataHandler:

    def __init__(
        self,
        min_for_norm,
        max_for_norm,
        mean_for_norm,
        std_for_norm,
        lattice_velocities,
        normalization_option: NormalizationOption,
    ) -> None:

        self.min_for_norm = min_for_norm
        self.max_for_norm = max_for_norm
        self.mean_for_norm = mean_for_norm
        self.std_for_norm = std_for_norm
        self.lattice_velocities = lattice_velocities
        self.normalization_option: NormalizationOption = normalization_option
        self.number_discrete_velocities = self.min_for_norm.shape[0]

    def log(self) -> None:
        print(f"Initialized min_for_norm to: {self.min_for_norm}")
        print(f"Initialized max_for_norm to: {self.max_for_norm}")
        print(f"Initialized mean_for_norm to: {self.mean_for_norm}")
        print(f"Initialized std_for_norm to: {self.std_for_norm}")
        print(f"Initialized lattice_velocities to: {self.lattice_velocities}")
        print(f"The number of discrete velocities is: {self.number_discrete_velocities}")
        print(f"The normalization option is: {self.normalization_option.name}")

    def normalize(self, data: torch.Tensor) -> torch.Tensor:

        if data.shape[-1] != self.number_discrete_velocities:
            raise TorchlbmError("The shape of the data that should be normalized does not match!\nThe shape is: {data.shape}.")

        if self.normalization_option == NormalizationOption.none:
            return data
        if self.normalization_option == NormalizationOption.zero_one:
            return (data - self.min_for_norm) / (self.max_for_norm - self.min_for_norm)
        elif self.normalization_option == NormalizationOption.mean_std:
            return (data - self.mean_for_norm) / self.std_for_norm
        raise TorchlbmError("No suitable normalization option")

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:

        if data.shape[-1] != self.number_discrete_velocities:
            raise TorchlbmError("The shape of the data that should be denormalized does not match!\nThe shape is: {data.shape}.")

        if self.normalization_option == NormalizationOption.none:
            return data
        if self.normalization_option == NormalizationOption.zero_one:
            return data * (self.max_for_norm - self.min_for_norm) + self.min_for_norm
        elif self.normalization_option == NormalizationOption.mean_std:
            return data * self.std_for_norm + self.mean_for_norm
        raise TorchlbmError("No suitable denormalization option")

    # def to_device(self, device):
    #     self.min_for_norm = self.min_for_norm.to(device=device)
    #     self.max_for_norm = self.max_for_norm.to(device=device)
    #     self.mean_for_norm = self.mean_for_norm.to(device=device)
    #     self.std_for_norm = self.std_for_norm.to(device=device)
    #     self.lattice_velocities = self.lattice_velocities.to(device=device)
