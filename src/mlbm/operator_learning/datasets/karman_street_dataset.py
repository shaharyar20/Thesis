import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List


class KarmanStreetDataset(Dataset):

    def __init__(self, data_file: str, transform=None, target_transform=None):
        self.data = torch.load(data_file)
        self.data = self.data[200:]
        self.data_x = self.data[:-1]
        self.data_y = self.data[1:]

        self.num_elements = len(self.data_x)
        self.normalization_mean = self.data_x.mean()
        print(f"self.normalization_mean: {self.normalization_mean}")
        self.normalization_std = self.data_x.std()
        print(f"self.normalization_std: {self.normalization_std}")
        self.data_x = (self.data_x - self.normalization_mean) / self.normalization_std
        self.data_y = (self.data_y - self.normalization_mean) / self.normalization_std

        # new_indices = torch.randperm(self.num_elements)
        # self.data_x = self.data_x[new_indices]
        # self.data_y = self.data_y[new_indices]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        input_tensor = self.data_x[idx, ...]
        output_tensor = self.data_y[idx, ...]
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            output_tensor = self.target_transform(output_tensor)
        return input_tensor, output_tensor


class MixedReKarmanStreetDataset(Dataset):

    def __init__(self, base_folder: str, Re_list: List[int], field_name: str, num_channels: int, transform=None, target_transform=None):
        self.data_x = torch.empty([0, num_channels, 512, 256])
        self.data_y = torch.empty([0, num_channels, 512, 256])

        for Re in Re_list:
            data_file = Path(base_folder).joinpath(f"{field_name}_karman_Re_{Re}.pt")
            data = torch.load(data_file)
            data = data[350:]

            self.data_x = torch.cat([self.data_x, data[:-1]], dim=0)
            self.data_y = torch.cat([self.data_y, data[1:]], dim=0)

        self.num_elements = len(self.data_x)

        # new_indices = torch.randperm(self.num_elements)
        # self.data_x = self.data_x[new_indices]
        # self.data_y = self.data_y[new_indices]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        input_tensor = self.data_x[idx, ...]
        output_tensor = self.data_y[idx, ...]
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            output_tensor = self.target_transform(output_tensor)
        return input_tensor, output_tensor


class PorousDataset(Dataset):

    def __init__(
        self,
        base_folder: str,
        resolution: int,
        field_name: str,
        num_channels: int,
        data_y_mean: List[int],
        data_y_std: List[int],
        transform=None,
        target_transform=None,
    ):
        self.data_x = torch.empty([0, num_channels, resolution, resolution, resolution])
        self.data_y = torch.empty([0, num_channels, resolution, resolution, resolution])

        data_file_x = Path(base_folder).joinpath(f"bounce_back_mask_inside_porous_resolution_{resolution}.pt")
        self.data_x = torch.load(data_file_x)

        data_file_y = Path(base_folder).joinpath(f"{field_name}_inside_porous_resolution_{resolution}.pt")
        self.data_y = torch.load(data_file_y)
        self.data_y_mean = torch.tensor(data_y_mean)
        data_y_mean_prepared = self.data_y_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.data_y_std = torch.tensor(data_y_std)
        data_y_std_prepared = self.data_y_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # data_y_mean = self.data_y.mean(dim=[0,2,3,4]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print(data_y_mean)
        # data_y_std = self.data_y.std(dim=[0,2,3,4]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print(data_y_std)

        self.data_y = (self.data_y - data_y_mean_prepared) / data_y_std_prepared

        self.num_elements = len(self.data_x)

        # new_indices = torch.randperm(self.num_elements)
        # self.data_x = self.data_x[new_indices]
        # self.data_y = self.data_y[new_indices]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        input_tensor = self.data_x[idx, ...]
        output_tensor = self.data_y[idx, ...]
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            output_tensor = self.target_transform(output_tensor)
        return input_tensor, output_tensor
