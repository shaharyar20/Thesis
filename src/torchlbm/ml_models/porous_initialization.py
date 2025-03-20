from typing import List, Tuple
import torch
import inspect
import torch.nn as nn
from dataclasses import dataclass
from modulus.models.meta import ModelMetaData
from modulus.registry import ModelRegistry
from modulus.launch.utils import load_checkpoint
from modulus.models.fno import FNO
import omegaconf

from torchlbm.node_data import NodeData
from torchlbm.unit_converter import UnitConverter


@dataclass
class PorousInitializationMetaData(modulus.ModelMetaData):
    name: str = "PorousInitialization"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class PorousInitializationModule(modulus.Module):
    def __init__(
        self,
        unit_converter: UnitConverter,
        num_halos: int,
        dimension: int,
    ) -> None:
        super(PorousInitializationModule, self).__init__(meta=PorousInitializationMetaData())

        self.unit_converter: UnitConverter = unit_converter

        cfg_velocity = omegaconf.OmegaConf.load("/home/jwinter/Development/TorchLBM/mlbm/config_porous_velocity.yaml")
        cfg_density = omegaconf.OmegaConf.load("/home/jwinter/Development/TorchLBM/mlbm/config_porous_density.yaml")

        self.velocity_model = FNO(
            in_channels=cfg_velocity.arch.fno.in_channels,
            out_channels=cfg_velocity.arch.decoder.out_features,
            decoder_layers=cfg_velocity.arch.decoder.layers,
            decoder_layer_size=cfg_velocity.arch.decoder.layer_size,
            dimension=cfg_velocity.arch.fno.dimension,
            latent_channels=cfg_velocity.arch.fno.latent_channels,
            num_fno_layers=cfg_velocity.arch.fno.fno_layers,
            num_fno_modes=cfg_velocity.arch.fno.fno_modes,
            padding=cfg_velocity.arch.fno.padding,
        ).eval()
        velocity_ckpt_args = {
            "path": f"/home/jwinter/Development/TorchLBM/mlbm/FNOporousVelocity",
            "models": self.velocity_model,
        }
        velocity_loaded_pseudo_epoch = load_checkpoint(device="cpu", **velocity_ckpt_args)

        self.density_model = FNO(
            in_channels=cfg_density.arch.fno.in_channels,
            out_channels=cfg_density.arch.decoder.out_features,
            decoder_layers=cfg_density.arch.decoder.layers,
            decoder_layer_size=cfg_density.arch.decoder.layer_size,
            dimension=cfg_density.arch.fno.dimension,
            latent_channels=cfg_density.arch.fno.latent_channels,
            num_fno_layers=cfg_density.arch.fno.fno_layers,
            num_fno_modes=cfg_density.arch.fno.fno_modes,
            padding=cfg_density.arch.fno.padding,
        ).eval()
        density_ckpt_args = {
            "path": f"/home/jwinter/Development/TorchLBM/mlbm/FNOporousDensity",
            "models": self.density_model,
        }
        density_loaded_pseudo_epoch = load_checkpoint(device="cpu", **density_ckpt_args)

        self.start = [
            num_halos,
            num_halos if dimension != 1 else 0,
            num_halos if dimension == 3 else 0,
        ]

        self.end = [
            -num_halos,
            -num_halos if dimension != 1 else 1,
            -num_halos if dimension == 3 else 1,
        ]

        self.density_normalization_mean = torch.tensor([1.0])
        self.register_buffer("density_normalization_mean_const", self.density_normalization_mean)
        self.density_normalization_std = torch.tensor([0.0004])
        self.register_buffer("density_normalization_std_const", self.density_normalization_std)
        self.velocity_normalization_mean = torch.tensor([1.345, 0.0, 0.0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("velocity_normalization_mean_const", self.velocity_normalization_mean)
        self.velocity_normalization_std = torch.tensor([1.118, 0.1522, 0.1522]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("velocity_normalization_std_const", self.velocity_normalization_std)

    def forward(self, node_data: NodeData) -> NodeData:
        bounce_back_mask_for_operator = (
            node_data.bounce_back_mask[self.start[0] : self.end[0], self.start[1] : self.end[1], self.start[2] : self.end[2]].unsqueeze(0).unsqueeze(0)
        )

        density_after_operator = (
            self.density_model(bounce_back_mask_for_operator).squeeze() * self.density_normalization_std_const + self.density_normalization_mean_const
        )
        velocity_after_operator = (
            self.velocity_model(bounce_back_mask_for_operator).squeeze() * self.velocity_normalization_std_const + self.velocity_normalization_mean_const
        )

        density_after_operator = self.unit_converter.convert_density_to_lattice_units(density_after_operator)
        velocity_after_operator = self.unit_converter.convert_velocity_to_lattice_units(velocity_after_operator)

        node_data.moments.density[self.start[0] : self.end[0], self.start[1] : self.end[1], self.start[2] : self.end[2]] = density_after_operator
        node_data.moments.velocity[:, self.start[0] : self.end[0], self.start[1] : self.end[1], self.start[2] : self.end[2]] = velocity_after_operator

        return node_data
