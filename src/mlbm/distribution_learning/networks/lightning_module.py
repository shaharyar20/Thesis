import lightning.pytorch as pl
import torch

from mlbm.distribution_learning.networks.activation_utilities import ActivationOption
from mlbm.distribution_learning.networks.loss_utilities import LossOption, get_loss_function
from mlbm.distribution_learning.data_handler.data_handler import DataHandler
from mlbm.distribution_learning.networks.networks import NaiveNetworkProd
from mlbm.distribution_learning.networks.equivariance_utilities import rotate_data, derotate_data

import torch.optim.lr_scheduler as lr_scheduler
from torch import optim


class SymmetryNetwork(pl.LightningModule):
    def __init__(
        self,
        data_handler: DataHandler,
        activation_option: ActivationOption,
        last_activation_option: ActivationOption,
        loss_option: LossOption,
        bias=True,
        layers=[20, 55, 55],
        lr=0.00005,
        reg=0,
        scheduler=[-1, 1],
    ):

        super(SymmetryNetwork, self).__init__()
        self.save_hyperparameters()

        self.model: NaiveNetworkProd = NaiveNetworkProd(
            data_handler=data_handler,
            activation_option=activation_option,
            last_activation_option=last_activation_option,
            loss_option=loss_option,
            bias=bias,
            layers=layers,
        )
        self.loss_function = get_loss_function(loss_option)
        self.per_dimension_loss = get_loss_function(LossOption.per_dimension)
        self.mass_loss = get_loss_function(LossOption.mass)
        self.momentum_loss = get_loss_function(LossOption.momentum)

    def on_fit_start(self):
        self.model.data_handler.min_for_norm = self.model.data_handler.min_for_norm.to(device=self.device)
        self.model.data_handler.max_for_norm = self.model.data_handler.max_for_norm.to(device=self.device)
        self.model.data_handler.mean_for_norm = self.model.data_handler.mean_for_norm.to(device=self.device)
        self.model.data_handler.std_for_norm = self.model.data_handler.std_for_norm.to(device=self.device)
        self.model.data_handler.lattice_velocities = self.model.data_handler.lattice_velocities.to(device=self.device)

    def on_fit_end(self):
        self.model.data_handler.min_for_norm = self.model.data_handler.min_for_norm.to(device=self.device)
        self.model.data_handler.max_for_norm = self.model.data_handler.max_for_norm.to(device=self.device)
        self.model.data_handler.mean_for_norm = self.model.data_handler.mean_for_norm.to(device=self.device)
        self.model.data_handler.std_for_norm = self.model.data_handler.std_for_norm.to(device=self.device)
        self.model.data_handler.lattice_velocities = self.model.data_handler.lattice_velocities.to(device=self.device)

    @torch.jit.export
    def run_inference(self, x, normalize_with_density: bool = False):
        density = torch.sum(x, dim=-1).unsqueeze(-1)
        if normalize_with_density:
            x = x / density
        rotated_data = rotate_data(x)
        logits_models = [self.model(data) for data in rotated_data]
        average = torch.zeros_like(x)
        for ele in derotate_data(logits_models):
            average = average + ele
        average = average / 8.0
        if normalize_with_density:
            average = average * density
        return average

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.reg)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss_to_optimize"}  # Changed scheduler to lr_scheduler

    def calculate_losses(self, y_true: torch.tensor, y_predicted: torch.tensor, stage: str) -> float:

        loss = self.loss_function(y_true / self.model.data_handler.mean_for_norm, y_predicted / self.model.data_handler.mean_for_norm)
        self.log(f"{stage}_loss", loss)

        per_dimension_loss = self.per_dimension_loss(y_true, y_predicted)
        for idx, dimension_loss in enumerate(per_dimension_loss.tolist()):
            self.log(f"{stage}_population_accuracy_{idx}", dimension_loss)

        mass_loss = self.mass_loss(y_true, y_predicted)
        self.log(f"{stage}_mass_accuracy", mass_loss)

        momentum_loss_tensor = self.momentum_loss(y_true, y_predicted, self.model.data_handler.lattice_velocities)
        for idx, single_momentum_loss in enumerate(momentum_loss_tensor.tolist()):
            self.log(f"{stage}_momentum_accuracy_{idx}", single_momentum_loss)

        loss_to_optimize = loss
        self.log(f"{stage}_loss_to_optimize", loss_to_optimize, prog_bar="True")
        return loss_to_optimize

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.run_inference(x)
        loss = self.calculate_losses(y, out, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.run_inference(x)
        loss = self.calculate_losses(y, out, stage="val")
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.run_inference(x)
        loss = self.loss_function(y, out)
        return loss
