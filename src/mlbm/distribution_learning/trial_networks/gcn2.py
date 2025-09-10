import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import lightning.pytorch as pl

from torchlbm.core.lattices.d2q9 import D2Q9
from mlbm.distribution_learning.networks.loss_utilities import LossOption, get_loss_function
from mlbm.distribution_learning.networks.activation_utilities import ActivationOption, get_activation_function


class GraphCollisionLayer2(gnn.MessagePassing):
    def __init__(self, emb_dim, edge_dim, aggr="add", bias=True, residual_connection=False, activation=nn.ReLU):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.residual_connection = residual_connection

        self.edge_input_dim = 2 * self.emb_dim + edge_dim + 1 + 1

        # self.mlp_msg = nn.Sequential(
        #     nn.Linear(self.edge_input_dim, 2 * self.emb_dim, bias=bias),
        #     activation(),
        #     nn.Linear(2 * self.emb_dim, self.emb_dim, bias=bias),
        #     activation(),
        # )
        self.mlp_msg = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.emb_dim, bias=bias),
            activation(),
            # nn.Linear(2 * self.emb_dim, self.emb_dim, bias=bias),
            # activation(),
        )

        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * self.emb_dim + 1, self.emb_dim, bias=bias),
            activation(),
            # nn.Linear(self.emb_dim, self.emb_dim, bias=bias),
        )

    def forward(self, h, edge_index, x, omega, edge_attr=None):
        print("Graph forward pass with shapes:")
        print(h.shape, edge_index.shape, x.shape, omega.shape)
        return (
            self.propagate(edge_index, h=h, x=x, omega=omega) if edge_attr is None else self.propagate(edge_index, h=h, x=x, omega=omega, edge_attr=edge_attr)
        )

    def message(self, h_i, h_j, x_i, x_j, omega_i, edge_attr=None):
        print("Graph message shapes:")
        print(h_i.shape, h_j.shape, omega_i.shape)
        print(a)
        coord_diff = torch.sum((x_i - x_j) ** 2, dim=1).unsqueeze(1)
        tmp = torch.cat([h_i, h_j, omega_i, coord_diff], dim=-1) if edge_attr is None else torch.cat([h_i, h_j, omega_i, coord_diff, edge_attr], dim=-1)
        return self.mlp_msg(tmp)

    def update(self, aggr_out, h, omega):
        tmp = torch.cat([omega, h, aggr_out], dim=-1)
        return h + self.mlp_upd(tmp) if self.residual_connection else self.mlp_upd(tmp)


class GraphCollisionNetwork2(pl.LightningModule):
    def __init__(self, hparams, input_dim, edge_dim, output_dim):
        super().__init__()
        self.save_hyperparameters(hparams)

        # self.hparams = hparams
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim

        # self.device = self.hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.emb_dim = self.hparams.get("emb_dim", 20)
        self.aggr = self.hparams.get("aggr", "add")
        self.bias = self.hparams.get("bias", True)
        self.residual_connection = self.hparams.get("residual_connection", False)
        self.pre_collision_residual = self.hparams.get("pre_collision_residual", False)
        self.activation = get_activation_function(self.hparams.get("activation", ActivationOption.relu))[0]

        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.emb_dim, bias=self.bias), self.activation())
        self.layer1 = GraphCollisionLayer2(
            emb_dim=self.emb_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias, residual_connection=self.residual_connection, activation=self.activation
        )
        # self.layer2 = GraphCollisionLayer(
        #     emb_dim=self.emb_dim, edge_dim=self.edge_dim, aggr=self.aggr, bias=self.bias, residual_connection=self.residual_connection, activation=self.activation
        # )
        self.decoder = nn.Sequential(nn.Linear(self.emb_dim, self.output_dim, bias=self.bias), self.activation())

        lattice = D2Q9()
        self.lattice_velocities = torch.tensor(lattice.lattice_velocities())
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)

        self.loss_function = get_loss_function(self.hparams.get("loss_function", LossOption.l1))  # Default is L1 loss
        self.per_dimension_loss = get_loss_function(LossOption.per_dimension)
        self.momentum_loss = get_loss_function(LossOption.momentum)
        self.mass_loss = get_loss_function(LossOption.mass)
        self.msre_loss = get_loss_function(LossOption.msre)

    def forward(self, h, edge_index, x, omega, edge_attr=None):
        fpre = h
        fpre = fpre[:, 0].unsqueeze(-1)
        h = self.encoder(h)
        h = self.layer1(h, edge_index, x, omega, edge_attr)
        # h = self.layer2(h, edge_index, x, omega, edge_attr)
        h = self.layer1(h, edge_index, x, omega, edge_attr)
        h = self.decoder(h)
        if self.pre_collision_residual:
            h = h + fpre

        fpred = h.reshape(-1, 9)
        fpre = fpre.reshape(-1, 9)
        fcorr = (
            fpred
            - 1 / 9 * torch.sum(fpred - fpre, dim=1).unsqueeze(-1)
            - 1 / 6 * torch.einsum("dQ,dN->NQ", self.lattice_velocities_const, torch.einsum("dQ,NQ->dN", self.lattice_velocities_const, fpred - fpre))
        )
        fcorr = fcorr.reshape(-1, 1)
        return fcorr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 1e-3), weight_decay=self.hparams.get("weight_decay", 0))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.get("step_size", 2), gamma=self.hparams.get("gamma", 0.8))
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def calculate_losses(self, y_true, y_pred, stage):

        loss = self.loss_function(y_true, y_pred)
        self.log(f"{stage}_loss", loss, batch_size=128)
        ytrue = y_true.reshape(-1, 9)
        ypred = y_pred.reshape(-1, 9)

        per_dimension_loss = self.per_dimension_loss(ytrue, ypred)
        for idx, dimension_loss in enumerate(per_dimension_loss.tolist()):
            self.log(f"{stage}_population_accuracy_{idx}", dimension_loss, batch_size=128)

        mass_loss = self.mass_loss(ytrue, ypred)
        self.log(f"{stage}_mass_accuracy", mass_loss, batch_size=128)

        momentum_loss_tensor = self.momentum_loss(ytrue, ypred, self.lattice_velocities_const)
        for idx, single_momentum_loss in enumerate(momentum_loss_tensor.tolist()):
            self.log(f"{stage}_momentum_accuracy_{idx}", single_momentum_loss, batch_size=128)

        msre_loss = self.msre_loss(ytrue, ypred)
        self.log(f"{stage}_msre_loss", msre_loss, batch_size=128)

        loss_to_optimize = loss
        self.log(f"{stage}_loss_to_optimize", loss_to_optimize, prog_bar=True, batch_size=128)
        return loss_to_optimize

    def training_step(self, batch, batch_idx):
        h, edge_index, x, omega, edge_attr = batch.x, batch.edge_index, batch.pos, batch.omega, batch.edge_attr
        # h = torch.cat([batch.x, batch.weights], dim=1)
        out = self.forward(h, edge_index, x, omega) if edge_attr is None else self.forward(h, edge_index, x, omega, edge_attr)
        loss = self.calculate_losses(batch.y, out, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        h, edge_index, x, omega, edge_attr = batch.x, batch.edge_index, batch.pos, batch.omega, batch.edge_attr
        # h = torch.cat([batch.x, batch.weights], dim=1)
        out = self.forward(h, edge_index, x, omega) if edge_attr is None else self.forward(h, edge_index, x, omega, edge_attr)
        loss = self.calculate_losses(batch.y, out, "val")
        return loss
