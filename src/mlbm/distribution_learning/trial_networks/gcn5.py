import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import lightning.pytorch as pl

from torchlbm.core.lattices.d2q9 import D2Q9
from mlbm.distribution_learning.networks.loss_utilities import LossOption, get_loss_function
from mlbm.distribution_learning.networks.activation_utilities import ActivationOption, get_activation_function


class GraphCollisionLayer5(gnn.MessagePassing):
    def __init__(self, mlp_msg, mlp_upd, coord_diff, aggr="add"):
        super().__init__(aggr=aggr)

        self.mlp_msg = mlp_msg
        self.mlp_upd = mlp_upd
        self.coord_diff = coord_diff
        self.register_buffer("coord_diff_const", self.coord_diff)

    def forward(self, h, edge_index, omega):
        # print("Graph forward pass with shapes:")
        # print(h.shape, edge_index.shape, omega.shape, self.coord_diff_const.shape)
        return self.propagate(edge_index, h=h, omega=omega)

    def message(self, h_i, h_j):
        # print("Graph message shapes:")
        # print(h_i.shape, h_j.shape, omega_i.shape, self.coord_diff_const.shape)
        # print(a)
        return self.mlp_msg(
            torch.cat([h_i, h_j, self.coord_diff_const.expand(h_i.shape[0], -1, -1)], dim=-1)
        )

    def update(self, aggr_out, h, omega):
        # print("Graph update shapes:")
        # print(h.shape, omega.shape, aggr_out.shape)
        # print(a)
        return h + self.mlp_upd(torch.cat([h, omega.expand(h.shape[0], h.shape[1], -1), aggr_out], dim=-1)) 


class GraphCollisionNetwork5(pl.LightningModule):
    def __init__(self, hparams, input_dim, edge_dim, output_dim):
        super().__init__()
        self.save_hyperparameters(hparams)

        # self.hparams = hparams
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim

        self.edge_index = torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 5, 2, 6, 3, 7, 4, 8, 5, 2, 6, 3, 7, 4, 8, 1],
                    [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 6, 3, 7, 4, 8, 1, 1, 5, 2, 6, 3, 7, 4, 8],
                ],
            )   
        self.register_buffer("edge_index_const", self.edge_index)

        # self.device = self.hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.emb_dim = self.hparams.get("emb_dim", 20)
        self.aggr = self.hparams.get("aggr", "add")
        self.bias = self.hparams.get("bias", True)
        self.residual_connection = self.hparams.get("residual_connection", False)
        self.pre_collision_residual = self.hparams.get("pre_collision_residual", False)
        self.activation = get_activation_function(self.hparams.get("activation", ActivationOption.relu))[0]

        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.emb_dim, bias=self.bias), self.activation())

        self.decoder = nn.Sequential(nn.Linear(self.emb_dim, self.output_dim, bias=self.bias), self.activation())

        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * self.emb_dim + 1, self.emb_dim, bias=self.bias),
            self.activation(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias),
            self.activation(),
        )

        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * self.emb_dim + 1, self.emb_dim, bias=self.bias),
            self.activation(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias),
        )

        lattice = D2Q9()
        self.lattice_velocities = torch.tensor(lattice.lattice_velocities())
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)

        pos = self.lattice_velocities.transpose(0, 1)
        coord_diff_i = pos[self.edge_index[0, :], :]
        coord_diff_j = pos[self.edge_index[1, :], :]
        self.coord_diff = torch.sum((coord_diff_i - coord_diff_j)**2, dim=-1).unsqueeze(-1)

        self.layer = GraphCollisionLayer5(
            mlp_msg=self.mlp_msg,
            mlp_upd=self.mlp_upd,
            coord_diff=self.coord_diff,
            aggr=self.aggr,
        )


        self.loss_function = get_loss_function(self.hparams.get("loss_function", LossOption.l1))  # Default is L1 loss
        self.per_dimension_loss = get_loss_function(LossOption.per_dimension)
        self.momentum_loss = get_loss_function(LossOption.momentum)
        self.mass_loss = get_loss_function(LossOption.mass)
        self.msre_loss = get_loss_function(LossOption.msre)
        self.rmsre_loss = get_loss_function(LossOption.rmsre)

    def forward(self, h, omega):
        # fpre = h
        # # fpre = fpre[:, 0].unsqueeze(-1)
        # h = self.encoder(h)
        # # print("Forward pass with shapes:")
        # # print(h.shape, self.edge_index_const.shape, omega.shape)
        # # print(a)
        # h = self.layer(h, self.edge_index_const, omega)
        # h = self.layer(h, self.edge_index_const, omega)
        # h = self.decoder(h)
        # if self.pre_collision_residual:
        #     h = h + fpre
        # out = self.decoder(self.layer(self.layer(self.encoder(h), self.edge_index_const, omega), self.edge_index_const, omega)).squeeze(-1)
        # out -= (1/9 * torch.sum(out, dim=1).unsqueeze(-1) + 1/6 * torch.einsum("dQ,dN->NQ", self.lattice_velocities_const, torch.einsum("dQ,NQ->dN", self.lattice_velocities_const, out)))
        # return out.unsqueeze(-1) + h

        out = self.decoder(self.layer(self.layer(self.encoder(h), self.edge_index_const, omega), self.edge_index_const, omega)).squeeze(-1)
        out = out - 1/9 * torch.sum(out, dim=1).unsqueeze(-1) - 1/6 * torch.einsum("dQ,dN->NQ", self.lattice_velocities_const, torch.einsum("dQ,NQ->dN", self.lattice_velocities_const, out))
        return h + out.unsqueeze(-1)
        # out = out - 1/9 * torch.sum(out - h.squeeze(-1), dim=1).unsqueeze(-1) - 1/6 * torch.einsum("dQ,dN->NQ", self.lattice_velocities_const, torch.einsum("dQ,NQ->dN", self.lattice_velocities_const, out - h.squeeze(-1)))
        # return out.unsqueeze(-1)
        # print("Post forward pass shapes:")
        # print(h.shape, fpre.shape)
        # print(a)
        # fpred = h.squeeze(-1)
        # fpre = fpre.squeeze(-1)
        # fcorr = (
        #     fpred
        #     - 1 / 9 * torch.sum(fpred - fpre, dim=1).unsqueeze(-1)
        #     - 1 / 6 * torch.einsum("dQ,dN->NQ", self.lattice_velocities_const, torch.einsum("dQ,NQ->dN", self.lattice_velocities_const, fpred - fpre))
        # )
        # fcorr = fcorr.reshape(-1, 1)
        # print("Final output shape:")
        # print(fcorr.shape)
        # print(a)
        # return fcorr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 1e-3), weight_decay=self.hparams.get("weight_decay", 0))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.get("step_size", 2), gamma=self.hparams.get("gamma", 0.8))
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     factor=self.hparams.get("gamma", 0.8),
        #     patience= self.hparams.get("step_size", 2),
        # )
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

        rmsre_loss = self.rmsre_loss(ytrue, ypred)
        self.log(f"{stage}_rmsre_loss", rmsre_loss, batch_size=128)

        loss_to_optimize = loss
        self.log(f"{stage}_loss_to_optimize", loss_to_optimize, prog_bar=True, batch_size=128)
        return loss_to_optimize

    def training_step(self, batch, batch_idx):
        h, edge_index, x, omega, edge_attr = batch.x, batch.edge_index, batch.pos, batch.omega, batch.edge_attr
        # h = torch.cat([batch.x, batch.weights], dim=1)
        h = h.reshape(-1, 9).unsqueeze(-1)
        omega = omega.reshape(-1, 9).unsqueeze(-1)
        omega = omega[:, 0, :].unsqueeze(-1)
        # print(omega.shape)
        # Log distance for omega
        # omega = torch.log(2.0 - omega)
        # print(omega.shape)
        # print(h.shape, edge_index.shape, x.shape, omega.shape)
        # print(a)
        out = self.forward(h, omega).reshape(-1, 1)
        loss = self.calculate_losses(batch.y, out, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        h, edge_index, x, omega, edge_attr = batch.x, batch.edge_index, batch.pos, batch.omega, batch.edge_attr
        # h = torch.cat([batch.x, batch.weights], dim=1)
        h = h.reshape(-1, 9).unsqueeze(-1)
        omega = omega.reshape(-1, 9).unsqueeze(-1)
        omega = omega[:, 0, :].unsqueeze(-1)
        # print(omega.shape)
        # Log distance for omega
        # omega = torch.log(2.0 - omega)
        # print(omega.shape)
        # print("Validation step with shapes:")
        # print(h.shape, edge_index.shape, x.shape, omega.shape)
        # print(h[3, :, :], omega[3, :, :])
        # print(a)
        out = self.forward(h, omega).reshape(-1, 1)
        # print(out.shape, batch.y.shape)
        # print(out[18:27], batch.y[18:27])
        # print(a)
        loss = self.calculate_losses(batch.y, out, "val")
        return loss
