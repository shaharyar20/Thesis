from typing import List

import torch
from torch import nn, optim
import lightning.pytorch as pl
import torch.optim.lr_scheduler as lr_scheduler

from mlbm.distribution_learning.networks.activation_utilities import ActivationOption, get_activation_function
from mlbm.distribution_learning.networks.loss_utilities import LossOption, get_loss_function
from mlbm.distribution_learning.data_handler.data_handler import DataHandler, NormalizationOption


class NaiveNetworkProd(nn.Module):
    def __init__(
        self,
        data_handler: DataHandler,
        activation_option: ActivationOption,
        last_activation_option: ActivationOption,
        loss_option: LossOption,
        bias: bool,
        layers: List[int],
    ):
        super(NaiveNetworkProd, self).__init__()

        activation_function, activation_params = get_activation_function(activation_option)
        last_activation_function, last_activation_params = get_activation_function(last_activation_option)
        loss_function = get_loss_function(loss_option)

        seq_layers = [nn.Linear(9, layers[0], bias=bias), activation_function(**activation_params)]

        for i in range(len(layers) - 1):
            seq_layers.append(nn.Linear(layers[i], layers[i + 1], bias=bias))
            seq_layers.append(activation_function(**activation_params))

        seq_layers.append(nn.Linear(layers[-1], 9, bias=bias))
        if last_activation_function is not None:
            seq_layers.append(last_activation_function(**last_activation_params))

        self.model = nn.Sequential(*seq_layers)
        self.data_handler: DataHandler = data_handler

        # A = [
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     [1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     [-0.5, 0.5, 0.0, -1.5, -1.0, 1.0, -1.0, -2.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     [0.5, 0.5, 0.0, 0.5, 1.0, 0, 0, 1.0, 1.0],
        # ]

        # B = [
        #     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     [-1.0, 0.0, 0.0, -2.0, -1.0, 0.0, -2.0, -2.0, 0.0],
        #     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        #     [0.5, -0.5, 0.0, 1.5, 1.0, 0.0, 1.0, 2.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        #     [-0.5, -0.5, 0.0, -0.5, -1.0, 0, 0, -1.0, 0.0],
        # ]

        A = [
            [1.0, 0.0, 0.0, 2.0, 2.0, -1.0, 1.0, 3.0, 1.0],
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        B = [
            [0.0, 0.0, 0.0, -2.0, -2.0, 1.0, -1.0, -3.0, -1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]

        self.A = torch.tensor(A)
        self.B = torch.tensor(B)
        self.register_buffer("A_const", self.A)
        self.register_buffer("B_const", self.B)
        self.last_activ = nn.Softmax(dim=1)

    def algebraic_reconstruction(self, fpre, fpred):
        recon1 = torch.einsum("iQ,NQ->Ni", self.A_const, fpre)
        recon2 = torch.einsum("iQ,NQ->Ni", self.B_const, fpred)
        return recon1 + recon2

    def forward(self, x):
        out = self.data_handler.normalize(x)
        out = self.model(out)
        out = self.data_handler.denormalize(out)
        out = self.algebraic_reconstruction(x, out)
        return out
