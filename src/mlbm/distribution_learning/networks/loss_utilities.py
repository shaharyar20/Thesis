from typing import List
import os
from pathlib import Path
from enum import Enum

import torch
from torch import nn
from torch.nn.functional import mse_loss, l1_loss

import torchlbm.standalone_operations.file_operations as file_o
from torchlbm.exceptions import TorchlbmError


def rmsre(y_true, y_pred, eps=1e-14):
    return torch.mean(torch.square((y_pred - y_true) / (y_true + eps)))


def msre(y_true, y_pred, eps=1e-14):
    return torch.mean(torch.abs((y_pred - y_true) / (y_true + eps)))
    # return torch.mean(torch.square((y_pred - y_true) / (y_true + eps)))


def mass_loss(y_true, y_pred):
    true_mass = torch.sum(y_true, dim=1)
    predicted_mass = torch.sum(y_pred, dim=1)
    accuracy = torch.mean(torch.abs((predicted_mass - true_mass) / true_mass))
    return accuracy


def per_dimension_loss(y_true, y_pred):
    accuracy = torch.abs((y_pred - y_true) / y_true)
    return torch.mean(accuracy, dim=0)


def momentum_loss(y_true, y_pred, discrete_velocities):
    momentum_true = torch.einsum("ij,kj->ik", y_true, discrete_velocities)
    momentum_pred = torch.einsum("ij,kj->ik", y_pred, discrete_velocities)
    accuracy = torch.mean(torch.abs((momentum_pred - momentum_true) / (momentum_true + 1.0e-15)), dim=0)
    return accuracy


class LossOption(Enum):
    none = 0
    l1 = 1
    mse = 2
    rmsre = 3
    msre = 4
    mass = 5
    momentum = 6
    per_dimension = 7


def get_loss_function(loss_option: LossOption):
    if loss_option == LossOption.none:
        return None
    elif loss_option == LossOption.l1:
        return nn.L1Loss()
    elif loss_option == LossOption.mse:
        return nn.MSELoss()
    elif loss_option == LossOption.rmsre:
        return rmsre
    elif loss_option == LossOption.msre:
        return msre
    elif loss_option == LossOption.mass:
        return mass_loss
    elif loss_option == LossOption.momentum:
        return momentum_loss
    elif loss_option == LossOption.per_dimension:
        return per_dimension_loss
    else:
        raise TorchlbmError("The selected loss funtion is not yet implemented!")
