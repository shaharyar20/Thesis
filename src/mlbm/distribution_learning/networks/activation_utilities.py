from typing import List
import os
from pathlib import Path
from enum import Enum

import torch
from torch import nn
from torch.nn.functional import mse_loss, l1_loss

import torchlbm.standalone_operations.file_operations as file_o
from torchlbm.exceptions import TorchlbmError


class ActivationOption(Enum):
    none = 0
    relu = 1
    leaky_relu = 2
    gelu = 3
    sigmoid = 4
    softmax = 5


def get_activation_function(activation_option: ActivationOption):
    if activation_option == ActivationOption.none:
        return None, None
    elif activation_option == ActivationOption.relu:
        activation_function = nn.ReLU
        activation_params = {"inplace": True}
        return activation_function, activation_params
    elif activation_option == ActivationOption.leaky_relu:
        activation_function = nn.LeakyReLU
        activation_params = {"inplace": True}
        return activation_function, activation_params
    elif activation_option == ActivationOption.gelu:
        activation_function = nn.GELU
        activation_params = {}
        return activation_function, activation_params
    elif activation_option == ActivationOption.sigmoid:
        activation_function = nn.Sigmoid
        activation_params = {}
        return activation_function, activation_params
    elif activation_option == ActivationOption.softmax:
        activation_function = nn.Softmax
        activation_params = {"dim": 1}
        return activation_function, activation_params
    else:
        raise TorchlbmError("The selected loss funtion is not yet implemented!")
