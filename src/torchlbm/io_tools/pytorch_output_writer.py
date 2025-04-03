import torch
import numpy as np
from typing import List

from torchlbm.state import TorchlbmState
from torchlbm.unit_converter import UnitConverter

import matplotlib
import matplotlib.pyplot as plt
from torchlbm.node_data import NodeData


def get_single_node_pytorch_data(state: TorchlbmState):
    """Returns a matplotlib figure object that contains information about the flow field.

    Args:
        state (TorchlbmState): The state of the simulation that contains all relevant information, including
                           the fields.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object that can be written to a file.
    """

    node: NodeData = state.node_data
    unit_converter: UnitConverter = state.unit_converter
    internal_cells_list = state.torchlbm_setup["Domain"]["InternalCells"].value
    num_halos = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value
    delta_x = state.lattice_distance

    start = [
        num_halos,
        num_halos if dimension != 1 else 0,
        num_halos if dimension == 3 else 0,
    ]

    end = [
        -num_halos,
        -num_halos if dimension != 1 else 1,
        -num_halos if dimension == 3 else 1,
    ]

    bounce_back_mask = node.bounce_back_mask[start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()

    density = node.moments.density[start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    density = unit_converter.convert_density_to_physical_units(density)
    density = torch.where(bounce_back_mask > 0, density.mean(), density)

    velocity = node.moments.velocity[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    velocity = unit_converter.convert_velocity_to_physical_units(velocity)
    velocity = torch.where(bounce_back_mask.unsqueeze(0) > 0, 0.0, velocity)

    forcing_velocity = node.moments.forcing_velocity[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    forcing_velocity = unit_converter.convert_velocity_to_physical_units(forcing_velocity)
    forcing_velocity = forcing_velocity

    volume_force_field = node.moments.volume_force_field[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    volume_force_field = unit_converter.convert_acceleration_to_physical_units(volume_force_field)
    volume_force_field = volume_force_field

    result = {}

    if state.torchlbm_setup["Output"]["Velocity"]["Active"].value:
        if "PyTorch" in state.torchlbm_setup["Output"]["Velocity"]["Types"].value:
            name = "velocity"
            result[name] = velocity

    if state.torchlbm_setup["Output"]["Density"]["Active"].value:
        if "PyTorch" in state.torchlbm_setup["Output"]["Density"]["Types"].value:
            name = "density"
            result[name] = density

    if state.torchlbm_setup["Output"]["BounceBackMask"]["Active"].value:
        if "PyTorch" in state.torchlbm_setup["Output"]["BounceBackMask"]["Types"].value:
            name = "bounce_back_mask"
            result[name] = bounce_back_mask

    if state.torchlbm_setup["Output"]["KinematicViscosity"]["Active"].value and state.torchlbm_setup["Physics"]["NonNewtonian"]["Active"].value:
        if "PyTorch" in state.torchlbm_setup["Output"]["KinematicViscosity"]["Types"].value:
            name = "kinematic_viscosity"
            relaxation_omega = node.relaxation_omega[start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
            relaxation_time = 1.0 / relaxation_omega
            kinematic_viscosity = unit_converter.convert_relaxation_time_to_kinematic_viscosity_physical_units(relaxation_time)
            kinematic_viscosity = torch.where(bounce_back_mask > 0, 0.0, kinematic_viscosity)
            kinematic_viscosity = kinematic_viscosity.detach().numpy()
            result[name] = kinematic_viscosity
    return result
