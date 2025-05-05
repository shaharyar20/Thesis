import torch
import numpy as np
from typing import List

from torchlbm.state import TorchlbmState
from torchlbm.unit_converter import UnitConverter

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchlbm.node_data import NodeData


def get_single_node_pyplot_data(state: TorchlbmState):
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
    density = torch.where(bounce_back_mask > 0, 1.0, density)
    density = density.detach().numpy()

    velocity = node.moments.velocity[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    velocity = unit_converter.convert_velocity_to_physical_units(velocity)
    velocity = torch.where(bounce_back_mask.unsqueeze(0) > 0, 0.0, velocity)
    velocity = velocity.detach().numpy()

    # forcing_velocity = node.moments.forcing_velocity[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    # forcing_velocity = unit_converter.convert_velocity_to_physical_units(forcing_velocity)
    # forcing_velocity = forcing_velocity.detach().numpy()

    # volume_force_field = node.moments.volume_force_field[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    # volume_force_field = unit_converter.convert_acceleration_to_physical_units(volume_force_field)
    # volume_force_field = volume_force_field.detach().numpy()

    num_cells_x = density.shape[0]
    num_cells_y = density.shape[1]

    result = {}

    name = "Velocity"
    if state.torchlbm_setup["Output"][name]["Active"].value:
        if "Picture" in state.torchlbm_setup["Output"][name]["Types"].value:
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(num_cells_x / 100 + 2.5, num_cells_y / 100 + 0.5))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            velocity_magnitude = np.linalg.norm(velocity, axis=0)
            velocity_magnitude = np.ma.masked_where(bounce_back_mask == 1, velocity_magnitude)
            velocity_array = np.transpose(np.squeeze(velocity_magnitude, axis=-1), axes=[1, 0])
            im = ax.imshow(
                velocity_array,
                origin="lower",
                cmap=plt.cm.viridis,  # plt.cm.Spectral,
                interpolation="none",
                extent=[0.0, num_cells_x * delta_x, 0.0, num_cells_y * delta_x],
                vmin=state.torchlbm_setup["Output"][name]["ValueBounds"].value[0] if state.torchlbm_setup["Output"][name]["UseValueBounds"].value else None,
                vmax=state.torchlbm_setup["Output"][name]["ValueBounds"].value[1] if state.torchlbm_setup["Output"][name]["UseValueBounds"].value else None,
            )
            ax.set_title(name)
            ax.set_aspect("equal", "box")
            fmt = lambda x, pos: "{:.2e}".format(x)
            fig.colorbar(im, cax=cax, orientation="vertical", label=state.torchlbm_setup["Output"][name]["ColorbarLabel"].value, format=FuncFormatter(fmt))
            result[name.lower()] = fig

    name = "Density"
    if state.torchlbm_setup["Output"][name]["Active"].value:
        if "Picture" in state.torchlbm_setup["Output"][name]["Types"].value:
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(num_cells_x / 100 + 2.5, num_cells_y / 100 + 0.5))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            density_array = np.transpose(np.squeeze(density, axis=-1), axes=[1, 0])
            im = ax.imshow(
                density_array,
                origin="lower",
                cmap=plt.cm.viridis,  # plt.cm.Spectral,
                interpolation="none",
                extent=[0.0, num_cells_x * delta_x, 0.0, num_cells_y * delta_x],
                vmin=state.torchlbm_setup["Output"][name]["ValueBounds"].value[0] if state.torchlbm_setup["Output"][name]["UseValueBounds"].value else None,
                vmax=state.torchlbm_setup["Output"][name]["ValueBounds"].value[1] if state.torchlbm_setup["Output"][name]["UseValueBounds"].value else None,
            )
            ax.set_title(name)
            ax.set_aspect("equal", "box")
            fmt = lambda x, pos: "{:.2e}".format(x)
            fig.colorbar(im, cax=cax, orientation="vertical", label=state.torchlbm_setup["Output"][name]["ColorbarLabel"].value, format=FuncFormatter(fmt))
            result[name.lower()] = fig

    name = "Temperature"
    if state.torchlbm_setup["Thermal"]["Active"].value and state.torchlbm_setup["Output"][name]["Active"].value:
        if "Picture" in state.torchlbm_setup["Output"][name]["Types"].value:
            temperature = node.moments.temperature[start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
            # temperature = unit_converter.convert_temperature_to_physical_units(temperature)
            temperature = torch.where(bounce_back_mask > 0, 0.0, temperature)
            temperature = temperature.detach().numpy()
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(num_cells_x / 100 + 2.5, num_cells_y / 100 + 0.5))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            temperature_array = np.transpose(np.squeeze(temperature, axis=-1), axes=[1, 0])
            im = ax.imshow(
                temperature_array,
                origin="lower",
                cmap=plt.cm.viridis,  # plt.cm.Spectral,
                interpolation="none",
                extent=[0.0, num_cells_x * delta_x, 0.0, num_cells_y * delta_x],
                vmin=state.torchlbm_setup["Output"][name]["ValueBounds"].value[0] if state.torchlbm_setup["Output"][name]["UseValueBounds"].value else None,
                vmax=state.torchlbm_setup["Output"][name]["ValueBounds"].value[1] if state.torchlbm_setup["Output"][name]["UseValueBounds"].value else None,
            )
            ax.set_title(name)
            ax.set_aspect("equal", "box")
            fmt = lambda x, pos: "{:.2e}".format(x)
            fig.colorbar(im, cax=cax, orientation="vertical", label=state.torchlbm_setup["Output"][name]["ColorbarLabel"].value, format=FuncFormatter(fmt))
            result[name.lower()] = fig

    name = "KinematicViscosity"
    if state.torchlbm_setup["Physics"]["NonNewtonian"]["Active"].value and state.torchlbm_setup["Output"][name]["Active"].value:
        if "Picture" in state.torchlbm_setup["Output"][name]["Types"].value:
            relaxation_omega = node.relaxation_omega[start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
            relaxation_time = 1.0 / relaxation_omega
            kinematic_viscosity = unit_converter.convert_relaxation_time_to_kinematic_viscosity_physical_units(relaxation_time)
            kinematic_viscosity = torch.where(bounce_back_mask > 0, 0.0, kinematic_viscosity)
            kinematic_viscosity = kinematic_viscosity.detach().numpy()

            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(num_cells_x / 100 + 2.5, num_cells_y / 100 + 0.5))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            kinematic_viscosity_array = np.transpose(np.squeeze(kinematic_viscosity, axis=-1), axes=[1, 0])
            im = ax.imshow(
                kinematic_viscosity_array,
                origin="lower",
                cmap=plt.cm.viridis,  # plt.cm.Spectral,
                interpolation="none",
                extent=[0.0, num_cells_x * delta_x, 0.0, num_cells_y * delta_x],
                vmin=state.torchlbm_setup["Output"][name]["ValueBounds"].value[0] if state.torchlbm_setup["Output"][name]["UseValueBounds"].value else None,
                vmax=state.torchlbm_setup["Output"][name]["ValueBounds"].value[1] if state.torchlbm_setup["Output"][name]["UseValueBounds"].value else None,
            )
            ax.set_title(name)
            ax.set_aspect("equal", "box")
            fmt = lambda x, pos: "{:.2e}".format(x)
            fig.colorbar(im, cax=cax, orientation="vertical", label=state.torchlbm_setup["Output"][name]["ColorbarLabel"].value, format=FuncFormatter(fmt))
            result[name.lower()] = fig

    name = "BounceBackMask"
    if state.torchlbm_setup["Output"][name]["Active"].value:
        if "Picture" in state.torchlbm_setup["Output"][name]["Types"].value:
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(num_cells_x / 100 + 2.5, num_cells_y / 100 + 0.5))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            mask_array = np.transpose(np.squeeze(bounce_back_mask, axis=-1), axes=[1, 0])
            im = ax.imshow(
                mask_array,
                origin="lower",
                cmap=plt.cm.viridis,  # plt.cm.Spectral,
                interpolation="none",
                extent=[0.0, num_cells_x * delta_x, 0.0, num_cells_y * delta_x],
                vmin=state.torchlbm_setup["Output"][name]["ValueBounds"].value[0] if state.torchlbm_setup["Output"][name]["UseValueBounds"].value else None,
                vmax=state.torchlbm_setup["Output"][name]["ValueBounds"].value[1] if state.torchlbm_setup["Output"][name]["UseValueBounds"].value else None,
            )
            ax.set_title(name)
            ax.set_aspect("equal", "box")
            fmt = lambda x, pos: "{:.2e}".format(x)
            fig.colorbar(im, cax=cax, orientation="vertical", label=state.torchlbm_setup["Output"][name]["ColorbarLabel"].value, format=FuncFormatter(fmt))
            result[name.lower()] = fig

    return result
