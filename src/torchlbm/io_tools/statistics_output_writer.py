import torch
import numpy as np
from typing import List

from torchlbm.state import TorchlbmState
from torchlbm.unit_converter import UnitConverter
from mlbm.distribution_learning.utils import plot_hist_vels_as_subplot

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchlbm.node_data import NodeData
import matplotlib.gridspec as gridspec


def get_statistics_figure(state: TorchlbmState):
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

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14))

    # print(pre_collision.shape)

    ax1.boxplot(node.distributions.old_population.flatten(1, -1).transpose(1, 0).numpy(), labels=[f"V{i}" for i in range(0, 9)])
    ax1.set_title("Box Plot of Discrete Velocities")
    # ax1.set_xlabel('Discrete Velocities')
    ax1.set_ylabel("Values")

    ax2.boxplot(node.distributions.neq_population.flatten(1, -1).transpose(1, 0).numpy(), labels=[f"V{i}" for i in range(0, 9)])
    ax2.set_title("Box Plot of nonequilibrium part of Discrete Velocities")
    # ax2.set_xlabel('Discrete Velocities')
    ax2.set_ylabel("Values")

    # ax3.boxplot(node.moments.density.flatten().numpy(), labels=['Density'])
    ax3.hist(node.moments.density.flatten().numpy())
    ax3.set_title("Box Plot of Density")
    # ax3.set_xlabel('Density')
    ax3.set_ylabel("Values")

    # ax4.boxplot(torch.norm(node.moments.velocity, dim=0).flatten().numpy(), labels=['Velocity Magnitude'])
    ax4.hist(torch.norm(node.moments.velocity, dim=0).flatten().numpy())
    ax4.set_title("Box Plot of Velocity Magnitude")
    # ax4.set_xlabel('Velocity Magnitude')
    ax4.set_ylabel("Values")

    plt.tight_layout()

    return fig


def get_boxplot_figure(population):
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.3)
    plot_hist_vels_as_subplot(fig, gs, population.flatten(1, -1).transpose(1, 0))
    return fig
