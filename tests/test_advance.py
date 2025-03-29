import pytest

import torch

from torchlbm.state import TorchlbmState
from torchlbm.module_factory.equilibrium_calculation_module_factory import get_equilibrium_calculation_module
from torchlbm.module_factory.collision_module_factory import get_collision_module
from torchlbm.module_factory.macroscopic_quantity_calculation_module_factory import get_macroscopic_quantity_calculation_module

from torchlbm.core.advance import AdvanceModule

from torchlbm.module_factory.collision_module_factory import get_collision_module
from torchlbm.module_factory.streaming_module_factory import get_streaming_module
from torchlbm.module_factory.macroscopic_quantity_calculation_module_factory import get_macroscopic_quantity_calculation_module
from torchlbm.module_factory.boundary_condition_factory import (
    get_periodic_boundary_module,
    get_wall_boundary_module,
    get_outlet_boundary_module,
    get_zero_gradient_boundary_module,
    get_bounce_back_boundary_module,
    get_inlet_boundary_module,
    get_time_space_dependent_wall_boundary_module,
)
from torchlbm.module_factory.multiphase_module_factory import get_multiphase_module
from torchlbm.module_factory.forcing_module_factory import get_forcing_module
from torchlbm.module_factory.equilibrium_calculation_module_factory import get_equilibrium_calculation_module
from torchlbm.module_factory.carreau_yasuda_module_factory import get_carreau_yasuda_module


def check_advance_call(state: TorchlbmState):
    advance_module = AdvanceModule(
        unit_converter=state.unit_converter,
        collision_module=get_collision_module(state),
        streaming_module=get_streaming_module(state),
        macroscopic_module=get_macroscopic_quantity_calculation_module(state),
        equilibrium_module=get_equilibrium_calculation_module(state, state.torchlbm_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["Type"].value),
        classical_equilibrium_module=get_equilibrium_calculation_module(state, "Classical"),
        multiphase_module=get_multiphase_module(state),
        periodic_module=get_periodic_boundary_module(state),
        wall_module=get_wall_boundary_module(state),
        time_space_dependent_wall_module=get_time_space_dependent_wall_boundary_module(state),
        inlet_module=get_inlet_boundary_module(state),
        outlet_module=get_outlet_boundary_module(state),
        zero_gradient_module=get_zero_gradient_boundary_module(state),
        bounce_back_module=get_bounce_back_boundary_module(state),
        forcing_module=get_forcing_module(state),
        is_forcing_active=state.torchlbm_setup["Physics"]["VolumeForces"]["Active"].value,
        carreau_yasuda_module=get_carreau_yasuda_module(state),
        is_carreau_yasuda_active=state.torchlbm_setup["Physics"]["CarreauYasuda"]["Active"].value,
    )
    new_node_data = advance_module(state.node_data)
    assert new_node_data.moments.density == pytest.approx(torch.ones_like(new_node_data.moments.density))


def test_advance(torchlbm_zero_one_state):
    state = torchlbm_zero_one_state

    check_advance_call(state)

    state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "TRT"
    check_advance_call(state)

    state.torchlbm_setup["Domain"]["Dimension"].value = "2D"
    # state.torchlbm_setup["Algorithm"]["Operators"]["Collision"].value = "MRT"
    check_advance_call(state)

    # state.torchlbm_setup["Domain"]["CellsPerNode"].value = 10
    # state.torchlbm_setup["Domain"]["NodeRatio"].value = [3, 2, 1]
    # check_advance_call(state)
