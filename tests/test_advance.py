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
from torchlbm.module_factory.boundary_condition_factory import get_boundary_condition_modules
from torchlbm.module_factory.multiphase_module_factory import get_multiphase_module
from torchlbm.module_factory.forcing_module_factory import get_forcing_module
from torchlbm.module_factory.equilibrium_calculation_module_factory import get_equilibrium_calculation_module
from torchlbm.module_factory.non_newtonian_module_factory import get_non_newtonian_module


def check_advance_call(state: TorchlbmState):
    advance_module = AdvanceModule(
        unit_converter=state.unit_converter,
        collision_module=get_collision_module(state),
        streaming_module=get_streaming_module(state),
        macroscopic_module=get_macroscopic_quantity_calculation_module(state),
        equilibrium_module=get_equilibrium_calculation_module(state, state.torchlbm_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["Type"].value),
        classical_equilibrium_module=get_equilibrium_calculation_module(state, "Classical"),
        multiphase_module=get_multiphase_module(state),
        boundary_condition_modules=get_boundary_condition_modules(state),
        forcing_module=get_forcing_module(state),
        is_forcing_active=state.torchlbm_setup["Physics"]["VolumeForces"]["Active"].value,
        non_newtonian_module=get_non_newtonian_module(state),
        is_non_newtonian_active=state.torchlbm_setup["Physics"]["NonNewtonian"]["Active"].value,
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
