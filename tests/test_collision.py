import pytest

import torch

from torchlbm.module_factory.equilibrium_calculation_module_factory import get_equilibrium_calculation_module
from torchlbm.module_factory.collision_module_factory import get_collision_module
from torchlbm.module_factory.macroscopic_quantity_calculation_module_factory import get_macroscopic_quantity_calculation_module


def test_mass_conservation_collision(torchlbm_random_state):
    state = torchlbm_random_state
    equilibrium_module = torch.jit.script(get_equilibrium_calculation_module(state))
    collision_module = torch.jit.script(get_collision_module(state))
    macroscopic_module = torch.jit.script(get_macroscopic_quantity_calculation_module(state))

    state.node_data = macroscopic_module(state.node_data)
    density_prev = state.node_data.moments.density
    state.node_data = equilibrium_module(state.node_data)

    discrete_velocities_post_collision = collision_module(state.node_data)
    state.node_data = macroscopic_module(state.node_data)
    assert torch.einsum("ijk->", density_prev) == pytest.approx(torch.einsum("ijk->", state.node_data.moments.density))
