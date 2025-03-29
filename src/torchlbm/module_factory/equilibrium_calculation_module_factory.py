from torchlbm.state import TorchlbmState
from torchlbm.core.equilibrium.equilibrium import EquilibriumCalculationModule

import torch


def get_equilibrium_calculation_module(state: TorchlbmState) -> EquilibriumCalculationModule:
    """Factory function that created an object for the equilibrium calculation.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        EquilibriumCalculationModule: The created object.
    """
    return EquilibriumCalculationModule(
        lattice_velocities=state.lattice.lattice_velocities(),
        lattice_weights=state.lattice.lattice_weights(),
    )
