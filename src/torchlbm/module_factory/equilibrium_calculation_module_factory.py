from torchlbm.state import TorchlbmState
from torchlbm.core.collision_models.linear_bgk import EquilibriumCalculationModule
from mlbm.distribution_learning.modules.ml_equilibrium_module import MLEquilibriumModule

from mlbm.distribution_learning.networks.lightning_module import SymmetryNetwork

import torch


def get_equilibrium_calculation_module(state: TorchlbmState, equilibrium_model_type: str = "Classical") -> EquilibriumCalculationModule:
    """Factory function that created an object for the equilibrium calculation.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        EquilibriumCalculationModule: The created object.
    """
    if equilibrium_model_type == "Classical":
        return EquilibriumCalculationModule(
            lattice_velocities=state.lattice.lattice_velocities(),
            lattice_weights=state.lattice.lattice_weights(),
        )
    if equilibrium_model_type == "MLP":
        # model = SymmetryNetwork.load_from_checkpoint(state.torchlbm_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["ModelPath"].value)
        scripted_module = torch.jit.load(state.torchlbm_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["ModelPath"].value)
        return MLEquilibriumModule(
            model=scripted_module,
            lattice_velocities=state.lattice.lattice_velocities(),
            lattice_weights=state.lattice.lattice_weights(),
        )
