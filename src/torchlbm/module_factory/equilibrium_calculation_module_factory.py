from torchlbm.state import TorchlbmState
from torchlbm.core.collision_models.linear_bgk import EquilibriumCalculationModule
from torchlbm.core.collision_models.thermal_linear_bgk import ThermalEquilibriumCalculationModule
from torchlbm.core.collision_models.compressible_eq import CompressibleEquilibriumCalculationModule


def get_equilibrium_calculation_module(state: TorchlbmState) -> EquilibriumCalculationModule:
    """Factory function that created an object for the equilibrium calculation.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        EquilibriumCalculationModule: The created object.
    """
    # return EquilibriumCalculationModule(
    # return ThermalEquilibriumCalculationModule(
    return CompressibleEquilibriumCalculationModule(
        lattice_velocities=state.lattice.lattice_velocities(),
        lattice_weights=state.lattice.lattice_weights(),
        shifted_velx=state.torchlbm_setup["Thermal"]["ShiftedVelocityX"].value,
        shifted_vely=state.torchlbm_setup["Thermal"]["ShiftedVelocityY"].value
    )
