from torchlbm.state import TorchlbmState
from torchlbm.core.macroscopic_quantities.collection import MacroscopicQuantityCalculationModule
from torchlbm.core.macroscopic_quantities.thermal_collection import ThermalMacroscopicQuantityCalculationModule  
from torchlbm.core.macroscopic_quantities.compressible_collection import CompressibleMacroscopicQuantityCalculationModule


def get_macroscopic_quantitiy_calculation_module(state: TorchlbmState) -> MacroscopicQuantityCalculationModule:
    """Factory function to create a macroscopic quantitiy calculation module.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        MacroscopicQuantityCalculationModule: The created object.
    """
    # return MacroscopicQuantityCalculationModule(
    # return ThermalMacroscopicQuantityCalculationModule(
    return CompressibleMacroscopicQuantityCalculationModule(
        lattice_velocities=state.lattice.lattice_velocities(), 
        cv=state.torchlbm_setup["Thermal"]["Cv"].value,
        shifted_velx=state.torchlbm_setup["Thermal"]["ShiftedVelocityX"].value,
        shifted_vely=state.torchlbm_setup["Thermal"]["ShiftedVelocityY"].value
    )
