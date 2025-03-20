from torchlbm.state import TorchlbmState
from torchlbm.core.macroscopic_quantities.collection import MacroscopicQuantityCalculationModule


def get_macroscopic_quantitiy_calculation_module(state: TorchlbmState) -> MacroscopicQuantityCalculationModule:
    """Factory function to create a macroscopic quantitiy calculation module.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        MacroscopicQuantityCalculationModule: The created object.
    """
    return MacroscopicQuantityCalculationModule(
        lattice_velocities=state.lattice.lattice_velocities(),
    )
