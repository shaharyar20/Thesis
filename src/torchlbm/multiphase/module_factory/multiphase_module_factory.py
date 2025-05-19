from typing import List

from torchlbm.state import TorchlbmState
from torchlbm.multiphase.pseudopotential.shan_chen_pseudopotential_calculation import ShanChenPseudopotentialCalculationModule
from torchlbm.multiphase.pseudopotential.carnahan_starling_pseudopotential_calculation import CarnahanSterlingPseudopotentialCalculationModule
from torchlbm.multiphase.force_calculation.force_calculation_multiphase import ForceCalculationMultiphaseModule
from torchlbm.multiphase.phase_change.carnahan_starling_thermal_calculation import CarnahanSterlingThermalCalculationModule

def get_pseudopotential_module(state: TorchlbmState) -> ShanChenPseudopotentialCalculationModule:
    """Factory function that returns the module to perform mutliphase simulations.

    Args:
        state (QlbmState): The state of the simulation that contains all relevant information about the simulations setup.

    Returns:
        ShanChenPseudopotentialMultiphaseModule: The class object that is returned by the factory function.
    """

    if state.torchlbm_setup["Multiphase"]["EOS"].value == "ShanChen":
        return ShanChenPseudopotentialCalculationModule(
            ref_density=state.torchlbm_setup["Multiphase"]["ShanChenEOS"]["ReferenceDensity"].value,
        )

    elif state.torchlbm_setup["Multiphase"]["EOS"].value == "CarnahanStarling":
        return CarnahanSterlingPseudopotentialCalculationModule(
            reduced_temperature=state.torchlbm_setup["Multiphase"]["CarnahanStarlingEOS"]["ReducedTemperature"].value,
        )


def get_multiphase_forcing_module(state: TorchlbmState) -> ForceCalculationMultiphaseModule:
    """Factory function that returns the module to perform mutliphase simulations.

    Args:
        state (QlbmState): The state of the simulation that contains all relevant information about the simulations setup.

    Returns:
        ShanChenPseudopotentialMultiphaseModule: The class object that is returned by the factory function.
    """
    num_halo_cells: int = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    internal_cells: List[int] = state.torchlbm_setup["Domain"]["InternalCells"].value
    dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value

    access_indices: List[List[int]] = [
        [0, num_halo_cells, num_halo_cells + internal_cells[0], 2 * num_halo_cells + internal_cells[0]],
        [0, num_halo_cells, num_halo_cells + internal_cells[1], 2 * num_halo_cells + internal_cells[1]],
        [0, num_halo_cells, num_halo_cells + internal_cells[2], 2 * num_halo_cells + internal_cells[2]],
    ]

    if state.torchlbm_setup["Multiphase"]["EOS"].value == "ShanChen":
        interaction_strength = state.torchlbm_setup["Multiphase"]["ShanChenEOS"]["InteractionStrength"].value

    else:
        interaction_strength = -1.0

    return ForceCalculationMultiphaseModule(
        lattice_velocities=state.lattice.lattice_velocities(),
        lattice_weights=state.lattice.lattice_weights(),
        access_indices=access_indices,
        dimension=dimension,
        interaction_strength=interaction_strength,
        n_discrete_velocities=state.lattice.number_of_discrete_velocities(),
    )

def get_phase_change_module(state: TorchlbmState) -> CarnahanSterlingThermalCalculationModule:
    """Factory function that returns the module to perform mutliphase simulations.

    Args:
        state (QlbmState): The state of the simulation that contains all relevant information about the simulations setup.

    Returns:
        ShanChenPseudopotentialMultiphaseModule: The class object that is returned by the factory function.
    """

    return CarnahanSterlingThermalCalculationModule(
        Cv=1.0,
        lattice_weights=state.lattice.lattice_weights(),
    )