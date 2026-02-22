from typing import List

from torchlbm.state import TorchlbmState
from torchlbm.multiphase.pseudopotential.shan_chen_pseudopotential_calculation import ShanChenPseudopotentialCalculationModule
from torchlbm.multiphase.pseudopotential.carnahan_starling_pseudopotential_calculation import CarnahanSterlingPseudopotentialCalculationModule
from torchlbm.multiphase.force_calculation.force_calculation_multiphase import ForceCalculationMultiphaseModule
from torchlbm.multiphase.phase_change.carnahan_starling_thermal_calculation import CarnahanSterlingThermalCalculationModule
from torchlbm.multiphase.pseudopotential.peng_robinson_pseudopotential_calculation import PengRobinsonPseudopotentialCalculationModule

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
            solid_density=state.torchlbm_setup["Multiphase"]["SolidDensity"].value,
            a=state.torchlbm_setup["Multiphase"]["CarnahanStarlingEOS"]["a"].value,
            b=state.torchlbm_setup["Multiphase"]["CarnahanStarlingEOS"]["b"].value,
            R=state.torchlbm_setup["Multiphase"]["CarnahanStarlingEOS"]["R"].value,
            solid_temperature=state.torchlbm_setup["Multiphase"]["SolidTemperature"].value if state.torchlbm_setup["Thermal"]["Active"].value == True else None,
        )
    
    elif state.torchlbm_setup["Multiphase"]["EOS"].value == "PengRobinson":
        return PengRobinsonPseudopotentialCalculationModule(
            reduced_temperature=state.torchlbm_setup["Multiphase"]["PengRobinsonEOS"]["ReducedTemperature"].value,
            solid_density=state.torchlbm_setup["Multiphase"]["SolidDensity"].value,
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

    gravity_value = state.unit_converter.convert_acceleration_to_lattice_units(
        state.torchlbm_setup["Multiphase"]["Gravity"]["Active"].value
    )

    return ForceCalculationMultiphaseModule(
        lattice_velocities=state.lattice.lattice_velocities(),
        lattice_weights=state.lattice.lattice_weights(),
        access_indices=access_indices,
        dimension=dimension,
        interaction_strength=interaction_strength,
        n_discrete_velocities=state.lattice.number_of_discrete_velocities(),
        is_gravity_active=state.torchlbm_setup["Multiphase"]["Gravity"]["Active"].value,
        gravity_value=gravity_value,
        free_parameter_indices=state.lattice.free_parameter_indices() if state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "MRT" else None,
        viscosity_indices=state.lattice.viscosity_indices() if state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "MRT" else None,
        free_parameters=state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["MRT"]["FreeParameters"].value if state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "MRT" else None,
        sigma=state.torchlbm_setup["Multiphase"]["MRTTuningParamater"].value if state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "MRT" else None,
    )

def get_phase_change_module(state: TorchlbmState) -> CarnahanSterlingThermalCalculationModule:
    """Factory function that returns the module to perform mutliphase simulations.

    Args:
        state (QlbmState): The state of the simulation that contains all relevant information about the simulations setup.

    Returns:
        ShanChenPseudopotentialMultiphaseModule: The class object that is returned by the factory function.
    """

    return CarnahanSterlingThermalCalculationModule(
        Cv=5.0,
        lattice_weights=state.lattice.lattice_weights(),
        lattice_velocities=state.lattice.lattice_velocities(),
    )