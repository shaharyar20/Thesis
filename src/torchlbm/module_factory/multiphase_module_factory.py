from typing import List
from torchlbm.state import TorchlbmState
from torchlbm.multiphase.shan_chen_pseudopotential_multiphase import ShanChenPseudopotentialMultiphaseModule


def get_multiphase_module(state: TorchlbmState) -> ShanChenPseudopotentialMultiphaseModule:
    """Factory function that returns the module to perform mutliphase simulations.

    Args:
        state (TorchlbmState): The state of the simulation that contains all relevant information about the simulations setup.

    Returns:
        ShanChenPseudopotentialMultiphaseModule: The class object that is returned by the factory function.
    """
    num_halo_cells: int = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    internal_cells: List[int] = state.torchlbm_setup["Domain"]["InternalCells"].value
    tau = 1.0 / state.torchlbm_setup["Physics"]["RelaxationOmega"].value
    dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value

    access_indices: List[List[int]] = [
        [0, num_halo_cells, num_halo_cells + internal_cells[0], 2 * num_halo_cells + internal_cells[0]],
        [0, num_halo_cells, num_halo_cells + internal_cells[1], 2 * num_halo_cells + internal_cells[1]],
        [0, num_halo_cells, num_halo_cells + internal_cells[2], 2 * num_halo_cells + internal_cells[2]],
    ]
    return ShanChenPseudopotentialMultiphaseModule(
        lattice_velocities=state.lattice.lattice_velocities(),
        lattice_weights=state.lattice.lattice_weights(),
        access_indices=access_indices,
        tau=tau,
        dimension=dimension,
    )
