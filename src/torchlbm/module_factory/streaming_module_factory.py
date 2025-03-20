from torchlbm.state import TorchlbmState
from torchlbm.core.streaming.streaming import StreamingModule


def get_streaming_module(state: TorchlbmState) -> StreamingModule:
    """Factory function to create the object of the streaming module.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        StreamingModule: The created object.
    """
    internal_cells_list = state.torchlbm_setup["Domain"]["InternalCells"].value
    num_halos = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    total_cell_list = [num_internal + 2 * num_halos for num_internal in internal_cells_list]
    return StreamingModule(
        number_of_discrete_velocities=state.lattice.number_of_discrete_velocities(),
        lattice_velocities=state.lattice.lattice_velocities(),
        num_total_cells=total_cell_list,
    )
