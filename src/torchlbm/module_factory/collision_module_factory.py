from torchlbm.state import TorchlbmState
from torchlbm.core.collision_models.srt_collision import SRTCollisionModule
from torchlbm.core.collision_models.trt_collision import TRTCollisionModule
from torchlbm.core.collision_models.mrt_collision import MRTCollisionModule
from torchlbm.core.collision_models.entropic_mrt_collision import EntropicMRTCollisionModule


def get_collision_module(state: TorchlbmState) -> SRTCollisionModule:
    """Factory function for the collision module object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        SRTCollisionModule: The created object.
    """

    if state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "SRT":
        return SRTCollisionModule()
    elif state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "TRT":
        return TRTCollisionModule(
            my_opposite_lattice_indices=state.lattice.opposite_lattice_indices(),
            magic_parameter=1 / 12.0,
        )
    elif state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "MRT":
        return MRTCollisionModule(
            free_parameters=state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["MRT"]["FreeParameters"].value,
            my_population_to_momentum_transform=state.lattice.population_to_momentum_transform(),
            my_momentum_to_population_transform=state.lattice.momentum_to_population_transform(),
            number_of_discrete_velocities=state.lattice.number_of_discrete_velocities(),
            free_parameter_indices=state.lattice.free_parameter_indices(),
            viscosity_indices=state.lattice.viscosity_indices(),
        )
    elif state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "EntropicMRT":
        return EntropicMRTCollisionModule(
            lattice_velocities=state.lattice.lattice_velocities(),
            lattice_weights=state.lattice.lattice_weights(),
            model=state.torchlbm_setup["Domain"]["DimensionInteger"].value,
        )
