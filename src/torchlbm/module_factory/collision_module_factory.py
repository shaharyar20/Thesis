from torchlbm.state import TorchlbmState
from torchlbm.core.collision_models.collision import CollisionModule
from torchlbm.core.collision_models.trt_collision import TRTCollisionModule
from torchlbm.core.collision_models.mrt_collision import MRTCollisionModule
from torchlbm.core.collision_models.entropic_mrt_collision import EntropicMRTCollisionModule


def get_collision_module(state: TorchlbmState) -> CollisionModule:
    """Factory function for the collision module object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        CollisionModule: The created object.
    """

    if state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "SRT":
        return CollisionModule()
    elif state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "TRT":
        return TRTCollisionModule(
            my_opposite_lattice_indices=state.lattice.opposite_lattice_indices(),
            magic_parameter=1 / 12.0,
        )
    elif state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "MRT":
        return MRTCollisionModule(
            relaxation_omega=state.node_data.relaxation_omega.relaxation_omega,
            my_population_to_momentum_transform=state.lattice.population_to_momentum_transform(),
            my_momentum_to_population_transform=state.lattice.momentum_to_population_transform(),
        )
    elif state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "EntropicMRT":
        return EntropicMRTCollisionModule(
            lattice_velocities=state.lattice.lattice_velocities(),
            lattice_weights=state.lattice.lattice_weights(),
            relaxation_omega=state.torchlbm_setup["Physics"]["RelaxationOmega"].value,
            model=state.torchlbm_setup["Domain"]["DimensionInteger"].value,
        )
