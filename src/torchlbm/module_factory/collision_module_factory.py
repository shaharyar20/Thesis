from torchlbm.state import TorchlbmState
from torchlbm.core.collision_models.collision import CollisionModule
from torchlbm.core.collision_models.trt_collision import TRTCollisionModule
from torchlbm.core.collision_models.mrt_collision import MRTCollisionModule
from torchlbm.core.collision_models.nn_collision import NNCollisionModule
from torchlbm.core.collision_models.gnn_collision import GNNCollisionModule
from torchlbm.core.collision_models.thermal_collision import ThermalCollisionModule
from torchlbm.core.collision_models.compressible_collision import CompressibleCollisionModule

# from mlbm.distribution_learning.networks.cons_sym_networks import ExactConsSymNN, LossBasedConsSymNN
# from mlbm.distribution_learning.networks.graph_networks import FirstGNN, SimpleGIN
# from mlbm.distribution_learning.networks.cons_graph_networks import ConsGIN, ConsGCN
# from mlbm.distribution_learning.networks.equivariant_networks import EGNN, CollisionEGNN


def get_collision_module(state: TorchlbmState) -> CollisionModule:
    """Factory function for the collision module object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        CollisionModule: The created object.
    """
    if state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "SRT":
        # return CollisionModule(state.torchlbm_setup["Physics"]["RelaxationOmega"].value)
        # return ThermalCollisionModule(state.torchlbm_setup["Physics"]["RelaxationOmega"].value, state.relaxation_omega_temp)
        return CompressibleCollisionModule(
            viscosity=state.torchlbm_setup["Thermal"]["Viscosity"].value,
            cp=state.torchlbm_setup["Thermal"]["Cp"].value,
            thermal_conductivity=state.torchlbm_setup["Thermal"]["ThermalConductivity"].value,
            lattice_velocities=state.lattice.lattice_velocities(),
            shifted_velx=state.torchlbm_setup["Thermal"]["ShiftedVelocityX"].value,
            shifted_vely=state.torchlbm_setup["Thermal"]["ShiftedVelocityY"].value
        )
    elif state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "TRT":
        return TRTCollisionModule(
            relaxation_omega=state.torchlbm_setup["Physics"]["RelaxationOmega"].value,
            my_opposite_lattice_indices=state.lattice.opposite_lattice_indices(),
            magic_parameter=1 / 12.0,
        )
    elif state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "MRT":
        return MRTCollisionModule(
            relaxation_omega=state.torchlbm_setup["Physics"]["RelaxationOmega"].value,
            my_population_to_momentum_transform=state.lattice.population_to_momentum_transform(),
            my_momentum_to_population_transform=state.lattice.momentum_to_population_transform(),
        )
    elif state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "NN":
        return NNCollisionModule(
            model=ExactConsSymNN,
            model_path=state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value,
        )
    elif state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["Type"].value == "GNN":
        return GNNCollisionModule(
            model=EGNN,
            model_path=state.torchlbm_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value,
            num_graphs=state.node_data.distributions.old_population.shape[1]
            * state.node_data.distributions.old_population.shape[2]
            * state.node_data.distributions.old_population.shape[3],
            lattice_velocities=state.lattice.lattice_velocities(),
        )
