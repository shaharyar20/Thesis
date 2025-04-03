import torch

from torchlbm.state import TorchlbmState
from torchlbm.non_newtonian.viscosity_models.carreau_yasuda import CarreauYasudaModule

def get_non_newtonian_module(state: TorchlbmState) -> CarreauYasudaModule:
    """Factory function to create the object of the streaming module.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        StreamingModule: The created object.
    """

    # number_of_discrete_velocities = state.lattice.number_of_discrete_velocities()
    # lattice_velocities = torch.tensor(state.lattice.lattice_velocities())
    

    if state.torchlbm_setup["Physics"]["NonNewtonian"]["Type"].value == "CarreauYasuda":
        return CarreauYasudaModule(
            unit_converter=state.unit_converter,
            lattice_velocities=state.lattice.lattice_velocities(),
            number_of_discrete_velocities=state.lattice.number_of_discrete_velocities(),
            viscosity_inf=state.torchlbm_setup["Physics"]["NonNewtonian"]["CarreauYasuda"]["ViscosityInf"].value,
            viscosity_0=state.torchlbm_setup["Physics"]["KinematicViscosityPu"].value,
            lam=state.torchlbm_setup["Physics"]["NonNewtonian"]["CarreauYasuda"]["lambda"].value,
            n=state.torchlbm_setup["Physics"]["NonNewtonian"]["CarreauYasuda"]["n"].value,
            a=state.torchlbm_setup["Physics"]["NonNewtonian"]["CarreauYasuda"]["a"].value,
        )
