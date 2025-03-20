import torch

from torchlbm.state import TorchlbmState
from torchlbm.core.viscosity_models.carreau_yasuda import CarreauYasudaModule


def get_carreau_yasuda_module(state: TorchlbmState) -> CarreauYasudaModule:
    """Factory function to create the object of the streaming module.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        StreamingModule: The created object.
    """

    number_of_discrete_velocities = state.lattice.number_of_discrete_velocities()
    lattice_velocities = torch.tensor(state.lattice.lattice_velocities())
    kron = torch.empty([number_of_discrete_velocities, 3, 3])
    for i in range(number_of_discrete_velocities):
        kron[i] = torch.kron(lattice_velocities[:, i].unsqueeze(1), lattice_velocities[:, i].unsqueeze(0))

    return CarreauYasudaModule(
        unit_converter=state.unit_converter,
        kron=kron,
        viscosity_inf=state.torchlbm_setup["Physics"]["CarreauYasuda"]["viscosity_inf"].value,
        viscosity_0=state.torchlbm_setup["Physics"]["CarreauYasuda"]["viscosity_0"].value,
        lam=state.torchlbm_setup["Physics"]["CarreauYasuda"]["lam"].value,
        n=state.torchlbm_setup["Physics"]["CarreauYasuda"]["n"].value,
        a=state.torchlbm_setup["Physics"]["CarreauYasuda"]["a"].value,
    )
