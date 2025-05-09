import torch

from torchlbm.state import TorchlbmState
from torchlbm.core.forcing.shan_chen_forcing import ShanChenForcingModule
from torchlbm.core.forcing.guo_forcing import GuoForcingModule
from torchlbm.exceptions import TorchlbmError


def get_forcing_module(state: TorchlbmState) -> ShanChenForcingModule:
    """Factory function to create a forcing module object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Raises:
        TorchlbmError: Performs sanity checks (force vector hast right dimension) and throws errors in case the check does not pass.

    Returns:
        ShanChenForcingModule: The created object
    """
    force_vector = state.unit_converter.convert_acceleration_to_lattice_units(
        torch.tensor(state.torchlbm_setup["Physics"]["VolumeForces"]["ForceVector"].value)
    )
    if not isinstance(force_vector, torch.Tensor):
        raise TorchlbmError("Please ensure that the force-vector is of type torch.Tensor.")
    if force_vector.shape != torch.Size([3]):
        raise TorchlbmError("Please ensure that the force-vector has dimensionality 3.")
    force_vector = force_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    if state.torchlbm_setup["Physics"]["VolumeForces"]["Type"].value == "ShanChen":
        return ShanChenForcingModule(
            force_vector=force_vector,
        )
    elif state.torchlbm_setup["Physics"]["VolumeForces"]["Type"].value == "Guo":
        return GuoForcingModule(
            force_vector=force_vector,
            lattice_velocities=state.lattice.lattice_velocities(),
            lattice_weights=state.lattice.lattice_weights(),
        )
