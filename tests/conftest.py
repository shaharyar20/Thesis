from tokenize import _all_string_prefixes
import pytest

import torch
from torchlbm.core.lattices.lattice import Lattice
from torchlbm.state import TorchlbmState
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.logger import Logger


class RandomInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        U_physical = 10 * torch.randn_like(X)
        V_physical = 10 * torch.randn_like(X)
        W_physical = 10 * torch.randn_like(X)
        return [
            U_physical,
            V_physical,
            W_physical,
        ]

    def get_initial_density(self, X, Y, Z):
        return 10 * torch.randn_like(X)


class ZeroOneInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        return [
            torch.zeros_like(X),
            torch.zeros_like(X),
            torch.zeros_like(X),
        ]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X)


@pytest.fixture(params=[Lattice])
def torchlbm_random_state(request) -> TorchlbmState:
    """Run a test for all lattices and lattices-of-vector;
    return a grid with 3^D sample distribution functions alongside the lattice.
    """
    simulation_setup = TorchlbmSetup("Random")
    simulation_setup["Domain"]["Dimension"].value = "3D"
    simulation_setup["Domain"]["NodeSize"].value = 1.0
    simulation_setup["Domain"]["CellsPerNode"].value = 50
    simulation_setup["Domain"]["NumHaloCells"].value = 1
    simulation_setup["Domain"]["NodeRatio"].value = [1, 1, 1]
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

    simulation_setup["Output"]["Active"].value = False

    simulation_setup["Physics"]["MachNumber"].value = 0.1
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 1.0
    simulation_setup["Physics"]["KinematicViscosityPu"].value = 0.01
    simulation_setup["Physics"]["EndTime"].value = 3.0
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 1.0
    simulation_setup["Physics"]["Precision"].value = "Double"

    simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
    simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
    simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

    check_torchlbm_setup(simulation_setup)

    initial_condition = RandomInitialCondition(simulation_setup)
    state = TorchlbmState(simulation_setup, initial_condition, Logger())
    return state


@pytest.fixture(params=[Lattice])
def torchlbm_zero_one_state(request) -> TorchlbmState:
    """Run a test for all lattices and lattices-of-vector;
    return a grid with 3^D sample distribution functions alongside the lattice.
    """
    simulation_setup = TorchlbmSetup("ZeroOne")
    simulation_setup["Domain"]["Dimension"].value = "3D"
    simulation_setup["Domain"]["NodeSize"].value = 1.0
    simulation_setup["Domain"]["CellsPerNode"].value = 50
    simulation_setup["Domain"]["NumHaloCells"].value = 1
    simulation_setup["Domain"]["NodeRatio"].value = [1, 1, 1]
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

    simulation_setup["Output"]["Active"].value = False

    simulation_setup["Physics"]["MachNumber"].value = 0.1
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 1.0
    simulation_setup["Physics"]["KinematicViscosityPu"].value = 0.01
    simulation_setup["Physics"]["EndTime"].value = 3.0
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 1.0
    simulation_setup["Physics"]["Precision"].value = "Double"

    simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
    simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
    simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

    check_torchlbm_setup(simulation_setup)

    initial_condition = ZeroOneInitialCondition(simulation_setup)
    state = TorchlbmState(simulation_setup, initial_condition, Logger())
    return state
