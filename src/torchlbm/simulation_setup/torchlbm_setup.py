from torchlbm.setup_definitions.setup_types import SetupSet
from torchlbm.simulation_setup.algorithm_setup import AlgorithmSetup
from torchlbm.exceptions import TorchlbmError
from .physical_setup import PhysicalSetup
from .domain_setup import DomainSetup
from .initial_condition_setup import InitialConditionSetup
from .io_setup import IoSetup
from .lattice_setup import LatticeSetup
from torchlbm.thermal.simulation_setup.thermal_setup import ThermalSetup
import torch


class TorchlbmSetup(SetupSet):
    """The Torchlbm setup used for a simulation. It represents all options and settings necessary to run the simulation.

    Args:
        SetupSet ([type]): The base class for all setups.
    """

    def __init__(self, name: str = "TorchlbmSimulation") -> None:
        """The initializer for the TorchlbmSetup.

        Args:
            name (str): The name of the setup. Can be chosen to describe the testcase which it refers to.
        """
        if name is None:
            name = "TorchlbmSetup"

        settings = [
            DomainSetup(),
            InitialConditionSetup(),
            IoSetup(),
            PhysicalSetup(),
            LatticeSetup(),
            AlgorithmSetup(),
            ThermalSetup(),
        ]

        super().__init__(name, settings)


def check_torchlbm_setup(setup: TorchlbmSetup):
    cells_per_node = setup["Domain"]["CellsPerNode"].value
    num_halos_cells = setup["Domain"]["NumHaloCells"].value
    total_cells_list = [element * cells_per_node + 2 * num_halos_cells for element in setup["Domain"]["NodeRatio"].value]
    internal_cells_list = [element * cells_per_node for element in setup["Domain"]["NodeRatio"].value]
    node_ratio_list = setup["Domain"]["NodeRatio"].value

    if setup["InitialCondition"]["ReadInitialConditionFromYaml"].value and setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value:
        raise TorchlbmError(
            "Either initial condition can be read from YAML or from PyTorch file. Both at the same time is not possible. Deactivate one of them or both!"
        )

    setup["Domain"]["DimensionInteger"].value = 3

    if setup["Domain"]["Dimension"].value == "1D":
        setup["Domain"]["DimensionInteger"].value = 1

        node_ratio_list[1] = 0
        node_ratio_list[2] = 0
        total_cells_list[1] = 1
        total_cells_list[2] = 1
        internal_cells_list[1] = 1
        internal_cells_list[2] = 1

    if setup["Domain"]["Dimension"].value == "2D":
        setup["Domain"]["DimensionInteger"].value = 2

        node_ratio_list[2] = 0
        total_cells_list[2] = 1
        internal_cells_list[2] = 1

    setup["Domain"]["TotalCells"].value = total_cells_list
    setup["Domain"]["InternalCells"].value = internal_cells_list
    setup["Domain"]["NodeRatio"].value = node_ratio_list

    if setup["Physics"]["Precision"].value == "Single":
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float64)
