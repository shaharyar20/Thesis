#!/usr/bin/env python3

# torchlbm modules
from torchlbm.setup_definitions.setup_types import SetupSet, SetupTag
from torchlbm.setup_definitions.type_converters import (
    StringConverter,
    FloatConverter,
    IntConverter,
    ListConverter,
)


def boundary_conditions():
    return ["Periodic", "Wall", "ZeroGradient", "Outlet", "Nothing", "Inlet", "TimeSpaceDependentWall"]


def dimensions():
    return ["1D", "2D", "3D"]

def bounce_back_implementations():
    return ["Fullway", "Halfway", "Interpolated"]


class DomainSetup(SetupSet):
    """The DomainSetup gives all information that are used in the domain block of an Alpaca inputfile.

    Args:
        SetupSet (SetupSet): The base class for all setups.
    """

    def __init__(self) -> None:
        """The initializer for the DomainSetup. The DomainSetup contains all informations regarding the simulated domain (size...)."""

        settings = [
            SetupTag("Dimension", "3D", True, StringConverter(dimensions(), False)),
            SetupTag("DimensionInteger", 2, False, IntConverter(1, 3, False)),
            SetupTag("NodeSize", 1.0, True, FloatConverter(0.0, None, False)),
            SetupTag("CellsPerNode", 16, True, IntConverter(1, None, False)),
            SetupTag("NumHaloCells", 1, True, IntConverter(0, None, False)),
            SetupTag("NodeRatio", [0, 0, 0], False, ListConverter(IntConverter(), 3, 3)),
            SetupTag("TotalCells", [0, 0, 0], False, ListConverter(IntConverter(), 3, 3)),
            SetupTag("InternalCells", [0, 0, 0], False, ListConverter(IntConverter(), 3, 3)),
            SetupSet(
                "BoundaryConditions",
                [
                    SetupSet(
                        "East",
                        [
                            SetupTag(
                                "Type",
                                "Periodic",
                                True,
                                StringConverter(boundary_conditions(), False),
                            ),
                            SetupTag("WallVelocity", [0.0, 0.0, 0.0], False, ListConverter(FloatConverter(), 3, 3)),
                            SetupTag("OutletDensity", 1.0, False, FloatConverter()),
                        ],
                    ),
                    SetupSet(
                        "West",
                        [
                            SetupTag(
                                "Type",
                                "Periodic",
                                True,
                                StringConverter(boundary_conditions(), False),
                            ),
                            SetupTag("WallVelocity", [0.0, 0.0, 0.0], False, ListConverter(FloatConverter(), 3, 3)),
                            SetupTag("OutletDensity", 1.0, False, FloatConverter()),
                        ],
                    ),
                    SetupSet(
                        "North",
                        [
                            SetupTag(
                                "Type",
                                "Periodic",
                                False,
                                StringConverter(boundary_conditions(), False),
                            ),
                            SetupTag("WallVelocity", [0.0, 0.0, 0.0], False, ListConverter(FloatConverter(), 3, 3)),
                            SetupTag("OutletDensity", 1.0, False, FloatConverter()),
                        ],
                    ),
                    SetupSet(
                        "South",
                        [
                            SetupTag(
                                "Type",
                                "Periodic",
                                False,
                                StringConverter(boundary_conditions(), False),
                            ),
                            SetupTag("WallVelocity", [0.0, 0.0, 0.0], False, ListConverter(FloatConverter(), 3, 3)),
                            SetupTag("OutletDensity", 1.0, False, FloatConverter()),
                        ],
                    ),
                    SetupSet(
                        "Top",
                        [
                            SetupTag(
                                "Type",
                                "Periodic",
                                False,
                                StringConverter(boundary_conditions(), False),
                            ),
                            SetupTag("WallVelocity", [0.0, 0.0, 0.0], False, ListConverter(FloatConverter(), 3, 3)),
                            SetupTag("OutletDensity", 1.0, False, FloatConverter()),
                        ],
                    ),
                    SetupSet(
                        "Bottom",
                        [
                            SetupTag(
                                "Type",
                                "Periodic",
                                False,
                                StringConverter(boundary_conditions(), False),
                            ),
                            SetupTag("WallVelocity", [0.0, 0.0, 0.0], False, ListConverter(FloatConverter(), 3, 3)),
                            SetupTag("OutletDensity", 1.0, False, FloatConverter()),
                        ],
                    ),
                    SetupTag("BounceBackType", "Fullway", False, StringConverter(bounce_back_implementations(), False)),
                ],
            ),
        ]

        super().__init__("Domain", settings)
