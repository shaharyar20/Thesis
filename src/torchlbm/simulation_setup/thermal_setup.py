#!/usr/bin/env python3

# torchlbm modules
from torchlbm.setup_definitions.setup_types import SetupSet, SetupTag
from torchlbm.setup_definitions.type_converters import (
    StringConverter,
    FloatConverter,
    IntConverter,
    ListConverter,
    BoolConverter
)

def boundary_conditions():
    return ["Periodic", "Wall", "ZeroGradient", "Outlet", "Nothing"]

class ThermalSetup(SetupSet):
    """The ThermalSetup gives all information that are used in the thermal block of an Alpaca inputfile.

    Args:
        SetupSet (SetupSet): The base class for all setups.
    """

    def __init__(self) -> None:
        """The initializer for the ThermalSetup. The ThermalSetup contains all informations regarding the simulated domain (size...)."""

        settings = [
            SetupTag("Active", False, False, BoolConverter()),
            SetupTag("ThermalConductivity", 0.5/3, False, FloatConverter(0.0, None, False)),
            SetupTag("BounceBackTemperature", 0.0, False, FloatConverter(None, None, False)),
            SetupTag("Cp", 2.0, False, FloatConverter(0.0, None, False)),
            SetupTag("Cv", 1.0, False, FloatConverter(0.0, None, False)),
            SetupTag("Viscosity", 0.025, False, FloatConverter(0.0, None, False)),
            SetupTag("ShiftedVelocityX", 0.01, False, FloatConverter(None, None, False)),
            SetupTag("ShiftedVelocityY", 0.0, False, FloatConverter(None, None, False)),
            SetupSet(
                "BoundaryConditions",
                [
                    SetupSet(
                        "East",
                        [
                            SetupTag(
                                "Type",
                                "Periodic",
                                False,
                                StringConverter(boundary_conditions(), False),
                            ),
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
                        ],
                    ),
                    SetupSet(
                        "West",
                        [
                            SetupTag(
                                "Type",
                                "Periodic",
                                False,
                                StringConverter(boundary_conditions(), False),
                            ),
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
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
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
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
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
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
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
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
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
                        ],
                    ),
                ],
            ),
        ]

        super().__init__("Thermal", settings)