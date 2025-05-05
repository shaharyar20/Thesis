#!/usr/bin/env python3

# torchlbm modules
from torchlbm.setup_definitions.setup_types import SetupSet, SetupTag
from torchlbm.setup_definitions.type_converters import (
    StringConverter,
    BoolConverter,
    PathConverter,
    FloatConverter,
)
from pathlib import Path


class ThermalSetup(SetupSet):
    """The InitialConditionSetup gives all information that are used in the domain block of an Alpaca inputfile.

    Args:
        SetupSet (SetupSet): The base class for all setups.
    """

    def __init__(self) -> None:
        """The initializer for the InitialConditionSetup. The InitialConditionSetup contains all informations regarding the simulated domain (size...)."""

        settings = [
            SetupTag("Active", False, False, BoolConverter()),
            SetupTag("HeatConductivity", 0.1, False, FloatConverter(0.0, None, False)),
            SetupSet(
                "BoundaryConditions",
                [
                    SetupSet(
                        "East",
                        [
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
                        ],
                    ),
                    SetupSet(
                        "West",
                        [
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
                        ],
                    ),
                    SetupSet(
                        "North",
                        [
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
                        ],
                    ),
                    SetupSet(
                        "South",
                        [
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
                        ],
                    ),
                    SetupSet(
                        "Top",
                        [
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
                        ],
                    ),
                    SetupSet(
                        "Bottom",
                        [
                            SetupTag("WallTemperature", 0.0, False, FloatConverter(None, None, False)),
                        ],
                    ),
                ],
            ),
        ]

        super().__init__("Thermal", settings)
