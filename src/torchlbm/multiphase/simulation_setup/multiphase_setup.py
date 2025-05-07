#!/usr/bin/env python3

# torchlbm modules
from torchlbm.setup_definitions.setup_types import SetupSet, SetupTag
from torchlbm.setup_definitions.type_converters import (
    StringConverter,
    BoolConverter,
    FloatConverter,
)
from pathlib import Path


def EOS():
    return ["ShanChen", "CarnahanStarling"]


class MultiphaseSetup(SetupSet):
    """The PhysicalSetup specifies the dimensionless numbers describing the underlying physical problem.

    Args:
        SetupSet (SetupSet): The base class for all setups.
    """

    def __init__(self) -> None:
        """The initializer for the PhysicalSetup. The PhysicalSetup contains all informations regarding dimensionless numbers."""

        settings = [
            SetupTag("Active", False, False, BoolConverter()),
            SetupTag("EOS", "CarnahanStarling", False, StringConverter(EOS(), False)),
            SetupSet(
                "ShanChenEOS",
                [
                    SetupTag("InteractionStrength", -4.7, False, FloatConverter(None, -4.0, False)),
                    SetupTag("ReferenceDensity", 1.0, False, FloatConverter(0.0, None, False)),
                ]
            ),
            SetupSet(
                "CarnahanStarlingEOS",
                [
                    SetupTag("ReducedTemperature", 0.9, False, FloatConverter(0.0, 1.0, False)),
                ]
            ),
        ]

        super().__init__("Multiphase", settings)
