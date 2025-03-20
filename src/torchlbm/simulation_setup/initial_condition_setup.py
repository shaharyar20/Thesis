#!/usr/bin/env python3

# torchlbm modules
from torchlbm.setup_definitions.setup_types import SetupSet, SetupTag
from torchlbm.setup_definitions.type_converters import (
    StringConverter,
    BoolConverter,
    PathConverter,
)
from pathlib import Path


class InitialConditionSetup(SetupSet):
    """The InitialConditionSetup gives all information that are used in the domain block of an Alpaca inputfile.

    Args:
        SetupSet (SetupSet): The base class for all setups.
    """

    def __init__(self) -> None:
        """The initializer for the InitialConditionSetup. The InitialConditionSetup contains all informations regarding the simulated domain (size...)."""

        settings = [
            SetupTag("ReadInitialConditionFromYaml", False, False, BoolConverter()),
            SetupTag("ReadInitialFieldsFromPyTorchFiles", False, False, BoolConverter()),
            SetupTag("Density", "1.0", False, StringConverter()),
            SetupSet(
                "Velocity",
                [
                    SetupTag("x", "1.0", False, StringConverter()),
                    SetupTag("y", "1.0", False, StringConverter()),
                    SetupTag("z", "1.0", False, StringConverter()),
                ],
            ),
            SetupTag("BounceBackMask", "1.0", False, StringConverter()),
            SetupSet(
                "PyTorchFields",
                [
                    SetupTag("Velocity", __file__, False, PathConverter()),
                    SetupTag("Density", __file__, False, PathConverter()),
                    SetupTag("BounceBackMask", __file__, False, PathConverter()),
                ],
            ),
        ]

        super().__init__("InitialCondition", settings)
