#!/usr/bin/env python3

# torchlbm modules
from torchlbm.setup_definitions.setup_types import SetupSet, SetupTag
from torchlbm.setup_definitions.type_converters.range_converter import FloatConverter
from torchlbm.setup_definitions.type_converters import StringConverter, BoolConverter
from torchlbm.setup_definitions.type_converters import ListConverter


def precision():
    return ["Single", "Double"]


def equation_type():
    return ["NSE", "AdvectionDiffusion"]


def volume_force_type():
    return ["ShanChen"]

def non_newtonian_type():
    return ["CarreauYasuda"]


class PhysicalSetup(SetupSet):
    """The PhysicalSetup specifies the dimensionless numbers describing the underlying physical problem.

    Args:
        SetupSet (SetupSet): The base class for all setups.
    """

    def __init__(self) -> None:
        """The initializer for the PhysicalSetup. The PhysicalSetup contains all informations regarding dimensionless numbers."""

        settings = [
            SetupTag("EquationType", "NSE", False, StringConverter(equation_type(), False)),
            SetupTag("MachNumber", 0.05, True, FloatConverter(0.0, None, False)),
            SetupTag("EndTime", 1.0, True, FloatConverter(0.0, None, False)),
            SetupTag("CharacteristicVelocityPu", 1.0, True, FloatConverter(0.0, None, False)),
            SetupTag("CharacteristicVelocityLu", 1.0, True, FloatConverter(0.0, None, False)),
            SetupTag("KinematicViscosityPu", 1.0, True, FloatConverter(0.0, None, False)),
            SetupTag("RelaxationOmega", 1.0, True, FloatConverter(0.0, None, False)),
            SetupSet(
                "VolumeForces",
                [
                    SetupTag("Active", False, False, BoolConverter()),
                    SetupTag("Type", "ShanChen", False, StringConverter(volume_force_type(), False)),
                    SetupTag("ForceVector", [0.0, 0.0, 0.0], False, ListConverter(FloatConverter(), 3, 3)),
                ],
            ),
            SetupSet(
                "NonNewtonian",
                [
                    SetupTag("Active", False, False, BoolConverter()),
                    SetupTag("Type", "CarreauYasuda", False, StringConverter(non_newtonian_type(), False)),
                    SetupSet(
                        "CarreauYasuda",
                        [
                            SetupTag("ViscosityInf", 1.0, False, FloatConverter()),
                            SetupTag("lambda", 1.0, False, FloatConverter()),
                            SetupTag("n", 1.0, False, FloatConverter()),
                            SetupTag("a", 1.0, False, FloatConverter()),
                        ],
                    ),
                ],
            ),
            SetupTag("Precision", "Double", True, StringConverter(precision(), False)),
        ]

        super().__init__("Physics", settings)
