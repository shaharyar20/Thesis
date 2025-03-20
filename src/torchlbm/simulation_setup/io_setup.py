#!/usr/bin/env python3

# torchlbm modules
from torchlbm.setup_definitions.setup_types import SetupSet, SetupTag
from torchlbm.setup_definitions.type_converters import BoolConverter, IntConverter, ListConverter, FloatConverter, StringConverter


def output_type():
    return ["VTK", "Picture", "PyTorch"]


output_quantity_settings = [
    SetupTag("Active", True, False, BoolConverter()),
    SetupTag("ValueBounds", [0.0, 1.0], False, ListConverter(FloatConverter(), 2, 2)),
    SetupTag("UseValueBounds", False, False, BoolConverter()),
    SetupTag("Types", "VTK", False, ListConverter(StringConverter(output_type()))),
]


class IoSetup(SetupSet):
    """The IoSetup specifies the IO behavior during the simulation

    Args:
        SetupSet (SetupSet): The base class for all setups.
    """

    def __init__(self) -> None:
        """The initializer for the IoSetup. The IoSetup contains all informations regarding the IO of the simulation."""

        velocity_output_quantity_settings = output_quantity_settings.copy()
        velocity_output_quantity_settings.append(SetupTag("ColorbarLabel", r"$|u| [m/s]$", False, StringConverter()))
        density_output_quantity_settings = output_quantity_settings.copy()
        density_output_quantity_settings.append(SetupTag("ColorbarLabel", r"$\rho [kg/m^3]$", False, StringConverter()))
        bounce_back_mask_output_quantity_settings = output_quantity_settings.copy()
        bounce_back_mask_output_quantity_settings.append(SetupTag("ColorbarLabel", "Bounce Back Mask", False, StringConverter()))

        settings = [
            SetupTag("Active", True, True, BoolConverter()),
            SetupTag("ProfilingActive", False, False, BoolConverter()),
            SetupTag("ModulusArtifactsActive", False, False, BoolConverter()),
            SetupTag("PrintTimingInformation", True, False, BoolConverter()),
            SetupTag("OutputTimeInterval", 1.0, True, FloatConverter(0.0, None, False)),
            SetupTag("OutputEveryStep", False, False, BoolConverter()),
            SetupSet(
                "Velocity",
                velocity_output_quantity_settings,
            ),
            SetupSet(
                "Density",
                density_output_quantity_settings,
            ),
            SetupSet(
                "BounceBackMask",
                bounce_back_mask_output_quantity_settings,
            ),
        ]

        super().__init__("Output", settings)
