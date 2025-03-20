#!/usr/bin/env python3

# torchlbm modules
from torchlbm.core.lattices.lattice_dictionaries import OneDimensionalLattices, ThreeDimensionalLattices, TwoDimensionalLattices
from torchlbm.setup_definitions.setup_types import SetupSet, SetupTag
from torchlbm.setup_definitions.type_converters import StringConverter


class LatticeSetup(SetupSet):
    """The DomainSetup gives all information that are used in the domain block of an Alpaca inputfile.

    Args:
        SetupSet (SetupSet): The base class for all setups.
    """

    def __init__(self) -> None:
        """The initializer for the DomainSetup. The DomainSetup contains all informations regarding the simulated domain (size...)."""

        settings = [
            SetupSet(
                "NSE",
                [
                    SetupTag(
                        "1D",
                        "D1Q2",
                        False,
                        StringConverter(list(OneDimensionalLattices.keys())),
                    ),
                    SetupTag(
                        "2D",
                        "D2Q9",
                        False,
                        StringConverter(list(TwoDimensionalLattices.keys())),
                    ),
                    SetupTag(
                        "3D",
                        "D3Q19",
                        False,
                        StringConverter(list(ThreeDimensionalLattices.keys())),
                    ),
                ],
            ),
        ]

        super().__init__("Lattice", settings)
