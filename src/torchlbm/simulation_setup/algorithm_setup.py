from torchlbm.setup_definitions.setup_types import SetupSet, SetupTag
from torchlbm.setup_definitions.type_converters import StringConverter, PathConverter


def operator_implementations():
    return ["Classical", "MLP"]


def collision_implementations():
    return ["SRT", "TRT", "MRT", "NN", "GNN", "EntropicMRT"]


def equilibrium_implementations():
    return ["Classical", "MLP"]


class AlgorithmSetup(SetupSet):
    """The AlgorithmSetup gives all information that are used to adjust the (Q)LBM algorithm.

    Args:
        SetupSet (SetupSet): The base class for all setups.
    """

    def __init__(self) -> None:
        """The initializer for the AlgorithmSetup. The AlgorithmSetup contains all informations to adjust the (Q)LBM algorithm."""

        settings = [
            SetupSet(
                "Operators",
                [
                    SetupTag("MacroscopicCalculation", "Classical", False, StringConverter(operator_implementations())),
                    SetupSet(
                        "EquilibriumCalculation",
                        [
                            SetupTag("Type", "Classical", False, StringConverter(equilibrium_implementations())),
                            SetupTag("ModelPath", None, False, PathConverter()),
                        ],
                    ),
                    SetupSet(
                        "Collision",
                        [
                            SetupTag("Type", "SRT", False, StringConverter(collision_implementations())),
                            SetupTag("ModelPath", None, False, PathConverter()),
                        ],
                    ),
                    SetupTag("Streaming", "Classical", False, StringConverter(operator_implementations())),
                ],
            )
        ]

        super().__init__("Algorithm", settings)
