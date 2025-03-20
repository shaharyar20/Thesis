#!/usr/bin/env python3
# Python modules
from typing import Any
from copy import deepcopy

# torchlbm modules
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions.setup_types import SetupTag
from torchlbm.setup_definitions.type_converters import TypeConverter, StringConverter


class SlurmVariable(SetupTag):
    """The SlurmVariable class represents a SetupTag that can be used for slurm script creation.

    The SlurmVariable is a specialization of the SetupTag class for slurm scripts. Beside basic functionality of the SetupTag
    a special formatting function is implemented that provides proper variable definition for the usage in Slurm scripts.

    Args:
        SetupTag (SetupTag): This class is a SetupTag.
    """

    def __init__(
        self,
        name: str,
        default_value: Any,
        type_converter: TypeConverter,
        variable_identifier: str,
    ) -> None:
        """Constructor.

        Args:
            name (str): The name of the variable for internal referencing.
            default_value (Any): The default value of the variable.
            type_converter (TypeConverter): The type converter to be used.
            variable_identifier (str): The identifier in the c++ file, where the variable can be found.

        Raises:
            SetupError: If the variable identifier or namespaces are not of type string.
        """
        # Check that the variable identifier is a string
        if not isinstance(variable_identifier, str):
            raise SetupError("Slurm variable identifier must be of type 'str'.") from None
        if not isinstance(type_converter, StringConverter):
            raise SetupError("Slurm variable type converter must be a string converter.") from None
        # Call the base class
        super().__init__(name, default_value, False, type_converter)
        # Assign the member variables
        self._variable_identifier = deepcopy(variable_identifier.strip())

    def formatted_tag(self) -> str:
        """Gives the slurm tag and its identifier to be used in Slurm scripts.

        Returns:
            str: The formatted Slurm variable with identifier.
        """
        return f"{self._variable_identifier}{self.value}"
