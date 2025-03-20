#!/usr/bin/env python3
# Python modules
from typing import List, Tuple, Union, Any
from copy import deepcopy
import re

# torchlbm modules
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions.setup_types import SetupTag
from torchlbm.setup_definitions.type_converters import TypeConverter


class CxxVariable(SetupTag):
    """The CxxVariable represents a specialization for the SetupTag to be used in C++ files.

    It allows the reading and replacement of variables in a c++ file. Beside the basic
    functionality of the SetupTag it has special read and replace function that carry out the appropriate handling.

    Args:
        SetupTag (SetupTag): This class is a SetupTag.

    Note:
        This classes uses a variable identifier to replace the content for this variable. Only variables should be used that occur once in that file with
        a single assignment, e.g.
            float variable_a = 5.0;
        In general, the variable is replaced at all positions.
    """

    def __init__(
        self,
        name: str,
        default_value: Any,
        type_converter: TypeConverter,
        variable_identifier: str,
        namespace_list: List[str] = [],
    ) -> None:
        """Constructor.

        Args:
            name (str): The name of the variable for internal referencing.
            default_value (Any): The default value of the variable.
            type_converter (TypeConverter): The type converter to be used.
            variable_identifier (str): The identifier in the c++ file, where the variable can be found.
            namespace_list (List[str]): The list of namespace prefixes that should be set in front of the variable. No :: required.

        Raises:
            SetupError: If the variable identifier or namespaces are not of type string.
        """
        # Check that the variable identifier is a string and all values in namespace list is also string
        namespaces = namespace_list if isinstance(namespace_list, list) else [namespace_list]
        if not isinstance(variable_identifier, str) or any([not isinstance(entry, str) for entry in namespaces]):
            raise SetupError("Cxx variable identifier and namespaces must be of type 'str'.") from None
        # Call the base class
        super().__init__(name, default_value, True, type_converter)
        # Assign the member variables
        self._namespace_prefix = f"{'::'.join([entry.strip() for entry in namespaces])}::" if namespaces else None
        self._variable_identifier = deepcopy(variable_identifier.strip())

    def _convert_string(self, value: str) -> str:
        """Replace a python string to compliant C++ string.

        Args:
            sequence (str): The string to be converted.

        Returns:
            str: The c++ string.
        """
        return f'"{value}"' if self._namespace_prefix is None else f"{self._namespace_prefix}{value}"

    def _convert_bool(self, value: bool) -> str:
        """Replace a bool to compliant C++ string.

        Args:
            sequence (bool): The bool to be converted.

        Returns:
            str: The c++ string.
        """
        return "true" if value else "false"

    def _convert_sequence(self, sequence: Union[List, Tuple]) -> str:
        """Replace a sequence to compliant C++ string.

        Args:
            sequence (Union[List, Tuple]): The sequence to be converted.

        Returns:
            str: The c++ string.
        """
        list_values = [
            self._convert_string(value) if isinstance(value, str) else self._convert_bool(value) if isinstance(value, bool) else f"{value}"
            for value in sequence
        ]
        return f"{{ {', '.join(list_values)} }}"

    def replace(self, content_to_replace: str) -> str:
        """Replaces the variable in the provided content.

        Args:
            content_to_replace (str): The file content where the variable should be replaced.

        Returns:
            str: The modified file content.
        """
        # Check if the content contains the variable otherwise log a warning
        if self._variable_identifier not in content_to_replace:
            raise SetupError(f"Cannot find {self._variable_identifier} in C++ file content.") from None
        # Convert the tag to the cxx compliant values
        if isinstance(self.value, str):
            value_to_set = self._convert_string(self.value)
        else:
            if self._namespace_prefix is not None:
                raise SetupError("Cannot replace non-string to cxx compliant type with prefixes. Name prefix must be None.") from None
            if isinstance(self.value, bool):
                value_to_set = self._convert_bool(self.value)
            elif isinstance(self.value, (list, tuple)):
                value_to_set = self._convert_sequence(self.value)
            else:
                value_to_set = f"{self.value}"
        # The leading and trailing white space in the sub command around the identifier is ESSENTIAL to indicate a new variable name.
        # Otherwise variable names with prefixes or suffixes are modified, too.
        return re.sub(
            f" ({self._variable_identifier}) ( *)=( *)(.+?);",
            r" \g<1> \g<2>=\g<3>" + f"{value_to_set}" + ";",
            content_to_replace,
        )

    def read(self, content_to_read: str) -> None:
        """Reads the variable from the provided content.

        Args:
            content_to_read (str): The content where the variable is read from.
        """
        # Check if the content contains the variable otherwise log a warning
        if self._variable_identifier not in content_to_read:
            raise SetupError(f"Cannot find {self._variable_identifier} in C++ file content.") from None
        # Define the prefix group to be searched for
        prefix_group = "" if self._namespace_prefix is None else "(.+?" + self._namespace_prefix + ")"
        # Read the appropriate value from the content
        read_value = re.search(f"(.+?{self._variable_identifier}.+?)={prefix_group}(.+?);", content_to_read)
        # Set the data of this tag if anything has been found
        if read_value is not None:
            self.value = read_value.group(2 if not prefix_group else 3).strip().strip('"')
