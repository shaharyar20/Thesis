#!/usr/bin/env python3
# Python modules
from typing import List, Union
from enum import Enum

# own modules
from torchlbm.exceptions import TypeConversionError


class EnumType(Enum):
    """Proxy class for all enum classes used in the torchlbm module. All enum types must inherit from it. It defines basic interface functions
    that can be used for all enum classes. All derived classes must use strings as values.
    """

    def __new__(cls, value: str) -> "EnumType":
        """Overwrites the new method to enforce variable assignment.

        Args:
            value (str): The value to be set.

        Raises:
            TypeConversionError: If the value if not an EnumType.

        Returns:
            EnumType: The enum type class.
        """
        if not cls.is_valid(value):
            raise TypeConversionError("Enum type values must be of type string and contain letters, number or underscores.") from None
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __str__(self) -> str:
        """Shows the class member identifier when using str() or print()

        Returns:
            str: The member name.
        """
        return self.value

    def __repr__(self) -> str:
        """Overwrites built in method __repr__."""
        return self.__str__()

    @classmethod
    def has_value(cls, value: str) -> bool:
        """Checks whether the value exists in the enum class.

        Args:
            value (str): The value to be checked.

        Returns:
            bool: True if the value exists.
        """
        if not isinstance(value, str):
            return False
        return value.lower() in list(map(lambda c: c.value.lower(), cls))

    @classmethod
    def get_value(cls, string: str) -> "EnumType":
        """Gives the enum type value for a given string.

        Args:
            string (str): The string identifier of the value that should be returned.

        Raises:
            TypeConversionError: If the value cannot be returned.

        Returns:
            EnumType: Tje enum type value.
        """
        if not cls.has_value(string):
            raise TypeConversionError(f"Value '{string}' cannot be found in enum '{cls.__name__}'") from None
        return [cls[name] for name, value in list(map(lambda c: (c.name, c.value), cls)) if value.lower() == string.lower()][0]

    @classmethod
    def get_values(cls, values_to_convert: List[Union[str, "EnumType"]]) -> List["EnumType"]:
        """Gives the enum type value for a given list of strings or enum types.

        Args:
            values_to_convert (List[Union[str,'EnumType']]): The string or EnumType identifier of the value that should be returned.

        Returns:
            List[EnumType]: The list of enum type values.
        """
        values_to_convert = [values_to_convert] if not isinstance(values_to_convert, (list, tuple)) else values_to_convert
        values_to_convert = [str(value) for value in values_to_convert]
        return list(set([cls.get_value(value) for value in values_to_convert]))

    @classmethod
    def is_valid(cls, string: str) -> bool:
        """Checks if string is a valid string for the enum name syntax, i.e. consists only of letters, numbers and underscores.

        Args:
            string (str): The string to be checked.

        Returns:
            bool: True if string is valid.
        """
        return isinstance(string, str) and all([char.isalpha() or char.isdigit() or char == "_" for char in string])
