#!/usr/bin/env python3
# Python modules
from typing import Any
from abc import ABC, abstractmethod

# torchlbm modules
from torchlbm.exceptions import TypeConversionError
import torchlbm.standalone_operations.string_operations as string_o


class TypeConverter(ABC):
    """Base class of all type converters.

    A type converter allows the conversion from a generic user defined type into a python-type. Furthermore, additional functionality
    can be added to restrict the types to certain ranges or allowing only specific values. All type converters must inherit from this class.
    All type converters must implement the conversion function _convert().

    Args:
        ABC (ABC): This class is an Abstract Base Class and cannot be instantiated without implementing the required functions.
    """

    def __init__(self, is_single_valued_type: bool) -> None:
        """Constructor.

        Args:
            is_single_valued_type (bool): Flag whether this is a single valued type or not.
        """
        self._is_single = is_single_valued_type

    def __str__(self) -> str:
        """Implementation of the built-in str function"""
        return str(self.__class__.__name__)

    def __repr__(self) -> str:
        """Implementation of the built-in repr function"""
        return self.__str__()

    def __call__(self, new_value: Any) -> Any:
        """Allows the usage of this class instance with () operator. Calls the convert function. See convert() definition."""
        return self.convert(new_value)

    def convert(self, value: Any) -> Any:
        """The convertsion function that converts a value into the appropriate type.

        Args:
            value (Any): The value that should be converted.

        Raises:
            TypeConversionError: If the conversion fails.

        Returns:
            Any: The converted value. None values are preserved.

        Note:
            Do not overwrite this method. Overwrite function body _convert().
        """
        if value is None or (isinstance(value, str) and string_o.is_none(value)):
            return None
        try:
            return self._convert(value)
        except (TypeConversionError, ValueError, TypeError) as err:
            raise TypeConversionError(f"Conversion to {self.__class__.__name__} failed.\n{str(err)}") from None

    def _compare(self, converted_value1: Any, converted_value2: Any) -> bool:
        """The comparison function each class can implement if no basic comparison with == should be used.
            May be re-implemented from derived classes.

        Args:
            converted_value1 (Any): The first already converted value.
            converted_value2 (Any): The second already converted value.

        Returns:
            bool: True if both are equal.
        """
        return converted_value1 == converted_value2

    def compare(self, value1: Any, value2: Any) -> bool:
        """Comapres to values with each other, if they are the same.

        Args:
            value1 (Any): The first value of the comparison.
            value2 (Any): The second value of the comparison.

        Raises:
            TypeConversionError: If the comparison process fails.

        Returns:
            bool: True if both are equal.

        Note:
            Do not overwrite this method. Overwrite function body _compare().
        """
        # First convert both values to the given type to check if they are correct
        try:
            converted_value1 = self.convert(value1)
            converted_value2 = self.convert(value2)
            if converted_value1 is None or converted_value2 is None:
                return converted_value1 == converted_value2
            else:
                return self._compare(converted_value1, converted_value2)
        except TypeConversionError as err:
            raise TypeConversionError(f"Comparison for {self.__class__.__name__} failed.\n{err}") from None

    def _format(self, converted_value: Any) -> str:
        """The formatting function each class can implement if no basic string conversion ({:s}) is desired.
            May be re-implemented by derived class.

        Args:
            converted_value (Any): The already converted value.

        Returns:
            str: The formatted string.
        """
        return "{:s}".format(converted_value)

    def format(self, value: Any) -> str:
        """Formats the value for the given type converter.

        Args:
            value (Any): The value that should be formatted.

        Raises:
            TypeConversionError: If the formatting process fails.

        Returns:
            str: The formatted string.

        Note:
            Do not overwrite this method. Overwrite function body _format().
        """
        # First convert both values to the given type to check if they are correct
        try:
            converted_value = self.convert(value)
            if converted_value is None:
                return "None"
            else:
                return self._format(converted_value)
        except TypeConversionError as err:
            raise TypeConversionError(f"Formatting for {self.__class__.__name__} failed.\n{err}") from None

    @property
    def is_single(self) -> bool:
        """Gives the flag if this converter is for single valued types or not as property.

        Returns:
            bool: True if single valued.
        """
        return self._is_single

    @abstractmethod
    def _convert(self, new_value: Any) -> Any:
        """The conversion function that needs to be implemented by all derived classes. See convert() for details.
            For properly catched exceptions this function may throw TypeError, ValueError or TypeConversionError. All
            other errors must be catched in the derived classes properly to allow error propagation.

        Args:
            new_value (Any): The new value that should be converted to the type.

        Returns:
            Any: The converted type.
        """
        pass
