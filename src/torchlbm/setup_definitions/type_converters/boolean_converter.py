#!/usr/bin/env python3
# Python modules
from typing import Any

# torchlbm modules
from torchlbm.setup_definitions.type_converters import TypeConverter
from torchlbm.exceptions import TypeConversionError


class BoolConverter(TypeConverter):
    """Wrapper class for all boolean type conversions.

    The conversion allows the usage of values of style ["true", "1", "false", "0"]. To convert a value into boolean, the str() method must be applicable.
    All other conversions fail and throw an error. The final type conversion is always a standard python-bool type.

    Args:
        TypeConverter (TypeConverter): The base class of all type converters.
    """

    def __init__(self) -> None:
        """Constructor."""
        super().__init__(True)

    def _convert(self, value: Any) -> bool:
        """See base class definition.

        Returns:
            bool: The converted python-bool value.
        """
        # Convert the entry to string (if not possible, it already raises error)
        converted_value = str(value).strip().lower()
        # Check if it fulfills the two criteria
        if converted_value in ["true", "1"]:
            return True
        elif converted_value in ["false", "0"]:
            return False
        else:
            raise TypeConversionError
