#!/usr/bin/env python3
# Python modules
from typing import List, Any

# torchlbm modules
from torchlbm.setup_definitions.type_converters import TypeConverter
from torchlbm.exceptions import TypeConversionError


class ClassConverter(TypeConverter):
    """Wrapper class for all class type conversions.

    This converter is the most general and only checks if the type of the new set value is of desired class type. No 'real' conversion is done.

    Args:
        TypeConverter (TypeConverter): The base class of all type converters.
    """

    def __init__(self, allowed_classes: List[Any] = []) -> None:
        """Constructor.

        Args:
            allowed_classes (List[Any], optional): A list of allowed classes. Defaults to [].
        """
        allowed_classes = [allowed_classes] if not isinstance(allowed_classes, (list, tuple)) else allowed_classes
        # Call the base class constructor
        super().__init__(True)
        # Assign the member variables
        self._allowed_classes = allowed_classes

    def _convert(self, value: Any) -> str:
        """See base class definition.

        Returns:
            str: The converted python-str value.
        """
        # Check if the value is of type of the desired class
        if not self._allowed_classes or any([isinstance(value, class_type) for class_type in self._allowed_classes]):
            return value
        else:
            raise TypeConversionError(f"Value is not of type {self._allowed_classes}") from None
