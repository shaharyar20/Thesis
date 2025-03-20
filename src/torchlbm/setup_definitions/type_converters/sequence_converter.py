#!/usr/bin/env python3
# Python modules
from typing import Union, Any

# torchlbm modules
from torchlbm.setup_definitions.type_converters import TypeConverter
from torchlbm.exceptions import TypeConversionError
import torchlbm.standalone_operations.string_operations as string_o


class SequenceConverter(TypeConverter):
    """Wrapper class for all sequence type conversions.

    It is the base class for all list and tuple type conversion. Each element of the sequence must be of the same type and a single-valued TypeConverter.
    The final type conversion is always a standard python-tuple or python-list type.

    Args:
        TypeConverter (TypeConverter): The base class of all type converters.
    """

    def __init__(
        self,
        value_type: Union[list, tuple],
        element_converter: TypeConverter,
        min_elements: int = 0,
        max_elements: int = None,
    ) -> None:
        """Constructor.

        Args:
            value_type (Union[list,tuple]): The value type of the sequence (list or tuple).
            element_converter (TypeConverter): The converter used for each element.
            min_elements (int, optional): The minimum number of elements required for the sequence. Defaults to 0.
            max_elements (int, optional): The maximum number of elements allowed for the sequence.. Defaults to None.

        Raises:
            TypeConversionError: If the element converters or sequence bounds are of incorrect type. If the bounds are non-positive.
        """
        # Check if the provided arguments are correct
        if not isinstance(element_converter, TypeConverter) and not element_converter.is_single:
            raise TypeConversionError("Element type for a SequenceConverter must be a single valued TypeConverter") from None
        if not isinstance(min_elements, int) or (max_elements is not None and not isinstance(max_elements, int)):
            raise TypeConversionError("The max and min number of elements for a SequenceConverter must be positive integer values.") from None
        if min_elements < 0 or (max_elements is not None and max_elements < 0):
            raise TypeConversionError("The max and min number of elements for SequenceConverter must be positive or zero numbers.") from None
        # Call the base class constructor
        super().__init__(False)
        # Assign the member variables for this class
        self._min_number_of_elements = min_elements
        self._max_number_of_elements = max_elements
        self._element_converter = element_converter
        self._value_type = value_type

    def _convert(self, value: Any) -> Union[list, tuple]:
        """See base class definition.

        Raises:
            TypeConversionError: If any sequence value or the sequence itself is of wrong type or the number of elements is incorrect.

        Returns:
            Union[list,tuple]: The converted python-list or python-tuple value.
        """
        # Convert single values into a list of single values
        value = [value] if not isinstance(value, (list, tuple)) and not isinstance(value, str) else value
        # In case a string is given convert the string to a list of elements
        value_list = string_o.convert_string_to_list(value) if isinstance(value, str) else value
        # Check the number of elements and if each element can be converted to the desired type
        if len(value_list) < self._min_number_of_elements or (self._max_number_of_elements is not None and len(value_list) > self._max_number_of_elements):
            raise TypeConversionError(
                f"Number of elements exceeded: Must be between [{self._min_number_of_elements} ; {self._max_number_of_elements}]"
            ) from None
        # Return the converted types. If any fails an error is thrown by the TypeConverter.
        return self._value_type([self._element_converter.convert(value) for value in value_list])

    def _compare(self, converted_value1: Any, converted_value2: Any) -> bool:
        """See base class definition."""
        return all([self._element_converter.compare(value1, value2) for value1, value2 in zip(converted_value1, converted_value2)])

    def _format(self, converted_value: Any) -> str:
        """See base class definition."""
        return all([self._element_converter.format(value) for value in converted_value])


class TupleConverter(SequenceConverter):
    """Simplified sequence type converter for tuple types with a fixed number of elements.

    Args:
        SequenceConverter (SequenceConverter): The base class of all sequence type converters.
    """

    def __init__(self, element_converter: TypeConverter, number_of_elements: int):
        """Constructor.

        Args:
            element_converter (TypeConverter): The converter used for each element.
            number_of_elements (int): The fixed number of elements the tuple is built of.
        """
        super().__init__(tuple, element_converter, number_of_elements, number_of_elements)


class ListConverter(SequenceConverter):
    """Simplified sequence type converter for list types with a non-fixed number of elements.

    Args:
        SequenceConverter (SequenceConverter): The base class of all sequence type converters.
    """

    def __init__(
        self,
        element_converter: TypeConverter,
        min_elements: int = 0,
        max_elements: int = None,
    ):
        """Constructor. See base class defintion."""
        super().__init__(list, element_converter, min_elements, max_elements)
