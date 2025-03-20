#!/usr/bin/env python3
# Python modules
from typing import Union, Any
import math

# torchlbm modules
from torchlbm.setup_definitions.type_converters import TypeConverter
from torchlbm.exceptions import TypeConversionError


class RangeConverter(TypeConverter):
    """Wrapper class for all range based type conversions.

    It is the base class for all float and int type conversion and allows to set a fixed range of the value. Furthermore, the range can be forced
    (Error if outside) or cut (min/max values if outside). The final type conversion is always a standard python-int or python-float type.

    Args:
        TypeConverter (TypeConverter): The base class of all type converters.
    """

    def __init__(
        self,
        value_type: Union[int, float],
        min_value: Union[int, float] = None,
        max_value: Union[int, float] = None,
        cut_range: bool = False,
    ) -> None:
        """Constructor.

        Args:
            value_type (Union[int, float]): The value type to be used (int or float).
            min_value (Union[int,float], optional): The min value of the allowed range (None for open). Defaults to None.
            max_value (Union[int,float], optional): The max value of the allowed range (None for open). Defaults to None.
            cut_range (bool, optional): Flag whether the range should be cut (True) or forced (False). Defaults to False.

        Raises:
            TypeConversionError: If the value type is not of float or int.
        """
        # Check that the min and max value are of correct type
        if any([value is not None and not isinstance(value, (float, int)) for value in [min_value, max_value]]):
            raise TypeConversionError(f"Min and max value for {self.__class__.__name__} must be of type 'int or float'") from None
        super().__init__(True)
        # Assign the member variables
        self._min_value = value_type(min_value) if min_value is not None else None
        self._max_value = value_type(max_value) if max_value is not None else None
        self._cut_range = cut_range
        self._value_type = value_type

    def _convert(self, value: Any) -> Union[int, float]:
        """See base class definition.

        Raises:
            TypeConversionError: If the conversion fails.

        Returns:
            Union[int,float]: The converted python-int or python-float value.
        """
        # Convert the value with the appropriate type. First conversion to string to make float and int values really distinguishable.
        # E.g.: 1.0 can be converted to integer, but str(1.0) cannot. This ensures that the type converter really works only for values
        # with the given type. Other conversion will fail.
        try:
            converted_value = self._value_type(str(value))
        except (ValueError, TypeError):
            converted_value = None
        if converted_value is None:
            raise TypeConversionError
        if not self._cut_range and (
            (self._max_value is not None and self._max_value < converted_value) or (self._min_value is not None and converted_value < self._min_value)
        ):
            raise TypeConversionError(f"Value {value} out of range [{self._min_value} ; {self._max_value}]") from None
        if self._cut_range:
            if self._max_value is not None and converted_value > self._max_value:
                converted_value = self._max_value
            if self._min_value is not None and converted_value < self._min_value:
                converted_value = self._min_value
        return converted_value

    def _format(self, value: Any) -> bool:
        """See base class definition."""
        if self._value_type == int:
            return "{:16d}".format(value)
        else:
            return "{:.10e}".format(value)

    def _compare(self, converted_value1: Any, converted_value2: Any) -> bool:
        """See base class definition."""
        if self._value_type == int:
            return converted_value1 == converted_value2
        else:
            return math.isclose(converted_value1, converted_value2, rel_tol=1e-9, abs_tol=1e-9)


class IntConverter(RangeConverter):
    """Simplified range type converter for int types.

    Args:
        RangeConverter (RangeConverter): The base class of all range type converters
    """

    def __init__(self, min_value: int = None, max_value: int = None, cut_range: bool = False) -> None:
        """See base class definition."""
        super().__init__(int, min_value, max_value, cut_range)


class FloatConverter(RangeConverter):
    """Simplified range type converter for float types.

    Args:
        RangeConverter (RangeConverter): The base class of all range type converters
    """

    def __init__(self, min_value: float = None, max_value: float = None, cut_range: bool = False) -> None:
        """See base class definition."""
        super().__init__(float, min_value, max_value, cut_range)
