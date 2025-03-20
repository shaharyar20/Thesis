#!/usr/bin/env python3
# Python modules
from typing import List, Any

# torchlbm modules
from torchlbm.enum_type import EnumType
from torchlbm.setup_definitions.type_converters import TypeConverter
from torchlbm.exceptions import TypeConversionError
from torchlbm.setup_definitions import SetupName


class EnumConverter(TypeConverter):
    """Wrapper class for all enum type conversions.

    It guarantees that all enum classes that are used are of the same form.
    Each member of the enum class elements must fulfill the following properties:
        - Must be a string, unique to lower case letters, contain only Ascii letters
        - Name and value must match.
    The final type conversion is always a EnumType.

    Args:
        TypeConverter (TypeConverter): The base class of all type converters.
    """

    def __init__(self, enum_class: EnumType, allowed_values: List[str] = []):
        """Constructor.

        Args:
            enum_class (Enum): The enum class that should be used for type conversion.

        Raises:
            TypeConversionError: If the requirements for the enum class are not fulfilled.
        """
        # Check that the enum class is built correctly
        if not issubclass(enum_class, EnumType):
            raise TypeConversionError("Enum converter classes must be of type 'Enum'") from None
        if not all([enum_class.is_valid(member.name) for member in enum_class]):
            raise TypeConversionError("Enum converter classes must contain members only of type str with Ascii letters.") from None
        lower_case_values = [SetupName.LowerCase.format(member.value) for member in enum_class]
        lower_case_names = [SetupName.LowerCase.format(member.name) for member in enum_class]
        if not len(enum_class) == len(set(lower_case_names)) == len(set(lower_case_values)):
            raise TypeConversionError("Enum converter classes must have unique members converted to lower case letters.") from None
        if any([name != value for name, value in zip(lower_case_names, lower_case_values)]):
            raise TypeConversionError("Enum converter classes must have equal names and values.") from None
        allowed_values = enum_class.get_values(allowed_values)
        # Call base class constructor
        super().__init__(True)
        # Assign the member variables
        self._enum_class = enum_class
        self._name_dict = {SetupName.LowerCase.format(member.name): member.name for member in enum_class}
        self._allowed_values = allowed_values

    def _convert(self, value: Any) -> EnumType:
        """See base class definition.

        Returns:
            EnumType: The converted Enum element. The type of the real value depends on the input argument.
        """
        if not SetupName.is_valid(value) and not isinstance(value, self._enum_class):
            raise TypeConversionError(f"Element {value} invalid. Only strings with letters/numbers or enum elements can be used.") from None
        if isinstance(value, self._enum_class):
            enum_value = value
        else:
            if value.lower() not in self._name_dict:
                raise TypeConversionError(f"Element {value} is not part of enum class {self._enum_class}") from None
            enum_value = self._enum_class[self._name_dict[value.lower()]]
        if self._allowed_values and enum_value not in self._allowed_values:
            raise TypeConversionError(
                f"Element {value} is not allowed. Allowed values are [{', '.join([str(value) for value in self._allowed_values])}]"
            ) from None

        return enum_value
