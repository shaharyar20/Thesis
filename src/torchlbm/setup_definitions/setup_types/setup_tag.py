#!/usr/bin/env python3
# Python modules
from typing import Any

# torchlbm modules
from torchlbm.exceptions import SetupError, TypeConversionError
from torchlbm.setup_definitions.setup_types import Setup
from torchlbm.setup_definitions.type_converters import TypeConverter


class SetupTag(Setup):
    """The SetupTag class.

    The SetupTag class is the last tag instance and holds a single scalar value of any type (int, bool, float, str). No lists, tuple, dicts or other
    Multi-classes should be use. Therefore, SetupList and SetupSet exists.

    Args:
        Setup (Setup): The base class of all setups.
    """

    def __init__(
        self,
        name: str,
        default_value: Any,
        tag_required: bool,
        type_converter: TypeConverter,
    ) -> None:
        """Constructor.

        Args:
            name (str): The name of the setup tag.
            default_value (Any): The default value of the setup tag.
            tag_required (bool): Flag whether this setup tag must be set explicitly.
            type_converter (Type): The type converter of the setup tag.

        Raises:
            SetupError: If the default value does not coincide with the provided type.
        """
        # Check if the value type is correct
        if not isinstance(type_converter, TypeConverter):
            raise SetupError(f"Type {type_converter} is not of type 'TypeConverter'") from None
        # Check that the default value coincides with the specified type
        try:
            converted_default_value = type_converter(default_value) if default_value is not None and not tag_required else default_value
        except TypeConversionError:
            raise SetupError(f"The provided default value does not coincide with the given type '{type_converter}'.") from None
        # Call the base class
        super().__init__(
            name,
            converted_default_value,
            none_is_valid=not tag_required,
            is_required=tag_required,
            is_set=False,
        )
        # Assign the variables required only for this dervied class
        self._type_converter = type_converter

    def _content(self, indent: int) -> str:
        """See base class definition."""
        return f"{self._value}"

    def _get_checked_tag(self, new_value: Any) -> Any:
        """See base class definition."""
        try:
            return self._type_converter(new_value)
        except TypeConversionError as err:
            raise SetupError(f"Error for SetupTag '{self.name}'\n{str(err)}") from None
