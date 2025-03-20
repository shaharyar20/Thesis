#!/usr/bin/env python3
# Python modules
from typing import List, Any

# torchlbm modules
from torchlbm.setup_definitions.type_converters import TypeConverter
from torchlbm.exceptions import TypeConversionError


class StringConverter(TypeConverter):
    """Wrapper class for all string type conversions.

    It can be used to allow only a set of strings and allow case insensitive string checks.
    The final type conversion is always a standard python-str type.

    Args:
        TypeConverter (TypeConverter): The base class of all type converters.
    """

    def __init__(
        self,
        allowed_strings: List[str] = [],
        case_sensitive: bool = True,
        prohibited_characters: List[str] = [],
        allowed_characters: List[str] = [],
    ) -> None:
        """Constructor.

        Args:
            allowed_strings (List[str], optional): A list of strings that are allowed. For all others an error is thrown. Defaults to [].
            case_sensitive (bool, optional): Flag whether the converter should check case sensitive. Defaults to True.

        Raises:
            TypeConversionError: If the allowed strings are not of type str.
        """
        # Check that the allowed string are correct
        if not isinstance(allowed_strings, list):
            raise TypeConversionError("Allowed strings variable for StringConverter must be of type 'list'") from None
        if any([not isinstance(string, str) for string in allowed_strings]):
            raise TypeConversionError("Each allowed string for StringConverter must be of type 'str'") from None
        if any([len(string) > 1 for string in prohibited_characters]):
            raise TypeConversionError("Each prohibited character must be a single character 'str'") from None
        if any([len(string) > 1 for string in allowed_characters]):
            raise TypeConversionError("Each allowed character must be a single character 'str'") from None
        # Call the base class constructor
        super().__init__(True)
        # Assign the member variables
        self._case_sensitive = case_sensitive
        self._allowed_strings = allowed_strings
        self._allowed_strings_lower = allowed_strings if case_sensitive else [string.lower() for string in allowed_strings]
        self._prohibited_characters_lower = prohibited_characters if case_sensitive else [char.lower() for char in prohibited_characters]
        self._allowed_characters_lower = allowed_characters if case_sensitive else [char.lower() for char in allowed_characters]

    def _convert(self, value: Any) -> str:
        """See base class definition.

        Returns:
            str: The converted python-str value.
        """
        # Convert the entry to string (if not possible, it already raises error)
        converted_string = str(value).strip()
        if not self._case_sensitive:
            converted_string = converted_string.lower()
        # Check if a prohibited character is in the value
        if any([char in converted_string for char in self._prohibited_characters_lower]):
            raise TypeConversionError(
                f"String '{value}' contains one of the prohibited characters:\n[{', '.join(self._prohibited_characters_lower)}]"
            ) from None
        if self._allowed_characters_lower and any([char not in self._allowed_characters_lower for char in converted_string]):
            raise TypeConversionError(f"String '{value}' must contain only an allowed character:\n[{', '.join(self._allowed_characters_lower)}]") from None
        # Check if the string is not in the allowed list
        if self._allowed_strings:
            if converted_string not in self._allowed_strings_lower:
                raise TypeConversionError(f"String '{value}' not in list of allowed values:\n[{', '.join(self._allowed_strings)}]") from None
            return self._allowed_strings[self._allowed_strings_lower.index(converted_string)]
        return str(value).strip()
