#!/usr/bin/env python3
# Python modules
import re

# own modules
from torchlbm.enum_type import EnumType
from torchlbm.exceptions import SetupError


class SetupName(EnumType):
    """The SetupName provides all possible styles a name for a setup can take. A name in general is defined in CamelCaseStyle.
    The different name styles are defines as:
       Camel       : Camel case, e.g. InternalCells.
       Xml, Json   : First letter lower case, e.g. internalCells.
       Log         : Only first letter is upper case and add space between camel case letters, e.g. Internal cells.
       Abbreviation: Only take all upper case letters, e.g. IC.
       ArgParser   : Convert all to lower case and add underscore between camel case letters, e.g. internal_cells.
       LowerCase   : Convert all to lower case letters
       UpperCase   : Convert all to upper case letters
    """

    Camel = "Camel"
    Log = "Log"
    LowerCase = "LowerCase"
    UpperCase = "UpperCase"
    Abbreviation = "Abbreviation"
    ArgParser = "ArgParser"
    Xml = "XML"
    Json = "JSON"
    Yaml = "YAML"

    def format(self, string: str) -> str:
        """Formats a string in the given name style.

        Args:
            string (str): The string that should be converted.

        Returns:
            str: The converted string.
        """
        # Remove all non allowed letters
        if not isinstance(string, str) or not self.is_valid(string):
            raise SetupError("Cannot format string. It is not a valid string for the SetupName convention.") from None
        # Turn everything into lower/upper case
        if self == SetupName.LowerCase:
            return string.lower()
        elif self == SetupName.UpperCase:
            return string.upper()
        # Replace the first entry with lower case but keep rest (e.g., Camel style). Applies only for tags, where the first letter is
        # followed by a lower case letter.
        elif self == SetupName.Xml or self == SetupName.Json or self == SetupName.Yaml:
            return f"{string[0].lower()}{string[1:]}" if len(string) > 1 and string[1].islower() else string
        # Replace camel style with spaces
        elif self == SetupName.Log:
            found_cases = re.findall("([a-z])([A-Z])", string)
            for case in found_cases:
                string = re.sub(case[0] + case[1], case[0] + " " + case[1], string)
            found_cases = re.findall("([a-z])([0-9])", string)
            for case in found_cases:
                string = re.sub(case[0] + case[1], case[0] + " " + case[1], string)
            found_cases = re.findall("([0-9])([a-zA-Z])", string)
            for case in found_cases:
                string = re.sub(case[0] + case[1], case[0] + " " + case[1], string)
            return string
        # For abbreviation style, the Camel case letters are printed only
        elif self == SetupName.Abbreviation:
            return "".join(letter for letter in string if letter.isupper() or letter.isdigit())
        # Name used for the argument parser (connect Camelcase with underscore and make everythin lower)
        elif self == SetupName.ArgParser:
            string = re.sub("([a-z])([A-Z])", r"\g<1>-\g<2>", string).lower()
            string = re.sub("([a-z])([0-9])", r"\g<1>-\g<2>", string).lower()
            return re.sub("([0-9])([a-z])", r"\g<1>-\g<2>", string).lower()
        else:
            return string
