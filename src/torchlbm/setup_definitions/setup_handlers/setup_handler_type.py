#!/usr/bin/env python3

# own modules
from torchlbm.enum_type import EnumType


class SetupHandlerType(EnumType):
    """The setup types that are allowed to be used to write general setups to file.

    Args:
        EnumType: The base class of all enum classes.
    """

    xml = "XML"
    json = "JSON"
    yaml = "YAML"
