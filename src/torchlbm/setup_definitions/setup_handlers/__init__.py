"""Setup handler."""

# Classes and functions
from .setup_handler_type import SetupHandlerType
from .setup_handler import SetupHandler
from .xml_setup_handler import XMLSetupHandler
from .json_setup_handler import JSONSetupHandler
from .setup_handler_factory import instantiate_setup_handler

# Data for wildcard import (from . import *)
__all__ = [
    "SetupHandlerType",
    "SetupHandler",
    "XMLSetupHandler",
    "JSONSetupHandler",
    "instantiate_setup_handler",
]
