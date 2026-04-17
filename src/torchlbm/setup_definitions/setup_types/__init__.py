""" Setup types. """

# Classes and functions
from .setup import Setup
from .setup_tag import SetupTag
from .setup_set import SetupSet
from .setup_variable_set import SetupVariableSet
from .setup_list import SetupList
from .cxx_variable import CxxVariable
from .cxx_namespace import CxxNamespace
from .slurm_variable import SlurmVariable

# Data for wildcard import (from . import *)
__all__ = [
    "Setup",
    "SetupSet",
    "SetupVariableSet",
    "SetupList",
    "SetupTag",
    "CxxVariable",
    "CxxNamespace",
    "SlurmVariable",
]
