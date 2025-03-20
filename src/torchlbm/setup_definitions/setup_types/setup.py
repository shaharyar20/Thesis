#!/usr/bin/env python3
# Python modules
from typing import Any
from copy import deepcopy
from abc import ABC, abstractmethod

# torchlbm modules
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions import SetupName
import torchlbm.standalone_operations.string_operations as string_o


class Setup(ABC):
    """The base class of all setups.

    Defines the generic variables and interfaces all SetupTag classes require. For derived classes the following
        functions must be implemented:
            - self._content(int): Prints the content of the class
            - self._get_checked_tag(Any): Checks a new input tag to be correct.

    Args:
        ABC (ABC): This class is an Abstract Base Class and cannot be instantiated without implementing the required functions.
    """

    def __init__(
        self,
        name: str,
        default: Any,
        none_is_valid: bool = False,
        is_required: bool = False,
        is_set: bool = False,
    ) -> None:
        """Constructor.

        Args:
            name (str): The name of the SetupTag (used for key-indexing).
            default (Any): The default tag value of the class.
            none_is_valid (bool): Flag whether None values are allowed for this SetupTag.
            is_required (bool): If the tag must be set by the user explicitly.

        Raises:
            SetupError: The name must be compliant to the SetupName convention.
            SetupError: If None is default, but None is not allowed.
            SetupError: If tag is required, but is allowed.
        """
        # Check the input arguments
        if not SetupName.is_valid(name):
            raise SetupError(f"Invalid name {name}. The setup name must contain only letters, numbers or underscores") from None
        if not none_is_valid and default is None:
            raise SetupError("Default is set to 'None', but None is not a valid value.") from None
        if is_required and none_is_valid:
            raise SetupError("'None' can only be a valid value if tag is not required.") from None
        # Set the class variables
        self._name = name
        self._value = None if is_required else deepcopy(default)
        self._default = deepcopy(default)
        self._none_is_valid = none_is_valid
        self._is_required = is_required
        self._is_set = is_set

    def __str__(self) -> str:
        """Reimplementation of the built-in __str__ function to print the content of the class using print() or str().

        Returns:
            str: The content.
        """
        return self.content(0)

    def __repr__(self) -> str:
        """Reimplementation of the built-in __repr__ function to print the content of the class using repr().

        Returns:
            str: The content.
        """
        return self.__str__()

    @abstractmethod
    def _content(self, indent: int) -> str:
        """Detailed implementation of the content function. Must be implemented by all derived classes.

        Args:
            indent (int): The current indent to be used.

        Returns:
            str: The content.
        """
        pass

    def content(self, indent: int, use_name: bool = True) -> str:
        """Detailed implementation of the content printing.

        Args:
            indent (int): Indent to be used for this class.
            use_name (bool, optional): Flag whether the name should be printed or not. Defaults to True.

        Returns:
            str: The content.

        Note:
            Do not overwrite this method. Only overwrite function body _content().
        """
        if use_name:
            string = f"{' ' * indent}{self.name}: "
            string += self._content(indent + 2)
        else:
            string = self._content(indent)
        return string

    @property
    def value(self) -> Any:
        """Allows accessing the current tag value of the class via .value as non-modifiable property.

        Returns:
            Any: The current tag value.
        """
        return self._value

    @value.setter
    def value(self, new_value: Any) -> None:
        """Allows setting the current value of the class via .value.

        Args:
            new_value (Any): The new value to be used.
        """
        new_value_is_none = string_o.is_none(new_value)
        if not self._none_is_valid and new_value_is_none:
            raise SetupError(f"The value for tag {self.name} must not be None.") from None
        self._value = None if new_value_is_none else self._get_checked_tag(new_value)
        self._is_set = True

    @abstractmethod
    def _get_checked_tag(self, new_value: Any) -> Any:
        """Detailed implementation of the check_value function.

        Args:
            new_value (Any): The new value to be set.

        Returns:
            Any: The checked value.
        """
        pass

    @property
    def name(self) -> str:
        """Allows accessing the name of the class via .name as non-modifiable property.

        Returns:
            str: The class name.
        """
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Sets the new name of the setup.

        Args:
            new_name (str): The new name.
        """
        if not SetupName.is_valid(new_name):
            raise SetupError("The setup name must contain only letters, numbers or underscores") from None
        self._name = new_name

    @property
    def default(self) -> "Setup":
        """Allows accessing the default content of the SetupTag .default as non-modifiable property.

        Returns:
            Setup: The default content.
        """
        return self._default

    def is_required(self) -> bool:
        """Gives the is_required tag.

        Returns:
            bool: If the tag must be set explicitly.
        """
        return self._is_required

    def is_set(self) -> bool:
        """Gives the is_set tag.

        Returns:
            bool: If the tag has been set explicitly.
        """
        return self._is_set

    def is_valid(self) -> bool:
        """Checks whether the tag is valid. A tag is not valid if it is None, but required.

        Returns:
            bool: If the tag is valid.
        """
        return not (self._is_required and self._value is None)

    def reset(self) -> None:
        """Resets the tag value to the default value. May be re-implemented by derived class."""
        self.value = self._default
