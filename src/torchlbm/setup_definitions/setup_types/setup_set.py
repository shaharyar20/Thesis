#!/usr/bin/env python3
# Python modules
from typing import List, Dict, Union, Any
from copy import deepcopy
from collections.abc import MutableMapping

# torchlbm modules
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions.setup_types import Setup
from torchlbm.setup_definitions import SetupName


class SetupSet(Setup, MutableMapping):
    """The SetupSet class.

    The SetupSet class is the most complex Setup. All Setups or SetupTagClasses should inherit from this class.
    It allows the usage of any combination of SetupTag, SetupList or SetupSet itself. To get down layers, name indexing can be used.
    The name indexing is type-safe, i.e. Name, name or NaMe are all valid, since per layer no double named setups can exists if converted
    to lower case letters.

    Args:
        Setup (Setup): The base class of all setups.
        MutableMapping (MutableMapping): The abstact base class for a mutable map.
    """

    def __init__(self, name: str, setups: List[Setup] = []) -> None:
        """Constructor.

        Args:
            name (str): The name of the SetupSet.
            setups (Setup): Unpacked number of SetupTags.

        Raises:
            SetupError: If the provided setups are not of type Setup.
        """
        # Check that the setups are of correct type
        if any([not isinstance(setup, Setup) for setup in setups]):
            raise SetupError("Setups for SetupSet must be of type 'Setup'.") from None
        # Call the base class
        super().__init__(
            name,
            default=[deepcopy(setup) for setup in setups],
            none_is_valid=False,
            is_required=False,
            is_set=True,
        )
        # Initialize the additional member variables
        self._tag_names = [setup.name for setup in setups]
        self._value_indices = {SetupName.LowerCase.format(setup.name): index for index, setup in enumerate(setups)}

    def _content(self, indent: int):
        """See base class definition."""
        string = "\n"
        string += "\n".join([setup.content(indent, True) for setup in self.value])
        return string

    def __getitem__(self, name: str) -> Setup:
        """Required implementation for indexing of a MutableMapping.

        Args:
            name (str): The name of the setup to be set.

        Returns:
            Setup: The base class of all setups.
        """
        return self._value[self._get_checked_index(name)]

    def __setitem__(self, name: str, new_setup: Any) -> None:
        """Required implementation for index setting of a MutableMapping.

        Args:
            name (str): The name of the setup to be set.
            new_setup (Any): The new setup value to be set. Can be any value that is allowed by indexed setup.
        """
        self._value[self._get_checked_index(name)] = new_setup

    def __delitem__(self, name: str) -> None:
        """Required implementation for deletion of an element in a MutableMapping. Is prohibited for this implementation to avoid SetupTag deletion.

        Args:
            name (str): The name of the setup to be set.
        """
        raise SetupError("It is not allowed to delete a setup from an existing SetupSet") from None

    def __iter__(self) -> List[Setup]:
        """Required implementation to get the keys of the dictionary of a MutableMapping.

        Returns:
            List[Setup]: The list of setups.
        """
        return iter({key: value for key, value in zip(self._tag_names, self._value)})

    def __len__(self):
        """Required implementation to get the length of a MutableMapping.

        Returns:
            int: The length of the dict.
        """
        return len(self._value)

    def __contains__(self, name: str) -> bool:
        """Checks whether a name exists in the dictionary. Not required for the MutableMapping, but used to allow type_safe function call.

        Args:
            name (str): The name that should be checked.
        Returns:
            bool: True if the name exists in the dictionary, False otherwise.
        """
        return SetupName.LowerCase.format(name) in self._value_indices

    def _get_checked_index(self, name: str) -> int:
        """Checks whether the name is allowed for indexing.

        Args:
            name (str): The name that should be checked.

        Raises:
            SetupError: If the name is not a string.
            SetupError: If the name converted to lower case does not exist in the dictionary.

        Returns:
            int: The checked index.
        """
        if not isinstance(name, str):
            raise SetupError("Only strings are allowed for key to access setups") from None
        lower_case_name = SetupName.LowerCase.format(name)
        if lower_case_name not in self._value_indices:
            raise SetupError(f"The given name '{name}' is not found in SetupSet '{self.name}'\n" f"Valid tags are: [{', '.join(self._tag_names)}]") from None
        return self._value_indices[lower_case_name]

    def _get_checked_tag(self, new_setup: Union[List[Setup], Dict[str, Any]]):
        """See base class definition.

        Raises:
            SetupError: If a list is given and not all elements are of type Setup.
            SetupError: If the input argument ist not of type dict or list.
            SetupError: If not all required setups are set with the input data set.
        """
        # Copy the full tag dict
        created_setups = deepcopy(self._value)
        if isinstance(new_setup, SetupSet):
            for name, value in new_setup.items():
                if isinstance(value, Setup):
                    created_setups[self._get_checked_index(name)] = value
                else:
                    created_setups[self._get_checked_index(name)].value = value
        elif isinstance(new_setup, list):
            for entry in new_setup:
                if not isinstance(entry, Setup):
                    raise SetupError("Cannot set tag value. Type is not a derived class of Setup") from None
            lower_case_names = [SetupName.LowerCase.format(entry.name) for entry in new_setup]
            for setup in created_setups:
                if setup.is_required() and SetupName.LowerCase.format(setup.name) not in lower_case_names:
                    raise SetupError(f"Cannot find {setup.name} in provided list for SetupSet '{self.name}'.") from None
            # Assign all values to the correct tag
            for entry in new_setup:
                created_setups[self._get_checked_index(entry.name)] = entry
        elif isinstance(new_setup, dict):
            lower_case_names = [SetupName.LowerCase.format(key) for key in new_setup.keys()]
            for setup in created_setups:
                if setup.is_required() and SetupName.LowerCase.format(setup.name) not in lower_case_names:
                    raise SetupError(f"Cannot find {setup.name} in provided dictionary for SetupSet '{self.name}'.") from None
            for name, tag_value in new_setup.items():
                if isinstance(tag_value, Setup):
                    created_setups[self._get_checked_index(name)] = tag_value
                else:
                    created_setups[self._get_checked_index(name)].value = tag_value
        else:
            raise SetupError("You can only set the SetupSet value with a list of SetupTags, a dict or a new SetupSet.") from None
        return created_setups
