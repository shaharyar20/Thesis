#!/usr/bin/env python3
# Python modules
from typing import List, Dict, Union, Any
from copy import deepcopy

# torchlbm modules
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions.setup_types import SetupSet, Setup, SetupTag
from torchlbm.setup_definitions import SetupName


class SetupVariableSet(SetupSet):
    """Implementation of a SetupSet if a variable number of setup tags.

    The SetupVariableSet class is basically the same as the SetupSet. The main difference between those two is that the SetupSet
    has fixed setups, but can consists of all sub-classes, whereas the SetupVariableSet can only consists of single SetupTags,
    but variable setups. This means, for the SetupSet all setups always exist, but for the SetupVariableSet only those
    exist that are required. All optional tags exist only when they are set once. The default tags of this class are always all tags that are tagged
    as required.

    In this class some functions of the SetupSet are overwritten to ensure the desired behavior.

    Args:
        SetupSet (SetupSet): This class is a SetupSet.
    """

    def __init__(self, name: str, setups: List[Setup] = []) -> None:
        """Constructor.

        Args:
            name (str): The name of the SetupSet.
            setups (Setup): Unpacked number of SetupTags.

        Raises:
            SetupError: If the provided setups are not of type SetupTag.
        """
        # Check that the setups are of correct type
        if any([not isinstance(setup, SetupTag) for setup in setups]):
            raise SetupError("Setups for SetupVariableSet must be of type SetupTag.") from None
        # Call the base class
        super().__init__(name, setups)
        # Initialize the additional member variables (here, the SetupSet overwrites the self._value variable of the base class.)
        self._value = []
        self._default = [setup for setup in setups if setup.is_required()]
        self._value_tags = {SetupName.LowerCase.format(setup.name): deepcopy(setup) for setup in setups}
        self._value_indices = {SetupName.LowerCase.format(setup.name): None for setup in setups}

    def _create_new_entry(self, name: str) -> None:
        """Creates a new entry for the given name, if it does not already exists.

        Args:
            name (str): The name of the setup that should be added.
        """
        lower_case_name = SetupName.LowerCase.format(name)
        if self._value_indices[lower_case_name] is None:
            # Get the appropriate tag from the default value
            new_tags = [tag for tag in self._value_tags.values() if SetupName.LowerCase.format(tag.name) == lower_case_name]
            new_tag = deepcopy(new_tags[0])
            # Assign the new tag and index to the current list of tags
            self._value_indices[lower_case_name] = len(self._value)
            self._value.append(new_tag)

    def _get_checked_index(self, name: str) -> int:
        """See base class definition.

        Raises:
            SetupError: If the name is not a string.
            SetupError: If the name is not a valid name for a setup.
        """
        if not isinstance(name, str):
            raise SetupError("Only strings are allowed for key to access setups") from None
        lower_case_name = SetupName.LowerCase.format(name)
        if lower_case_name not in self._value_indices:
            raise SetupError(
                f"The given name '{name}' is not found in SetupVariableSet '{self.name}'\n" f"Valid tags are: [{', '.join(self._tag_names)}]"
            ) from None
        # If the name is currently not present in the list, we create the element
        self._create_new_entry(name)
        # Return the index
        return self._value_indices[lower_case_name]

    def _get_checked_tag(self, new_setup: Union[List[Setup], Dict[str, Any]]) -> Any:
        """See base class definition.

        Raises:
            SetupError: If a list is given and not all elements are of type Setup.
            SetupError: if the input argument ist not of type dict or list.
        """
        # Get all names of the new setups
        if isinstance(new_setup, list):
            if any([not isinstance(entry, Setup) for entry in new_setup]):
                raise SetupError("Cannot set tag value. Type is not a derived class of Setup") from None
            new_setup_names = [entry.name for entry in new_setup]
        elif isinstance(new_setup, (dict, SetupSet)):
            new_setup_names = list(new_setup.keys())
        else:
            raise SetupError("You can only set the SetupVariableSet value with a list of SetupTags, a dict or a SetupSet.") from None
        # Create all new entries for the given names
        for name in new_setup_names:
            self._create_new_entry(name)
        # Call the base class method
        return super()._get_checked_tag(new_setup)

    def reset(self) -> None:
        """OVerwrites the reset function, since default value is used differently in this context."""
        self._value = []
        self._value_indices = {key: None for key in self._value_indices.keys()}

    def all_values(self) -> List[Setup]:
        """Gives all values of the set.

        Returns:
            List[Setup]: List with all setup variables of this set.
        """
        return list(self._value_tags.values())
