#!/usr/bin/env python3
# Python modules
from typing import List, Dict, Union, Any
from copy import deepcopy
from collections.abc import MutableSequence

# torchlbm modules
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions.setup_types import Setup, SetupTag, SetupSet


class SetupList(Setup, MutableSequence):
    """The SetupList class.

    The SetupList class holds n-numbers of SetupTag`s in a list. Where each element in the list represents the same SetupTag or SetupSet. The default
    value of this class is pre-defined an empty list. The list always must be filled after creation. The SetupTag(Set) per entry is the same for all entries.

    Args:
        Setup (Setup): The base class of all setups.
        MutableSequence (MutableSequence): The abstact base class for a mutable list.
    """

    def __init__(self, name: str, entry_tag: Setup):
        """Constructor.

        Args:
            name (str): The name of the SetupList.
            entry_tag (Setup): The SetupTag(Set) the list is built of.

        Raises:
            SetupError: If the provided entry tag is not a SetupTag or SetupSet.
        """
        # Check that the entry_tag is either SetupTag or SetupSet
        if not isinstance(entry_tag, (SetupTag, SetupSet)):
            raise SetupError(f"For {self.__class__.__name__} only SetupTag or SetupSet can be used as entry.") from None
        # Call the base class
        super().__init__(name, default=[], none_is_valid=False, is_required=False, is_set=False)
        # Assign the member variables
        self._entry_tag = deepcopy(entry_tag)

    def __getitem__(self, index: int) -> Setup:
        """Required implementation for indexing of a MutableSequence.

        Args:
            index (int): The index of the list for which the item should be obtained.

        Returns:
            Setup: The base class of all setups.
        """
        return self._value[self._get_checked_index(index)]

    def __setitem__(self, index: int, new_setups: Union[List[Setup], Dict[str, Any]]) -> None:
        """Required implementation for index setting of a MutableSequence.

        Args:
            index (int): The index of the list for which the item should be set.
            new_setups (Union[List[Setup], Dict[str, Any]]): The new setup value to be set.
        """
        self._value[self._get_checked_index(index)] = self._create_list_entry(new_setups)

    def __delitem__(self, index: int) -> None:
        """Required implementation for deletion of an element in a MutableSequence.

        Args:
            index (int): The index to be deleted.
        """
        del self._value[self._get_checked_index(index)]

    def __len__(self) -> int:
        """Required implementation to get the length of a MutableSequence.

        Returns:
            int: The length of the list.
        """
        return len(self._value)

    def insert(self, index: int, new_setups: Union[List[Setup], Dict[str, Any]]) -> None:
        """Required implementation for inserting a new element in a MutableSequence.

        Args:
            index (int): The index of the list at which position the item should be inserted.
            new_setups (Union[List[Setup], Dict[str, Any]]): The new setup value to be set.
        """
        self._value.insert(index, self._create_list_entry(new_setups))

    def _content(self, indent: int):
        """See base class definition."""
        return "[" + f",\n{' ' * indent}".join([entry.content(indent, False) if isinstance(entry, Setup) else str(entry) for entry in self._value]) + "]"

    def _get_checked_index(self, index: int) -> int:
        """Checks whether the index is in the bounds of the sequence.

        Args:
            index (int): The index to be checked.

        Raises:
            SetupError: If the index is out of bounds.

        Returns:
            int: The checked index.
        """
        if isinstance(index, slice):
            return index
        if not isinstance(index, int) or index > len(self) or index < 0:
            raise SetupError(f"Index {index} out of range for SetupList '{self.name}' with length {len(self)}") from None
        return index

    def _get_checked_tag(self, new_setup: List[Dict[str, Any]]):
        """See base class definition.

        Raises:
            SetupError: If the new value is not a list.
        """
        if not isinstance(new_setup, (list, tuple)):
            raise SetupError("SetupList can only be assigned with list") from None
        return [self._create_list_entry(entry) for entry in new_setup if entry is not None]

    def _create_list_entry(self, new_setup: Union[List[Setup], Dict[str, Any]]) -> Setup:
        """Creates a proper list entry for an input argument.

        Args:
            new_setup (Union[List[Setup], Dict[str, Any]]): The new value to be set.

        Raises:
            SetupError: If a list is given and not all elements are of type Setup.
            SetupError: If the input argument ist not of type dict or list.

        Returns:
            Setup: The fully created SetupTag class.
        """
        # Copy the entry tag to a local variable
        new_list_entry = deepcopy(self._entry_tag)
        # If the entry tag is a single setup differ to SetupSets. The same applies if the list entry is a single setup tag
        if isinstance(new_setup, Setup) or isinstance(new_list_entry, SetupTag):
            new_list_entry.value = new_setup.value if isinstance(new_setup, SetupTag) else new_setup
        else:
            if isinstance(new_setup, list):
                for entry in new_setup:
                    if not isinstance(entry, Setup):
                        raise SetupError("Cannot set value. Type is not a derived class of Setup.") from None
                    new_list_entry[entry.name] = entry
            elif isinstance(new_setup, dict):
                for name, tag_value in new_setup.items():
                    if isinstance(tag_value, Setup):
                        new_list_entry[name] = tag_value
                    else:
                        new_list_entry[name].value = tag_value
            else:
                raise SetupError("You can only set the SetupList value with a list of Setups or a dict.") from None
        return new_list_entry

    def entry_names(self) -> List[str]:
        """Gives the names of all list entries. This function called to get indexed names. Otherwise each entry will have the same name.

        Returns:
            List[str]: The list with all names of the entries.
        """
        return [f"{self._entry_tag.name}{index}" for index, entry in enumerate(self.value, start=1)]

    @property
    def entry_tag(self) -> Setup:
        """Gives the entry tag each list element is built of.

        Returns:
            Setup: The entry setup.
        """
        return self._entry_tag
