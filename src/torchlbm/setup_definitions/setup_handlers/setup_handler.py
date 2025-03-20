#!/usr/bin/env python3
# Python modules
from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod

# torchlbm modules
from torchlbm.setup_definitions import SetupName
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions.setup_types import Setup, SetupTag, SetupSet, SetupList, SetupVariableSet


class SetupHandler(ABC):
    """The base class for all setup handlers (readers/writers).

    It gives the basic interface to set default values, replace to setups with each other or read and write a setup from/to file.
    Some method must be implemented by all derived classes:
        - _write_to_file : Provides the interface to write setups to a file.
        - _read_from_file: Provides the interface to read setup data from a file.

    Args:
        ABC (ABC): This class is an Abstract Base Class and cannot be instantiated without implementing the required functions.
    """

    def __init__(self) -> None:
        """Constructor."""
        pass

    def __str__(self) -> str:
        """The built-in str method.

        Returns:
            str: The name of this class a string.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        """The built-in repr method.

        Returns:
            str: The name of this class a string.
        """
        return self.__str__()

    def read_from_file(self, file_path: Path, setup: Setup) -> None:
        """The read function that reads content from a specific file.

        Args:
            file_path (Path): The ABSOLUTE path to the file that should be read.
            setup (Setup): The setup that should be filled.
        """
        try:
            self._read_from_file(file_path, setup)
        except SetupError as err:
            raise SetupError(f"Error reading data from file {file_path}\n{str(err)}") from None

    def write_to_file(
        self,
        setup: Setup,
        file_path: Path,
        enforce_defaults: bool = False,
        use_set_values_only: bool = False,
    ) -> None:
        """Writes the content of all setups to the given file. Here, basic checks are done that the setup is properly set.
            The actual writing function needs to be implemented by the derived classes.

        Args:
            setup (Setup): The setup SetupSet that should be written.
            file_path (Path): The ABSOLUTE path to the file where the data is written to.
            enforce_defaults (bool): Flag whether default values should be enforced for not set required values.
            use_set_values_only (bool): Flag whether only set values should be printed to file.

        Raises:
            SetupError: If tags exist that are required, but not set.
        """
        # Use the set values flag only if enforce defaults is not set and flag is set. If both are true the enforce defaults is prioritized.
        set_values_only = not enforce_defaults and use_set_values_only
        if not self.is_valid(setup) and not set_values_only:
            if enforce_defaults:
                self.set_defaults(setup, False)
            else:
                raise SetupError("Setup incomplete. Some values are not set. Cannot write to file.") from None
        try:
            self._write_to_file(setup, file_path, set_values_only)
        except SetupError as err:
            raise SetupError(f"Error writing data to file {file_path}\n{str(err)}") from None

    def set_defaults(self, setup: Setup, overwrite: bool = False) -> None:
        """Sets default values to all tags that are required, but not set. Allows overwriting set values with default.

        Args:
            setup (Setup): The setup where the defaults should be set.
            overwrite (bool, optional): Flag whether set tags should be overwritten. Defaults to False.

        Note:
            This function may be modified. The use of body function _set_defaults_recursively() is highly recommended, since it provides the recursive
            loop to modify the full setup.
        """
        # Call the recursive function only
        self._set_defaults_recursively(setup, overwrite)

    def replace(self, ref_setup: Setup, new_setup: Setup) -> None:
        """Replaces a reference setup with a new setup. Only the set values of the new setup are taken into account.

        Args:
            ref_setup (Setup): The reference setup, where the new data should be set.
            new_setup (Setup): The setup where the set data is taken from.

        Raises:
            SetupError: If both setups are of distinct type.

        Note:
            This function may be modified. The use of body function _replace_recursively() is highly recommended, since it provides the recursive
            loop to modify the full setup.
        """
        # Check that both setups are of same type
        if not (type(ref_setup) == type(new_setup)):
            raise SetupError("Cannot replace setups of distinct type.") from None
        # Call the recursive function only
        self._replace_recursively(ref_setup, new_setup)

    def is_valid(self, setup: Setup) -> bool:
        """Checks whether the setup is valid, i.e. all required tags are set and if the data tags have not been overwritten accidentally by other
            python-types.

        Args:
            setup (Setup): The setup to be checked.

        Returns:
            bool: True if all tags are set properly.

        Note:
            Do not overwrite this method.
        """
        if isinstance(setup, SetupTag):
            return setup.is_valid()
        elif isinstance(setup, SetupList) and any([not isinstance(entry, Setup) for entry in setup.value]):
            return True
        elif isinstance(setup, Setup):
            all_are_valid = True
            for setup_tag in setup.value:
                all_are_valid = self.is_valid(setup_tag) and all_are_valid
                if not all_are_valid:
                    return False
            return all_are_valid
        else:
            raise SetupError(
                "Each value in the setup must be of type 'Setup', maybe you have accidentally overwritten\n"
                "Use .value to assign values to an existing SetupTag."
            ) from None

    @abstractmethod
    def _read_from_file(self, file_path: Path, setup: Setup) -> None:
        """The read function that needs to be implemented by all derived classes.

        Args:
            file_path (Path): The ABSOLUTE path to the file that should be read.
            setup (Setup): The setup that should be filled.

        Raises:
            SetupError: Not implemented yet.
        """
        pass

    @abstractmethod
    def _write_to_file(self, setup: Setup, file_path: Path, use_set_values_only: bool) -> None:
        """The write function that needs to be implemented by all derived classes.

        Args:
            setup (Setup): The setup SetupSet that should be written.
            file_path (Path): The ABSOLUTE path to the file where the data is written to.
            use_set_values_only (bool): Flag whether only set values should be printed to file.

        Raises:
            SetupError: Not implemented yet.
        """
        pass

    def _set_defaults_recursively(self, setup: Setup, overwrite: bool) -> None:
        """Recursive function that sets all default values. See details in set_defaults().

        Note:
            Do not overwrite this method, since it provides the generic interface to set defaults for all setups recursively.
        """
        # For tags and list we can immediately set the default values. For lists it will be the empty list.
        if isinstance(setup, (SetupTag, SetupList)):
            if not setup.is_valid() or not setup.is_set() or overwrite:
                setup.value = setup.default
        elif isinstance(setup, SetupSet):
            # We take the default tags and check if they already exist in the current set values
            default_tags = setup.default
            default_names = [SetupName.LowerCase.format(tag.name) for tag in default_tags]
            set_tags = setup.value
            set_names = [SetupName.LowerCase.format(tag.name) for tag in set_tags]
            if overwrite:
                tags_to_set = default_tags + [tag for name, tag in zip(set_names, set_tags) if name not in default_names]
            else:
                tags_to_set = set_tags + [tag for name, tag in zip(default_names, default_tags) if name not in set_names]
            # Create the final list that is added
            final_tag_list = []
            for setup_tag in tags_to_set:
                self._set_defaults_recursively(setup_tag, overwrite)
                final_tag_list.append(deepcopy(setup_tag))
            setup.value = final_tag_list

    def _replace_recursively(self, ref_setup: Setup, new_setup: Setup) -> None:
        """Recursive function that replaces values between two setups. See details in replace().

        Note:
            Do not overwrite this method, since it provides the generic interface to replace all setups recursively.
        """
        # We make a safety check that both are of same type
        if not (type(ref_setup) == type(new_setup)):
            raise SetupError("Recursive error: Cannot replace setups of distinct type.") from None
        # For setup tags we can set the values immediately
        if isinstance(new_setup, (SetupTag, SetupList)):
            if new_setup.is_set():
                ref_setup.value = new_setup.value
        elif isinstance(new_setup, SetupSet):
            if isinstance(new_setup, SetupVariableSet):
                # Loop through all tags of the new setup (only those are present) and add them in the ref
                for tag in new_setup.value:
                    if tag.is_set():
                        ref_setup[tag.name].value = tag.value
            else:
                # Loop through all tags of both sets
                for ref_tag, new_tag in zip(ref_setup.values(), new_setup.values()):
                    self._replace_recursively(ref_tag, new_tag)
