#!/usr/bin/env python3
# Python modules
from typing import Dict, Any
import json
from pathlib import Path

# torchlbm modules
from torchlbm.setup_definitions import SetupName
from torchlbm.setup_definitions.setup_handlers import SetupHandler
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions.setup_types import Setup, SetupTag, SetupList


class JSONSetupHandler(SetupHandler):
    """The JSONSetupHandler is specialization of the SetupHandler class to read/write setups from to json file.

    The JSONSetupHandler handles reading and writing proccesses for a setup. For a given setup, the
    full set is written to file or read from it. The reading is type safe due to the inherit correct definition of the SetupTags with unique names.

    Args:
        SetupHandler (SetupHandler): The base class of all setup handlers.
    """

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()

    def _create_dict_value(self, setup: Setup, use_set_values_only: bool) -> Any:
        """Recursive function to write sub elements to an existing element.

        Args:
            setup (Setup): The setup that should be added.
        Returns:
            Any: The dict value.
        """
        # If the setup is a last layer setup, add the data to the element
        if isinstance(setup, SetupTag):
            return setup.value if isinstance(setup.value, (list, tuple, bool, int, float, str)) or setup.value is None else str(setup.value)
        else:
            # Depending on the setup type differ the execution
            if isinstance(setup, SetupList):
                # Add each tag individually to the list of elements
                tag_list = []
                for list_value in setup.value:
                    # Loop through all elements of the list value (JSON does not require to print the name of each tag value).
                    if not isinstance(list_value, Setup):
                        new_list_entry = str(list_value)
                    elif isinstance(list_value, SetupTag):
                        new_list_entry = self._create_dict_value(list_value, use_set_values_only)
                    else:
                        new_list_entry = {}
                        for setup_tag in list_value.values():
                            if (setup_tag.is_set() and use_set_values_only) or not use_set_values_only:
                                new_list_entry[SetupName.Json.format(setup_tag.name)] = self._create_dict_value(setup_tag, use_set_values_only)
                    tag_list.append(new_list_entry)
                return tag_list
            else:
                new_json_dict = {}
                for setup_tag in setup.value:
                    if (setup_tag.is_set() and use_set_values_only) or not use_set_values_only:
                        new_json_dict[SetupName.Json.format(setup_tag.name)] = self._create_dict_value(setup_tag, use_set_values_only)
                return new_json_dict

    def _check_dict(self, json_dict: Dict[str, Any], parent_key: str = None) -> bool:
        """Checks whether the given json dict is appropriate to be used for setting to setups. Only checks the keys. The values are
            check during setting to setups.

        Args:
            json_dict (Dict[str, Any]): The json dict to be checked.
            parent_key (str, optional): The parent key for which the dict is checked. Defaults to None.

        Raises:
            SetupError: If the keys are not valid.

        Returns:
            bool: True if everything is alright.
        """
        # Check if the keys are valid converted to lower case letters
        if any([not isinstance(key, str) for key in json_dict]):
            raise SetupError("JSON dict must contain only strings as keys.") from None
        keys_lower_case = set([SetupName.LowerCase.format(key) for key in json_dict])
        if len(keys_lower_case) != len(json_dict.keys()):
            raise SetupError(
                f"Key '{parent_key}' in JSON dict contains not only unique key elements.\n" f"Check key elements: {list(json_dict.keys())}"
            ) from None
        # Loop through all childs to check if the next layer is correct. Differ for lists and non-lists
        for key, value in json_dict.items():
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict):
                        self._check_dict(entry, key)
            elif isinstance(value, dict):
                self._check_dict(value, key)

    def _read_from_file(self, file_path: Path, setup: Setup) -> None:
        """See base class definition.

        Raises:
            SetupError: If the reading process fails (missing elements, incorrect data types, ...).
        """
        # Read the json dict from file
        with open(file_path, "r") as json_file:
            json_dict = json.load(json_file)
        # Check that all keys fulfill the requirements
        self._check_dict(json_dict)
        # Get the correct value for the initial setup
        setup_name = SetupName.LowerCase.format(setup.name)
        dict_keys = {SetupName.LowerCase.format(key): key for key in json_dict}
        if setup_name not in dict_keys:
            raise SetupError(f"Cannot find '{setup.name}' in JSON dict.") from None
        # Read all values from the dict (this is done recursively by the SetupTags)
        setup.value = json_dict[dict_keys[setup_name]]

    def _write_to_file(self, setup: Setup, file_path: Path, use_set_values_only: bool) -> None:
        """See base class definition."""
        # Create the root element of the dictionary that is stored.
        if (setup.is_set() and use_set_values_only) or not use_set_values_only:
            json_dict = {SetupName.Json.format(setup.name): self._create_dict_value(setup, use_set_values_only)}
        # Write the json dict to file
        with open(file_path, "w") as json_file:
            json.dump(json_dict, json_file, indent=3, sort_keys=False)
