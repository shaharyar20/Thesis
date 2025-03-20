#!/usr/bin/env python3
# Python modules
from typing import Dict, Any
from omegaconf import OmegaConf
from pathlib import Path

# torchlbm modules
from torchlbm.setup_definitions import SetupName
from torchlbm.setup_definitions.setup_handlers import SetupHandler
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions.setup_types import Setup, SetupTag, SetupList


class YAMLSetupHandler(SetupHandler):
    """The YAMLSetupHandler is specialization of the SetupHandler class to read/write setups from to yaml file.

    The YAMLSetupHandler handles reading and writing proccesses for a setup. For a given setup, the
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
                    # Loop through all elements of the list value (YAML does not require to print the name of each tag value).
                    if not isinstance(list_value, Setup):
                        new_list_entry = str(list_value)
                    elif isinstance(list_value, SetupTag):
                        new_list_entry = self._create_dict_value(list_value, use_set_values_only)
                    else:
                        new_list_entry = {}
                        for setup_tag in list_value.values():
                            if (setup_tag.is_set() and use_set_values_only) or not use_set_values_only:
                                new_list_entry[SetupName.Yaml.format(setup_tag.name)] = self._create_dict_value(setup_tag, use_set_values_only)
                    tag_list.append(new_list_entry)
                return tag_list
            else:
                new_yaml_dict = {}
                for setup_tag in setup.value:
                    if (setup_tag.is_set() and use_set_values_only) or not use_set_values_only:
                        new_yaml_dict[SetupName.Yaml.format(setup_tag.name)] = self._create_dict_value(setup_tag, use_set_values_only)
                return new_yaml_dict

    def _check_dict(self, yaml_dict: Dict[str, Any], parent_key: str = None) -> bool:
        """Checks whether the given yaml dict is appropriate to be used for setting to setups. Only checks the keys. The values are
            check during setting to setups.

        Args:
            yaml_dict (Dict[str, Any]): The yaml dict to be checked.
            parent_key (str, optional): The parent key for which the dict is checked. Defaults to None.

        Raises:
            SetupError: If the keys are not valid.

        Returns:
            bool: True if everything is alright.
        """
        # Check if the keys are valid converted to lower case letters
        if any([not isinstance(key, str) for key in yaml_dict]):
            raise SetupError("YAML dict must contain only strings as keys.") from None
        keys_lower_case = set([SetupName.LowerCase.format(key) for key in yaml_dict])
        if len(keys_lower_case) != len(yaml_dict.keys()):
            raise SetupError(
                f"Key '{parent_key}' in YAML dict contains not only unique key elements.\n" f"Check key elements: {list(yaml_dict.keys())}"
            ) from None
        # Loop through all childs to check if the next layer is correct. Differ for lists and non-lists
        for key, value in yaml_dict.items():
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
        with open(file_path, "r") as yaml_file:
            yaml_conf = OmegaConf.load(yaml_file)
        yaml_dict = OmegaConf.to_container(yaml_conf)
        # Check that all keys fulfill the requirements
        self._check_dict(yaml_dict)
        # Get the correct value for the initial setup
        setup_name = SetupName.LowerCase.format(setup.name)
        dict_keys = {SetupName.LowerCase.format(key): key for key in yaml_dict}
        if setup_name not in dict_keys:
            raise SetupError(f"Cannot find '{setup.name}' in YAML dict.") from None
        # Read all values from the dict (this is done recursively by the SetupTags)
        setup.value = yaml_dict[dict_keys[setup_name]]

    def _write_to_file(self, setup: Setup, file_path: Path, use_set_values_only: bool) -> None:
        """See base class definition."""
        # Create the root element of the dictionary that is stored.
        if (setup.is_set() and use_set_values_only) or not use_set_values_only:
            yaml_dict = {SetupName.Yaml.format(setup.name): self._create_dict_value(setup, use_set_values_only)}
        # Write the json dict to file
        with open(file_path, "w") as yaml_file:
            conf = OmegaConf.create(yaml_dict)
            OmegaConf.save(config=conf, f=yaml_file)
