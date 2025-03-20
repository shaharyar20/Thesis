#!/usr/bin/env python3
# Python modules
from typing import List, Tuple
import xml.etree.ElementTree as et
from pathlib import Path
from copy import deepcopy

# torchlbm modules
import torchlbm.standalone_operations.xml_operations as xml_o
from torchlbm.setup_definitions import SetupName
from torchlbm.setup_definitions.setup_handlers import SetupHandler
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions.setup_types import Setup, SetupTag, SetupList, SetupVariableSet


class XMLSetupHandler(SetupHandler):
    """The XMLSetupHandler is specialization of the SetupHandler class to read/write setups from to xml file.

    The XMLSetupHandler handles reading and writing proccesses for a setup. For a given setup, the
    full set is written to file or read from it. The reading is type safe due to the inherit correct definition of the SetupTags with unique names.

    Args:
        SetupHandler (SetupHandler): The base class of all setup handlers.
    """

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()

    def _add_tag_to_tree(self, xml_element: et.Element, setup: Setup, use_set_values_only: bool) -> None:
        """Recursive function to write sub elements to an existing element.

        Args:
            xml_element (et.Element): The parent element to which the child tags are added (indirect return).
            setup (Setup): The setup that should be added. The element for this setup must already exist.
        """
        # If the setup is a last layer setup, add the data to the element
        if isinstance(setup, SetupTag):
            if isinstance(setup.value, Setup):
                self._add_tag_to_tree(xml_element, setup.value, use_set_values_only)
            elif isinstance(setup.value, (list, tuple)):
                xml_element.text = ", ".join([str(entry) for entry in setup.value])
            else:
                xml_element.text = str(setup.value)
        else:
            # For setup lists differ in behavior
            if isinstance(setup, SetupList):
                list_names = setup.entry_names()
                for name, child_tag in zip(list_names, setup.value):
                    if not isinstance(child_tag, Setup):
                        new_xml_element = et.SubElement(xml_element, SetupName.Xml.format(name))
                        new_xml_element.text = str(child_tag)
                    elif (child_tag.is_set() and use_set_values_only) or not use_set_values_only:
                        new_xml_element = et.SubElement(xml_element, SetupName.Xml.format(name))
                        self._add_tag_to_tree(new_xml_element, child_tag, use_set_values_only)
            else:
                # Define all child tags to be added
                child_tags = [setup.value] if isinstance(setup, SetupTag) else setup.value
                # Call this function recursively for all child tags
                for child_tag in child_tags:
                    if (child_tag.is_set() and use_set_values_only) or not use_set_values_only:
                        # Create the element for this child tag
                        new_xml_element = et.SubElement(xml_element, SetupName.Xml.format(child_tag.name))
                        # Write the child tag
                        self._add_tag_to_tree(new_xml_element, child_tag, use_set_values_only)

    def _read_tag_from_tree(self, xml_element: et.Element, setup: Setup) -> Tuple[str, List[str]]:
        """Recursive function to read the data for a given setup from the xml Element.

        Args:
            xml_element (et.Element): The xml element from which the data should be read. The element must already point to the correct element in the tree.
            setup (Setup): The setup that should be read.

        Returns:
            Tuple[str, List[str]]: The error message and list of setups (Used for backward error propagation).
        """
        # If the setup is a last layer setup, add the data to the element
        if isinstance(setup, SetupTag):
            try:
                setup.value = xml_o.read_xml_text(xml_element)
                return ("", [])
            except SetupError as err:
                return (str(err), [setup.name])
        else:
            # NOTE: For list elements, no check is done that the tags in xml-tree are valid tags. As long they can be converted to the list element.
            if isinstance(setup, SetupList):
                # Get the set that needs to be filled for each list entry
                list_entry_tag = setup.entry_tag
                # For each sub element carry out the reading process
                final_data_list = []
                for sub_element in xml_element:
                    # Check if the name of the sub element coincides with the list entry name and all other letters are numbers only
                    list_entry_name = SetupName.LowerCase.format(list_entry_tag.name)
                    sub_element_name = SetupName.LowerCase.format(sub_element.tag)
                    if list_entry_name not in sub_element_name or any([not char.isdigit() for char in sub_element_name.replace(list_entry_name, "")]):
                        return (
                            f"List tag '{sub_element.value}' incorrect for list with tags '{list_entry_tag.name}'",
                            [sub_element.tag, setup.name],
                        )
                    else:
                        err_msg, tag_list = self._read_tag_from_tree(sub_element, list_entry_tag)
                        if err_msg:
                            return (err_msg, tag_list + [setup.name])
                        final_data_list.append(deepcopy(list_entry_tag.value))
                setup.value = final_data_list
                return ("", [])
            else:
                # Define all child tags to be read (differ between SetupTag and the more complex)
                child_tags = [setup.value] if isinstance(setup, SetupTag) else setup.all_values() if isinstance(setup, SetupVariableSet) else setup.value
                # Call this function recursively for all child tags
                err_msg, tag_list = ("", [])
                final_tag_list = []
                for child_tag in child_tags:
                    # To make the reading type safe, check if only one single child element can be matched to the required one
                    child_lower_case_name = SetupName.LowerCase.format(child_tag.name)
                    child_elements = [child for child in xml_element if SetupName.LowerCase.format(child.tag) == child_lower_case_name]
                    if not child_tag.is_required() and len(child_elements) == 0:
                        continue
                    if len(child_elements) == 0:
                        err_msg, tag_list = (
                            f"Child tag {child_tag.name} for {setup.name} does not exist!",
                            [child_tag.name],
                        )
                    elif len(child_elements) > 1:
                        err_msg, tag_list = (
                            f"Child tag {child_tag.name} for {setup.name} exists multiple times!",
                            [child_tag.name],
                        )
                    else:
                        err_msg, tag_list = self._read_tag_from_tree(child_elements[0], child_tag)
                        final_tag_list.append(deepcopy(child_tag))
                    if err_msg:
                        break
                setup.value = final_tag_list
                return (err_msg, tag_list + [setup.name])

    def _remove_empty_tags(self, xml_element: et.Element) -> bool:
        """Removes all tags that do not contain any sub element.

        Args:
            xml_element (et.Element): The xml element that need to be checked.
        """
        # Loop through all child elements to check them. List first to remove iteratively afterwards
        child_to_remove = [child_element for child_element in xml_element if self._remove_empty_tags(child_element)]
        for child_element in child_to_remove:
            xml_element.remove(child_element)
        # Return if the full element is empty now (no text and not subelements) (must be done afterwards)
        return not xml_element.text and len(xml_element) == 0

    def _read_from_file(self, file_path: Path, setup: Setup) -> None:
        """See base class definition.

        Raises:
            SetupError: If the reading process fails (missing elements, incorrect data types, ...).
        """
        # Open the tree from the file. Do not use comments here, since those cannot be used at all.
        err_msg = ""
        try:
            root_element = xml_o.open_xml_file(file_path, parse_comments=False).getroot()
        except IOError as err:
            err_msg = str(err)
        # Read all tags from the tree
        if not err_msg:
            err_msg, tag_list = self._read_tag_from_tree(root_element, setup)
            err_msg = f"\n{err_msg}\nTag hierarchy: { ' / '.join([tag for tag in reversed(tag_list)])}" if err_msg else ""
        if err_msg:
            raise SetupError(err_msg) from None

    def _write_to_file(self, setup: Setup, file_path: Path, use_set_values_only: bool) -> None:
        """See base class definition."""
        # Create the root element and call the recursive function
        root_element = et.Element(SetupName.Xml.format(setup.name))
        self._add_tag_to_tree(root_element, setup, use_set_values_only)
        # Remove all tags that are empty
        self._remove_empty_tags(root_element)
        # Create the tree from the elements. Format it and write it to file.
        tree = et.ElementTree(root_element)
        root = tree.getroot()
        xml_o.pretty_print_xml_tree(root, level_indent=3)
        tree.write(file_path)
