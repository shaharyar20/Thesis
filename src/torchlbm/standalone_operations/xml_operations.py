#!/usr/bin/env python3
# Python modules
import os
import re
import xml.etree.ElementTree as et

# torchlbm modules
from . import string_operations as string_o


def open_xml_file(xml_file_path: str, parse_comments: bool = True) -> et:
    """Opens a xml file and throws an appropriate error if not possible. A special tree builder class is used that preserves
        comments in the xml file. Without this class, comments would not be parsed.

    Args:
        xml_file_path (str): The xml_file_path of the xml that should be opened.
        parse_comments (bool): Flag whether comments should be parsed or not.

    Raises:
        IOError: If there are syntax errors in the file, etc.

    Returns:
        xml.etree.ElementTree: The xml element Tree.
    """

    class TreeBuildWithCommens(et.TreeBuilder):
        """Taken from https://stackoverflow.com/questions/33573807/faithfully-preserve-comments-in-parsed-xml"""

        def comment(self, data):
            self.start(et.Comment, {})
            self.data(data)
            self.end(et.Comment)

    try:
        if parse_comments:
            return et.parse(xml_file_path, parser=et.XMLParser(target=TreeBuildWithCommens()))
        else:
            return et.parse(xml_file_path)
    except et.ParseError as error:
        raise IOError(f"Error parsing xml file '{xml_file_path}':\n{str(error)}") from None


def pretty_print_xml_text(xml_text: str, indent: int, level_indent: int) -> str:
    """Strips off all spaces and new line characters from a xml text. It adds indents, white spaces and new lines to get a consistent appearance. If new lines
        are present inside the text, they are kept. Single leading and/or trailing whitespaces are removed.

    Args:
        xml_text (str): The xml text to be modified.
        indent (int): The total indent a line should have if new line characters are present.
        level_indent (int): The indent a single level adds to a xml tag.

    Returns:
        str: The stripped xml text.
    """
    # Strip all leading and trailing spaces and line breaks from the string
    tmp_string = xml_text.strip()
    # All new lines have been stripped before and after the string. If an intermediate newline exists, add again a leading and trailing white space.
    # Furthermore, add the correct indent to all newlines.
    if "\n" in tmp_string:
        tmp_string = "\n" + tmp_string
        tmp_string = re.sub(" *\n *", "\n", tmp_string).replace("\n", "\n" + str(indent))
        tmp_string += "\n" + (len(indent) - level_indent) * " "
    else:
        # Otherwise add a single leading and trailing white spaces
        tmp_string = " " + tmp_string + " "
    return tmp_string


def pretty_print_xml_tree(
    xml_element: et.Element,
    level: int = 1,
    is_last_child: bool = False,
    level_indent: int = 1,
    strip_text: bool = True,
) -> None:
    """Prints a xml tree in pretty format, where each line is indented by a certain value.

    Args:
        xml_element (et.Element): The root element of the xml tree to be modified (in-place modification).
        level (int, optional): The level of the current element under investigation (do not change the default value). Defaults to 1.
        is_last_child (bool, optional): Flag whether the current element is the last element under all other subelements. Defaults to False.
        level_indent (int, optional): The indent that is desired per level. Defaults to 1.
        strip_text (bool, optional): Flag whether text stripping should be applied on elements. Use this flag if you have not modified the elements in advance.
                                     Defaults to True.

    Raises:
        ValueError: If the level is specified below 1.
    """
    # Consistency check
    if level < 1:
        raise ValueError("The level must not be smaller than 1") from None

    # Define the indent of the current level
    indent = level * level_indent * " "
    # Check if the element has sub elements
    if len(xml_element):
        # Append the indent to the current element text and strip all line skips and spaces from the string
        if xml_element.text is not None:
            xml_element.text = (pretty_print_xml_text(xml_element.text, indent, level_indent) if strip_text else xml_element.text) + os.linesep + indent
        else:
            xml_element.text = os.linesep + indent
        # Add the tail to the current element (depending on the number of the current element beyond all other subelements)
        if is_last_child:
            indent = os.linesep + (level - 2) * level_indent * " " if level > 1 else ""
        else:
            indent = os.linesep + (level - 1) * level_indent * " "
            if level == 2:
                indent = 2 * indent
        xml_element.tail = indent

        # Loop through all sub elements and carry out the printing process
        for number, sub_element in enumerate(xml_element):
            pretty_print_xml_tree(
                sub_element,
                level + 1,
                number + 1 == len(xml_element),
                level_indent,
                strip_text,
            )
    else:
        # If it is the final element strip all linebreaks and spaces before and after the actual value and add single leading and trailing white space
        xml_element.text = (
            pretty_print_xml_text(
                xml_element.text if xml_element.text is not None else "",
                indent,
                level_indent,
            )
            if strip_text
            else xml_element.text if xml_element.text is not None else ""
        )
        if is_last_child:
            indent = os.linesep + (level - 2) * level_indent * " " if level > 1 else ""
        else:
            indent = os.linesep + (level - 1) * level_indent * " " if level > 1 else ""
        xml_element.tail = indent


def read_xml_text(xml_element: et.Element) -> str:
    """Reads the text of a single scalar value of an xml element. Consideres conversion to None.

    Args:
        xml_element (et.Element): The xml element from which the text should be read.

    Returns:
        str: The read string, None if it is None.
    """
    stripped_text = xml_element.text.strip()
    return None if not stripped_text or string_o.is_none(stripped_text) else stripped_text
