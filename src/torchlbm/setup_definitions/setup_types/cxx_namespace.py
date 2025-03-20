#!/usr/bin/env python3
# Python modules
from typing import List, Tuple, Union
from copy import deepcopy
import re

# torchlbm modules
from torchlbm.exceptions import SetupError
from torchlbm.setup_definitions.setup_types import SetupSet, CxxVariable


class CxxNamespace(SetupSet):
    """The CxxNamespace class represents a specialization of the SetupSet to be used in C++ files.

    Beside the basic SetuPSet functionality, it adds functionality to read and replace variables in a c++ namespace.
    This class provides the functionality to set variables inside specific namespaces. A namespace can be nested.

    Args:
        SetupSet (SetupSet): This class is a SetupSet.
    """

    def __init__(
        self,
        name: str,
        namespace_identifier: str,
        cxx_tags: List[Union["CxxNamespace", CxxVariable]] = [],
    ) -> None:
        """Constructor.

        Args:
            name (str): The name of the namespace for internal referencing.
            namespace_identifier (str): The namespace identifier. Empty string for empty namespaces, None to prevent namespace functionality.
            cxx_tags (List[Union[, optional): The Cxx variables and nested namespaces. Defaults to [].

        Raises:
            SetupError: If the cxx tags are not of type CxxNamespace or CxxVariable.
            SetupError: if the namespace identifier is not of type string.
        """
        # Check that the cxx tags are of correct type
        if any([not isinstance(cxx_tag, (CxxNamespace, CxxVariable)) for cxx_tag in cxx_tags]):
            raise SetupError(f"{self.__class__.__name__} must only contain CxxNamespace or CxxVariables.") from None
        if namespace_identifier is not None and not isinstance(namespace_identifier, str):
            raise SetupError("Cxx namespace identifier must be of type 'str'.") from None
        if any([cxx_tag.identifier() is None for cxx_tag in cxx_tags if isinstance(cxx_tag, CxxNamespace)]):
            raise SetupError("Nested Cxx namespaces cannot have None as identifier.") from None
        # Call the base class
        super().__init__(name, [deepcopy(cxx_tag) for cxx_tag in cxx_tags])
        # Assign additional variables
        self._identifier = namespace_identifier

    def identifier(self) -> str:
        """Gives the namespace identifier.

        Returns:
            str: The namespace identifier.
        """
        return f"namespace {self._identifier}" if self._identifier is not None else None

    def _get_matched_braces(self, content: str) -> List[Tuple[int, int]]:
        """Gives all matched braces positions found in the c++ content.

        Args:
            content (str): The content where the braces should be obtained from.

        Raises:
            SetupError: If the opening and closing braces are not consistent.

        Returns:
            List[Tuple[int,int]]: A list with all indices of matched braces. Sorted based on the start index.
        """
        # Obtain the positions of all opening and closing braces
        opening_brace_positions = [(brace.start(), True) for brace in re.finditer(r"\{", content)]
        closing_brace_positions = [(brace.start(), False) for brace in re.finditer(r"\}", content)]
        # Check that opening and closing braces are consistent
        if len(opening_brace_positions) != len(closing_brace_positions):
            raise SetupError(f"Opening and closing braces mismatch for namespace {self.name}") from None
        # Sort all braces by their position
        sorted_list = sorted(opening_brace_positions + closing_brace_positions, key=lambda tup: tup[0])
        # Loop through the sorted list and get all start and end positions for matching braces. A brace match is indicated when an open brace is followed
        # by and closing brace. Remove those matched iteratively to get all matching braces
        start_end_list = []
        while sorted_list:
            # Loop through all elements in the sorted list (except last)
            for index, element in enumerate(sorted_list[:-1]):
                # If two subsequent elements are different add them to the final list
                if element is not None and element[1] and not sorted_list[index + 1][1]:
                    start_end_list.append((element[0], sorted_list[index + 1][0]))
                    sorted_list[index] = None
                    sorted_list[index + 1] = None
            # Remove all None flagged elements
            sorted_list = [element for element in sorted_list if element is not None]
        # Return the brace positions in sorted order based on the start position
        return sorted(start_end_list, key=lambda tup: tup[0])

    def _get_namespace_start_end(self, brace_positions: List[Tuple[int, int]], namespace_start: int) -> Tuple[int, int]:
        """Gives the start and end indices for a namespace start position based on the brace positions.

        Args:
            brace_positions (List[Tuple[int,int]]): The list of matched braces indices.
            namespace_start (int): The index where the namespace starts.

        Raises:
            SetupError: If no possible combination can be found for the namespace start index based on the brace positions.

        Returns:
            Tuple[int,int]: The namespace start and end.
        """
        start_end_list = [(namespace_start, start_end[1] + 1) for start_end in brace_positions if start_end[0] > namespace_start]
        if len(start_end_list) == 0:
            raise SetupError("Namespace start and end cannot be determined. Fatal error.") from None
        return start_end_list[0]

    def _get_inner_spaces_positions(self, content: str, inner_spaces: List["CxxNamespace"]) -> List[Tuple["CxxNamespace", int, int]]:
        """Gives the positions of the innerspaces in ascending order.

        Args:
            content (str): The content where the inner spaces should be obtained from.
            content_list (List['CxxNamespace']): A list with inner spaces that should be obtained.

        Raises:
            SetupError: If any namespace identifier cannot be found in the content.
            SetupError: If the namespace nesting is not set properly.

        Returns:
            List[Tuple['CxxNamespace', int, int]]: A list with all inner spaces and there start and end positions in the content string.
        """
        # Get the start indices of the current namespace and all inner namespaces
        this_space_start = (self.identifier(), content.find(self.identifier())) if self.identifier() is not None else (None, 0)
        inner_spaces_start = [(space.identifier(), content.find(space.identifier())) for space in inner_spaces]
        # Check that all spaces are found
        for space_start in inner_spaces_start + [this_space_start]:
            if space_start[1] == -1:
                raise SetupError(f"Cannot find namespace identifier '{space_start[0]}' in file content.") from None
            elif space_start[1] < this_space_start[1]:
                raise SetupError(
                    f"Identifier '{space_start[0]}'' found in front of '{self._identifier}''. Maybe the CxxNamespace is not set properly."
                ) from None
        # Get start and end positions of the namespaces based on the brace positions
        brace_positions = self._get_matched_braces(content)
        this_space_start_and_end = self._get_namespace_start_end(brace_positions, this_space_start[1]) if self.identifier() is not None else (0, len(content))
        inner_space_start_and_end = [
            (space,) + self._get_namespace_start_end(brace_positions, space_start[1]) for space, space_start in zip(inner_spaces, inner_spaces_start)
        ]
        # Return the inner space and positions in sorted order based on the start index
        return this_space_start_and_end, sorted(inner_space_start_and_end, key=lambda tup: tup[1])

    def replace(self, content: str) -> str:
        """Replaces all data of this namespace in the given content.

        Args:
            content (str): The content that should be replaced.

        Returns:
            str: The replaced content.
        """
        # Obtain all namespaces and cxx tags that need to be used currently
        inner_spaces = [cxx_tag for cxx_tag in self.value if isinstance(cxx_tag, CxxNamespace)]
        cxx_tags_to_use = [cxx_tag for cxx_tag in self.value if isinstance(cxx_tag, CxxVariable)]
        # Differ between existing namespaces and only existing variables
        if inner_spaces:
            # Get the start and end positions for this and the inner spaces
            (
                this_space_start_and_end,
                inner_space_start_and_end,
            ) = self._get_inner_spaces_positions(content, inner_spaces)
            inner_space_start_and_end += [(None, this_space_start_and_end[1], -1)]
            # Replace the content before the first namespace
            new_content = content[this_space_start_and_end[0] : inner_space_start_and_end[0][1]]
            for cxx_tag in cxx_tags_to_use:
                new_content = cxx_tag.replace(new_content)
            content_list = [new_content]
            # Loop through all inner namespaces and substitute the data
            for current_space, next_space in zip(inner_space_start_and_end[:-1], inner_space_start_and_end[1:]):
                # Replace the content for this namespace
                content_list.append(current_space[0].replace(content[current_space[1] : current_space[2]]))
                # Replace the content between the current and the next namespace
                new_content = content[current_space[2] : next_space[1]]
                for cxx_tag in cxx_tags_to_use:
                    new_content = cxx_tag.replace(new_content)
                content_list.append(new_content)
            # Return the joined string of the content list
            return "".join(content_list)
        else:
            new_content = content
            for cxx_tag in cxx_tags_to_use:
                new_content = cxx_tag.replace(new_content)
            return new_content

    def read(self, content: str) -> None:
        """Reads the namespace content from the provided string.

        Args:
            content (str): The content where the data should be read from.
        """
        # Obtain all namespaces and cxx tags that need to be used currently
        inner_spaces = [cxx_tag for cxx_tag in self.value if isinstance(cxx_tag, CxxNamespace)]
        cxx_tags_to_use = [cxx_tag for cxx_tag in self.value if isinstance(cxx_tag, CxxVariable)]
        # Differ between existing namespaces and only existing variables
        if inner_spaces:
            # Get the start and end positions for this and the inner spaces
            (
                this_space_start_and_end,
                inner_space_start_and_end,
            ) = self._get_inner_spaces_positions(content, inner_spaces)
            inner_space_start_and_end += [(None, this_space_start_and_end[1], -1)]
            # Replace the content before the first namespace
            for cxx_tag in cxx_tags_to_use:
                cxx_tag.read(content[this_space_start_and_end[0] : inner_space_start_and_end[0][1]])
            # Loop through all inner namespaces and substitute the data
            for current_space, next_space in zip(inner_space_start_and_end[:-1], inner_space_start_and_end[1:]):
                # Replace the content for this namespace
                current_space[0].read(content[current_space[1] : current_space[2]])
                # Replace the content between the current and the next namespace
                for cxx_tag in cxx_tags_to_use:
                    cxx_tag.read(content[current_space[2] : next_space[1]])
        else:
            for cxx_tag in cxx_tags_to_use:
                cxx_tag.read(content)
