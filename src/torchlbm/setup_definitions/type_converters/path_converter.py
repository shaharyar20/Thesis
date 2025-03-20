#!/usr/bin/env python3
# Python modules
from typing import List, Any
from pathlib import Path

# torchlbm modules
from torchlbm.setup_definitions.type_converters import TypeConverter
from torchlbm.exceptions import TorchlbmError, TypeConversionError
from torchlbm.file_extension import FileExtension
import torchlbm.standalone_operations.file_operations as file_o


class PathConverter(TypeConverter):
    """Wrapper class for all path type conversions.

    It allows the check for file/folder existence. The final type conversion is always a pathlib-Path type.

    Args:
        TypeConverter (TypeConverter): The base class of all type converters.
    """

    def __init__(self, file_must_exist: bool = False, folder_must_exist: bool = False) -> None:
        """Constructor.

        Args:
            file_must_exist (bool, optional): Flag whether file existence should be checked or not. Defaults to False.
            folder_must_exist (bool, optional): Flag whether folder existence should be checked or not. Defaults to False.
        """
        # Call base class constructor
        super().__init__(True)
        # Assign the member variables
        self._check_file = file_must_exist
        self._check_folder = folder_must_exist

    def _convert(self, value: Any) -> Path:
        """See base class definition.

        Returns:
            Path: The converted pathlib-Path value.
        """
        converted_path = Path(value).resolve().absolute()
        try:
            if self._check_file and self._check_folder:
                return file_o.get_checked_path(converted_path)
            elif self._check_file:
                return file_o.get_checked_file(converted_path)
            elif self._check_folder:
                return file_o.get_checked_folder(converted_path)
            else:
                return converted_path
        except TorchlbmError as err:
            raise TypeConversionError(str(err)) from None


class FileConverter(PathConverter):
    """Simplified path type converter for file only converters.

    Args:
        PathConverter (PathConverter): The base class of all path converters.
    """

    def __init__(self, must_exist: bool, allowed_extensions: List[str] = []) -> None:
        """Constructor.

        Args:
            must_exist (bool, optional): Flag whether file existence should be checked. Defaults to False.
            allowed_extensions (List[str]): List of extensions that are allowed.
        """
        super().__init__(must_exist, False)
        self._allowed_extensions = FileExtension.get_values(allowed_extensions)

    def _convert(self, value: Any) -> Path:
        """Overwriting the method for additional extension check.

        Args:
            value (Any): The value to be checked.

        Returns:
            Path: The checked path.
        """
        converted_path = super()._convert(value)
        if self._allowed_extensions and FileExtension.from_path(converted_path) not in self._allowed_extensions:
            raise TypeConversionError(
                f"File extension {converted_path.suffix} is not allowed.\n" f"Allowed extensions: [{', '.join([str(ext) for ext in self._allowed_extensions])}]"
            ) from None
        return converted_path


class FolderConverter(PathConverter):
    """Simplified path type converter for folder only converters.

    Args:
        PathConverter (PathConverter): The base class of all path converters.
    """

    def __init__(self, must_exist: bool) -> None:
        """Constructor.

        Args:
            must_exist (bool, optional): Flag whether folder existence should be checked. Defaults to False.
        """
        super().__init__(False, must_exist)
