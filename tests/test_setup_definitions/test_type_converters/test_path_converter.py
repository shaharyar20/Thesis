# python modules
import pytest
from pathlib import Path

# torchlbm modules
from torchlbm.setup_definitions.type_converters import (
    PathConverter,
    FileConverter,
    FolderConverter,
)
from torchlbm.exceptions import TypeConversionError


def test_path_conversion(tmp_path):
    """Checks the conversion function."""
    test_file = tmp_path.joinpath("Test.txt")
    test_folder = tmp_path.joinpath("Test")
    # Path converter without requiring file existence
    converter = PathConverter(False, False)
    assert isinstance(converter.convert(test_file), Path)
    assert converter.convert(test_file).is_absolute()
    # Path converter with requiring file existence
    converter = PathConverter(True, False)
    with pytest.raises(TypeConversionError):
        converter.convert(test_file)
    test_file.write_text("Test")
    assert isinstance(converter.convert(test_file), Path)
    # Path converter with requiring folder existence
    converter = PathConverter(False, True)
    with pytest.raises(TypeConversionError):
        converter.convert(test_folder)
    test_folder.mkdir(parents=True)
    assert isinstance(converter.convert(test_folder), Path)


def test_file_conversion(tmp_path):
    """Checks the conversion function."""
    test_file1 = tmp_path.joinpath("Test.txt")
    test_file2 = tmp_path.joinpath("Test.log")
    # File converter without file restrictions
    converter = FileConverter(False)
    assert isinstance(converter.convert(test_file1), Path)
    assert converter.convert(test_file1).is_absolute()
    assert converter.convert(test_file2).is_absolute()
    # File converter with file restrictions
    converter = FileConverter(False, ["log"])
    with pytest.raises(TypeConversionError):
        converter.convert(test_file1)
    assert converter.convert(test_file2).is_absolute()
    # File converter with requiring file existence
    converter = PathConverter(True)
    with pytest.raises(TypeConversionError):
        converter.convert(test_file1)
    test_file1.write_text("Test")
    assert isinstance(converter.convert(test_file1), Path)


def test_folder_conversion(tmp_path):
    """Checks the conversion function."""
    test_folder = tmp_path.joinpath("Test")
    # Folder converter without requiring folder existence
    converter = FolderConverter(False)
    assert isinstance(converter.convert(test_folder), Path)
    assert converter.convert(test_folder).is_absolute()
    # Folder converter with requiring folder existence
    converter = FolderConverter(True)
    with pytest.raises(TypeConversionError):
        converter.convert(test_folder)
    test_folder.mkdir(parents=True)
    assert isinstance(converter.convert(test_folder), Path)


def test_file_comparison(tmp_path):
    """Checks the comparison function."""
    test_file1 = tmp_path.joinpath("Test.txt")
    test_file2 = tmp_path.joinpath("Test.log")
    test_file3 = tmp_path.joinpath("Test.log")
    # File converter without file restrictions
    converter = FileConverter(False)
    assert not converter.compare(test_file1, test_file2)
    assert converter.compare(test_file2, test_file3)


def test_folder_comparison(tmp_path):
    """Checks the comparison function."""
    test_folder1 = tmp_path.joinpath("Test")
    test_folder2 = tmp_path.joinpath("Test1")
    test_folder3 = tmp_path.joinpath("Test1")
    # File converter without folder restrictions
    converter = FolderConverter(False)
    assert not converter.compare(test_folder1, test_folder2)
    assert converter.compare(test_folder2, test_folder3)
