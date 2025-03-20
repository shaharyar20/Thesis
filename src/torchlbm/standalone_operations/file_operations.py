# Python modules
from typing import List
import numpy as np
from pathlib import Path

# own modules
from . import string_operations as string_o
from torchlbm.exceptions import TorchlbmError


def create_folder(folder_path: Path) -> bool:
    """Creates a folder and makes error check. Parent folders are created.

    Args:
        folder_path (Path): The folder that should be created.

    Returns:
        bool: True, if the folder creation was successful.
    """
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def remove_file(file_path: Path) -> bool:
    """Removes a file and makes error check.

    Args:
        file_path (Path): The path to the file that should be deleted.

    Returns:
        bool: True if successful.
    """
    try:
        file_path.unlink()
        return True
    except Exception:
        return False


def get_checked_file(file_path: Path) -> Path:
    """Gives absolute path to file and checks if it exists.

    Args:
        file_path (Path): The path to the file that should be checked.

    Raises:
        TorchlbmError: If file does not exist.

    Returns:
        Path: The ABSOLUTE path to the file.
    """
    file_path = Path(file_path).resolve().absolute()
    if not file_path.is_file():
        raise TorchlbmError(f"The file path {file_path} does not exist!") from None
    return file_path


def get_checked_folder(folder_path: Path) -> Path:
    """Gives absolute path to folder and checks if it exists.

    Args:
        folder_path (Path): The path to the folder that should be checked.

    Raises:
        TorchlbmError: If folder does not exist.

    Returns:
        Path: The ABSOLUTE path to the folder.
    """
    folder_path = Path(folder_path).resolve().absolute()
    if not folder_path.is_dir():
        raise TorchlbmError(f"The folder path {folder_path} does not exist!") from None
    return folder_path


def get_checked_path(data_path: Path) -> Path:
    """Gives absolute path (file or folder) and checks if it exists.

    Args:
        data_path (Path): The path (file or folder) that should be checked.

    Raises:
        TorchlbmError: If path does not exist.

    Returns:
        Path: The ABSOLUTE path.
    """
    try:
        return get_checked_folder(data_path)
    except TorchlbmError:
        try:
            return get_checked_file(data_path)
        except TorchlbmError:
            raise TorchlbmError(f"The path {data_path} is neither folder nor file path!") from None


def filter_files(files: List[Path], prefix: str = "", suffix: str = "", extension: str = None) -> List[Path]:
    """Filters files based on a given prefix, suffix and extension.

    Args:
        files (List[Path]): The list of files that need to be filtered.
        prefix (str, optional): The prefix files must have. Defaults to "".
        suffix (str, optional): The suffix files must have. Defaults to "".
        extension (str, optional): The extension files must have. Defaults to None.

    Returns:
        List[Path]: The filtered files (full path).
    """
    files = [file for file in files if prefix in file.stem and suffix in file.stem]
    # Filter those that do not fulfil the file extension
    if extension is not None:
        extension = f".{extension}" if "." not in extension else extension
        files = [file for file in files if file.suffix == extension]
    return files


def get_files_in_folder(folder_path: Path, prefix: str = "", suffix: str = "", extension: str = None) -> List[str]:
    """Gives all files in a folder.

    Args:
        folder_path (Path): The path to the folder.
        prefix (str, optional): The prefix files must have. Defaults to "".
        suffix (str, optional): The suffix files must have. Defaults to "".
        extension (str, optional): The extension files must have. Defaults to None.

    Returns:
        List[str]: A list with all files in the folder that match the conditions (full path).
    """
    # Get all files from the folder
    files = [file for file in folder_path.iterdir() if file.is_file()]
    return filter_files(files, prefix, suffix, extension)


def get_folders_in_folder(folder_path: Path, prefix="", suffix="") -> List[str]:
    """Gives all sub-folders in the folder.

    Args:
        folder_path (Path): The path to the folder.
        prefix (str, optional): The prefix sub-folders must have. Defaults to "".
        suffix (str, optional): The suffix sub-folders must have. Defaults to "".

    Returns:
        List[str]: A list with all sub-folders in the folder that match the conditions (full path).
    """
    return [folder for folder in folder_path.iterdir() if folder.is_dir() and prefix in folder.stem and suffix in folder.stem]


def sort_files_on_numbers(
    file_list: List[Path],
    prefix: str = "data_",
    suffix: str = "",
    is_float: bool = False,
) -> List[np.array]:
    """Sort files on numbers.

    Args:
        file_list (List[Path]): The list of file paths that should be sorted.
        prefix (str, optional): The prefix of the files. Defaults to "data_".
        suffix (str, optional): The suffix of the files. Defaults to "".
        is_float (bool, optional): Flag whether float comparison is used or not. Defaults to False (=integer).

    Returns:
        List[np.array]: An array of all files and numbers.
    """
    # Get the correct convert function
    convert_func = string_o.string_to_float if is_float else string_o.string_to_int
    # Extract all numbers
    numbers = [file.stem.replace(prefix, "").replace(suffix, "") for file in file_list]
    if any([convert_func(number, None) is None for number in numbers]):
        return np.array([]), np.array([])
    else:
        numbers = np.array([convert_func(number, None) for number in numbers])
        sorted_indices = np.argsort(numbers)
        return np.array(file_list)[sorted_indices], numbers[sorted_indices]


def get_unused_folder(folder_path: Path) -> Path:
    """Gives an unused folder path of the provided folder. Adds _index until unused folder is found.

    Args:
        folder_path (Path): The folder path.

    Returns:
        Path: The unused version of the folder path.
    """
    new_folder_path = folder_path
    number = 1
    while new_folder_path.is_dir():
        new_folder_path = folder_path.with_name(f"{folder_path.name}_{number}")
        number += 1
    return new_folder_path


def get_unused_file(file_path: Path) -> Path:
    """Gives an unused file path of the provided. Adds _index until unused file is found.

    Args:
        file_path (Path): The file path.

    Returns:
        Path: The unused version of the file path.
    """
    new_file_path = file_path
    number = 1
    while new_file_path.is_file():
        new_file_path = file_path.with_name(f"{file_path.stem}_{number}{file_path.suffix}")
        number += 1
    return new_file_path
