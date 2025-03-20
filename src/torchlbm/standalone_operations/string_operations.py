#!/usr/bin/env python3
# Python modules
from typing import List, Tuple, Any
import re
import string as str_lib
from pathlib import Path
import datetime


def string_to_float(value: str, default: float) -> float:
    """Converts a string into float value.

    Args:
        value (str): The value that should be converted.
        default (float): Fallback default value if string not convertible.

    Returns:
        float: The converted/default value.
    """
    try:
        return float(value)
    except ValueError:
        return default


def string_to_int(value: str, default: int) -> int:
    """Converts a string into int value.

    Args:
        value (str): The value that should be converted.
        default (int): Fallback default value if string not convertible.

    Returns:
        int: The converted/default value.
    """
    try:
        return int(value)
    except ValueError:
        return default


def convert_value_to_string(value: Any, width: int = None, precision: int = 8) -> str:
    """Converts a general value into a proper string.

    Args:
        value (Any): The value to be converted.
        width (int, optional): The width of the string. Defaults to None.
        precision (int, optional): The precision of floating values. Defaults to 8.

    Returns:
        str: The converted value.
    """
    # Differ between float, int and str
    if isinstance(value, float):
        width = "" if width is None else width
        return ("{:" + str(width) + "." + str(precision) + "e}").format(value)
    elif isinstance(value, int):
        return str(value) if width is None else str(value).rjust(width)
    else:
        return value if width is None else value.rjust(width)


def convert_to_percentage(value: float = None, width: int = None, precision: int = 0) -> str:
    """Converts a floating value to a percentage string.

    Args:
        value (float, optional): The value to be converted. Defaults to None.
        width (int, optional): The width of the string generated. Defaults to None.
        precision (int, optional): The precision of the percentag value. Defaults to 0.

    Returns:
        str: The percentags string.
    """
    if value is not None and not isinstance(value, str):
        if width is None:
            width = ""
        return ("{:" + str(width) + "." + str(precision) + "f}").format(round(value * 100, 2))
    else:
        return value


def cut_string(string: str, width: int) -> List[str]:
    """Cuts the string into substrings of a maximum width.

    Args:
        string (str): Thr string that should be cut.
        width (int): The width at which the string is cut.

    Returns:
        List[str]: The list with all substrings
    """
    return [string[i : i + width] for i in range(0, len(string), width)]


def remove_vowels(string: str) -> str:
    """Removes all vowels from a string.

    Args:
        string (str): The string where the vowels should be removed.

    Returns:
        str: The modified string.
    """
    vowels = ["a", "o", "e", "i", "u"]
    return "".join([char for char in string if char.lower() not in vowels])


def reduce_path(file_path: Path, number_of_parents=1) -> str:
    """Reduces the path to a certain degree starting from the realtive origin of the current path.

    Args:
        file_path (Path): The path that should be reduced.
        number_of_parents (int, optional): The number of parents to be used. Defaults to 1.

    Returns:
        str: The reduced path string.
    """
    parents = file_path.parents
    try:
        return f"{'.' * 3}/{file_path.relative_to(parents[number_of_parents])}" if len(parents) >= number_of_parents + 1 else f"{file_path}"
    except ValueError:
        return f"{file_path}"


def get_common_part_of_filename(ref_file_path: Path, match_file_path: Path) -> str:
    """Gives the common part of the filename for two file paths. Always starts searching from the left.

    Args:
        ref_file_path (Path): The reference file paths.
        match_file_path (Path): The file path to be matched.

    Returns:
        str: The common part of both filenames. Empty if there is not common string from the beginning.
    """
    try:
        unmatched_index = next(index for index, (ref_char, match_char) in enumerate(zip(ref_file_path.stem, match_file_path.stem)) if ref_char != match_char)
        return ref_file_path.stem[:unmatched_index]
    except StopIteration:
        return ref_file_path.stem


def number_to_roman(number: int, lower_case: bool = False) -> str:
    """Converts a number to a roman numeral. Starts at 1 in case 4000 is exceeded. Zero is converted to 1. Fail safe version.

    Args:
        number (int): The number that should be converted.
        lower_case (bool, optional): Flag whether lower case version should be used. Defaults to False.

    Raises:
        ValueError: If the number is not positive non-zero.

    Returns:
        str: The roman numeral.
    """
    if number <= 0:
        number = 1
    if number > 4000:
        number = number % 4000
    values = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
    romans = ("M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I")
    if lower_case:
        romans = tuple([letter.lower() for letter in romans])
    result = []
    for i in range(len(values)):
        count = int(number / values[i])
        result.append(romans[i] * count)
        number -= values[i] * count
    return "".join(result)


def number_to_alpha(number: int, lower_case: bool = False) -> str:
    """Converts a number to latin alpha numeral. Starts at 1 in case 26 is exceeded. Zero is converted to 1. Fail safe version.

    Args:
        number (int): The number that should be converted.
        lower_case (bool, optional): Flag whether lower case version should be used. Defaults to False.

    Raises:
        ValueError: If the number is not positive non-zero.

    Returns:
        str: The alpha numeral.
    """
    if number <= 0:
        number = 1
    if number > 26:
        number = number % 26
    letters = str_lib.ascii_lowercase if lower_case else str_lib.ascii_uppercase
    return letters[number - 1]


def convert_string_to_list(
    string: str,
    separators: str = ",|;|\t|\n",
    opening_brackets=r"\(|\{|\[",
    closing_brackets=r"\)|\}|\]",
) -> List[str]:
    """Converts a string into a list of substrings using the predefined separators.

    Args:
        string (str): The string that should be converted.
        separators (str, optional): A string of separators to be used. Different separators must be separated by |. Defaults to ",|;|\t|\n".
        opening_brackets (str, optional): A string of opening brackets that should be removed (only the first is removed).
                                          Different separators must be separated by |. Defaults to ( { [.
        closing_brackets (str, optional): A string of closing brackets that should be removed (only the first is removed).
                                          Different separators must be separated by |. Defaults to ) } ].

    Returns:
        List[str]: The list of strings.
    """
    # Strip all characters from the string
    converted_string = string.strip()
    # Remove the FIRST leading bracket from the string
    split_string = re.split(opening_brackets, converted_string)
    converted_string = converted_string if split_string[0] else converted_string[1:]
    # Remove the LAST closing bracket from the string
    split_string = re.split(closing_brackets, converted_string)
    converted_string = converted_string if split_string[-1] else converted_string[:-1]
    # Return the split list elements
    return [element.strip() for element in re.split(separators, converted_string)]


def is_none(string: str) -> bool:
    """Checks whether a given string can be mapped to None. Allowed variations are None or Null.

    Args:
        string (str): The string that should be investigated.

    Returns:
        bool: True if string is None or not.
    """
    if string is None:
        return True
    elif isinstance(string, str) and string.strip().upper() in ["NONE", "NULL"]:
        return True
    else:
        return False


def convert_time(time: int) -> str:
    """Converts the time in seconds in to format HH::MM::SS.

    Args:
        time (int): The time that should be converted.

    Returns:
        str: The converted time as string.
    """
    return str(datetime.timedelta(seconds=time)).split(".")[0]


def convert_bytes(bytes: int) -> str:
    """Converts the number of bytes into human readable version.

    Args:
        bytes (int): The number of bytes.

    Returns:
        str: Human readable bytes string.

    Note:
        Negative bytes are converted to zero.
    """
    if bytes < 0:
        bytes = 0
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(bytes) < 1000.0:
            return f"{bytes:3.1f} {unit}B"
        bytes /= 1000.0
    return f"{bytes:.1f} YB"


def split_scientific_float_in_parts(number_string: str) -> Tuple[str, str, str]:
    """Splits a fixed float string into its parts (leading number, mantissa and exponent)

    Args:
        number_string (str): The fixed float number as string.

    Returns:
        Tuple[str, str, str]: leading number, mantissa and exponent as strings.
    """
    number, exp = number_string.split("e")
    leading, mantissa = number.split(".")
    return (leading, mantissa, exp)


def get_relevant_digit_of_scientific_float(ref_number: str, match_number: str) -> int:
    """Gives the relevant digit of the compariston between tow fixed float string values.

    Args:
        ref_number (str): The reference number.
        match_number (str): The match number.

    Returns:
        int: the relevant digit in the mantissa.
    """
    _, ref_mantissa, _ = split_scientific_float_in_parts(ref_number)
    _, match_mantissa, _ = split_scientific_float_in_parts(match_number)
    try:
        return next(index for index, (ref_digit, match_digit) in enumerate(zip(ref_mantissa, match_mantissa), start=1) if ref_digit != match_digit)
    except StopIteration:
        return None
