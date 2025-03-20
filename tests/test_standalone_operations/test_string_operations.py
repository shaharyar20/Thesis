# python modules
import pytest
from pathlib import Path

# torchlbm modules
from torchlbm.standalone_operations import string_operations as string_o


def test_string_to_float():
    """Tests the string_to_float() function."""
    assert string_o.string_to_float("0.123", None) == pytest.approx(0.123)
    assert string_o.string_to_float("0. 123", None) == None
    assert string_o.string_to_float("0.123i", None) == None


def test_string_to_int():
    """Tests the string_to_int() function."""
    assert string_o.string_to_int("0.123", None) == None
    assert string_o.string_to_int("1", None) == 1
    assert string_o.string_to_int("1o", None) == None


def test_convert_to_percentage():
    """Tests the convert_to_percentage() function."""
    assert string_o.convert_to_percentage(0.5512345, width=5, precision=2) == "55.12"
    assert string_o.convert_to_percentage(None, width=5, precision=2) == None


def test_cut_string():
    """Tests the cut_string() function."""
    assert string_o.cut_string("TestString", 15) == ["TestString"]
    assert string_o.cut_string("TestString", 10) == ["TestString"]
    assert string_o.cut_string("TestString", 4) == ["Test", "Stri", "ng"]


def test_remove_vowels():
    """Tests the remove_vowels() function."""
    assert string_o.remove_vowels("TestStrIng") == "TstStrng"
    assert string_o.remove_vowels("OOaaeeoouuii") == ""


def test_reduce_path():
    """Tests the reduce_path() function."""
    assert string_o.reduce_path(Path("/first/second/third/file.txt"), 0) == ".../file.txt"
    assert string_o.reduce_path(Path("/first/second/third/file.txt"), 1) == ".../third/file.txt"
    assert string_o.reduce_path(Path("/first/second/third/file.txt"), 2) == ".../second/third/file.txt"
    assert string_o.reduce_path(Path("/first/second/third/file.txt"), 4) == "/first/second/third/file.txt"


def test_get_common_part_of_filename():
    """Tests the get_common_part_of_filename() function."""
    ref_file = Path("/first/file_with_prefix_and_suffix.txt")
    assert string_o.get_common_part_of_filename(ref_file, Path("/first/file_with_prefix_and_suffix.txt")) == "file_with_prefix_and_suffix"
    assert string_o.get_common_part_of_filename(ref_file, Path("file_with_prefix_and_suffix.txt")) == "file_with_prefix_and_suffix"
    assert string_o.get_common_part_of_filename(ref_file, Path("file_with_prefix_andd_suffix.txt")) == "file_with_prefix_and"
    assert string_o.get_common_part_of_filename(ref_file, Path("gfile_with_prefix_and_suffix.txt")) == ""


def test_number_to_roman():
    """Tests the number_to_roman() function."""
    assert string_o.number_to_roman(0) == "I"
    assert string_o.number_to_roman(12, lower_case=True) == "xii"
    assert string_o.number_to_roman(143) == "CXLIII"
    assert string_o.number_to_roman(2564) == "MMDLXIV"
    assert string_o.number_to_roman(4033, lower_case=True) == "xxxiii"


def test_number_to_alpha():
    """Tests the number_to_alpha() function."""
    assert string_o.number_to_alpha(0) == "A"
    assert string_o.number_to_alpha(5, lower_case=True) == "e"
    assert string_o.number_to_alpha(20) == "T"
    assert string_o.number_to_alpha(28, lower_case=True) == "b"


def test_convert_string_to_list():
    """Tests the convert_string_to_list() function."""
    assert string_o.convert_string_to_list("(1, 2;3 \n4 h]") == ["1", "2", "3", "4 h"]
    assert string_o.convert_string_to_list(" ([1, 2,3 ,4 h]}") == [
        "[1",
        "2",
        "3",
        "4 h]",
    ]


def test_is_none():
    """Tests the is_none() function."""
    assert string_o.is_none("noNe") == True
    assert string_o.is_none("noNee") == False
    assert string_o.is_none("No ne") == False
    assert string_o.is_none("None") == True


def test_convert_time():
    """Tests the convert_time() function."""
    assert string_o.convert_time(360) == "0:06:00"
    assert string_o.convert_time(86399) == "23:59:59"
    assert string_o.convert_time(86400) == "1 day, 0:00:00"


def test_convert_bytes():
    """Tests the convert_bytes() function."""
    assert string_o.convert_bytes(100.1) == "100.1 B"
    assert string_o.convert_bytes(1205.73) == "1.2 KB"
    assert string_o.convert_bytes(50.5e6) == "50.5 MB"
    assert string_o.convert_bytes(100.734e9) == "100.7 GB"


def test_split_scientific_float_in_parts():
    """Tests the split_scientific_float_in_parts() function."""
    assert string_o.split_scientific_float_in_parts("1.234567e-05") == (
        "1",
        "234567",
        "-05",
    )
    assert string_o.split_scientific_float_in_parts("7.6543210e+07") == (
        "7",
        "6543210",
        "+07",
    )


def test_get_relevant_digit_of_scientific_float():
    """Tests the get_relevant_digit_of_scientific_float() function."""
    assert string_o.get_relevant_digit_of_scientific_float("1.2345e-5", "1.3456e-5") == 1
    assert string_o.get_relevant_digit_of_scientific_float("1.123123123123e-7", "1.123123123023e-6") == 10
