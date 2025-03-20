# python modules
import pytest
from pathlib import Path

# torchlbm modules
from torchlbm.exceptions import TorchlbmError
import torchlbm.standalone_operations.file_operations as file_o


def test_create_folder(tmp_path):
    """Tests the create_folder() function."""
    path = tmp_path.joinpath("TestFolder")
    assert file_o.create_folder(path) == True


def test_remove_file(tmp_path):
    """Tests the remove_file() function."""
    path = tmp_path.joinpath("test_file.txt")
    # Currently file does not exists
    assert file_o.remove_file(path) == False
    # Write something to file and remove it
    path.write_text("Test")
    assert file_o.remove_file(path) == True
    assert len(list(tmp_path.iterdir())) == 0


def test_get_checked_path(tmp_path):
    """Tests the get_checked_path() function."""
    file_path = tmp_path.joinpath("test_file.txt")
    folder_path = tmp_path.joinpath("sub_folder/sub_sub_folder")
    # Currently neither file nor folder exists
    with pytest.raises(TorchlbmError):
        file_o.get_checked_file(file_path)
    with pytest.raises(TorchlbmError):
        file_o.get_checked_folder(folder_path)
    # Write something to file and create the folder
    file_path.write_text("Test")
    folder_path.mkdir(parents=True)
    assert file_o.get_checked_file(file_path) == file_path.resolve().absolute()
    assert file_o.get_checked_folder(folder_path) == folder_path.resolve().absolute()


def test_filter_files(tmp_path):
    """Tests the filter_files() function."""
    # Consider three files (no creation required)
    file_path_txt = tmp_path.joinpath("test_file.txt")
    file_path_log = tmp_path.joinpath("test_file_suffix.log")
    file_path_csv_log = tmp_path.joinpath("prefix_test_file.csv.log")
    # Get the correct files
    assert file_o.filter_files([file_path_txt, file_path_log, file_path_csv_log]) == [
        file_path_txt,
        file_path_log,
        file_path_csv_log,
    ]
    assert file_o.filter_files([file_path_txt, file_path_log, file_path_csv_log], prefix="prefix") == [file_path_csv_log]
    assert file_o.filter_files([file_path_txt, file_path_log, file_path_csv_log], suffix="suffix") == [file_path_log]
    assert file_o.filter_files([file_path_txt, file_path_log, file_path_csv_log], extension="log") == [file_path_log, file_path_csv_log]
    assert file_o.filter_files([file_path_txt, file_path_log, file_path_csv_log], extension="csv") == []
    assert file_o.filter_files([file_path_txt, file_path_log, file_path_csv_log], extension="txt") == [file_path_txt]


def test_get_files_in_folder(tmp_path):
    """Tests the get_files_in_folder() function."""
    # Consider two files and create them
    file_path_txt = tmp_path.joinpath("test_file.txt")
    file_path_log = tmp_path.joinpath("test_file_suffix.log")
    file_path_log_log = tmp_path.joinpath("test_file.log.log")
    file_path_txt.write_text("Test")
    file_path_log.write_text("Test")
    file_path_log_log.write_text("Test")
    # Get the correct files
    files = file_o.get_files_in_folder(tmp_path)
    assert file_path_txt in files and file_path_log in files and file_path_log_log in files
    files = file_o.get_files_in_folder(tmp_path, prefix="test")
    assert file_path_txt in files and file_path_log in files and file_path_log_log in files
    files = file_o.get_files_in_folder(tmp_path, suffix="suffix")
    assert file_path_log in files
    files = file_o.get_files_in_folder(tmp_path, extension="txt")
    assert file_path_txt in files
    files = file_o.get_files_in_folder(tmp_path, extension="log")
    assert file_path_log in files and file_path_log_log in files


def test_get_folders_in_folder(tmp_path):
    """Tests the get_folders_in_folder() function."""
    # Consider three sub folder and create them
    first_sub_folder = tmp_path.joinpath("sub_folder_suffix")
    second_sub_folder = tmp_path.joinpath("prefix_sub_folder")
    third_sub_folder = tmp_path.joinpath("sub_folder/sub_sub_folder")
    first_sub_folder.mkdir(parents=True)
    second_sub_folder.mkdir(parents=True)
    third_sub_folder.mkdir(parents=True)
    # Get the correct folders
    folders = file_o.get_folders_in_folder(tmp_path)
    assert all([folder in folders for folder in [first_sub_folder, second_sub_folder, third_sub_folder.parent]])
    folders = file_o.get_folders_in_folder(tmp_path, prefix="prefix")
    assert second_sub_folder in folders
    folders = file_o.get_folders_in_folder(tmp_path, prefix="Prefix")
    assert folders == []
    folders = file_o.get_folders_in_folder(tmp_path, suffix="suffix")
    assert first_sub_folder in folders


def test_sort_files_on_numbers(tmp_path):
    """Tests the sort_files_on_numbers() function."""
    # Consider five files (no creation required)
    file_path_1 = tmp_path.joinpath("prefix_1.0_suffix.txt")
    file_path_2 = tmp_path.joinpath("prefix_1.02_suffix.txt")
    file_path_3 = tmp_path.joinpath("prefix_2.0_suffix.txt")
    file_path_4 = tmp_path.joinpath("prefix1_3.0_suffix.csv")
    # Sorted files (integer based)
    files, numbers = file_o.sort_files_on_numbers(
        [file_path_3, file_path_1, file_path_2, file_path_4],
        prefix="prefix_",
        suffix="_suffix",
    )
    assert files.tolist() == []
    assert numbers.tolist() == []
    # Sorted files (float based)
    files, numbers = file_o.sort_files_on_numbers(
        [file_path_3, file_path_1, file_path_2],
        prefix="prefix_",
        suffix="_suffix",
        is_float=True,
    )
    assert files.tolist() == [file_path_1, file_path_2, file_path_3]
    assert numbers.tolist() == [
        pytest.approx(1.0),
        pytest.approx(1.02),
        pytest.approx(2.0),
    ]
    # Sorted files (float based, with wrong file)
    files, numbers = file_o.sort_files_on_numbers(
        [file_path_3, file_path_1, file_path_4],
        prefix="prefix_",
        suffix="_suffix",
        is_float=True,
    )
    assert files.tolist() == []
    assert numbers.tolist() == []


def test_get_unused_folder(tmp_path):
    """Tests the get_unused_folder() function."""
    # Consider sub folder and create
    sub_folder = tmp_path.joinpath("sub_folder")
    sub_folder.mkdir(parents=True)
    # Get the correct unused folder
    assert file_o.get_unused_folder(sub_folder) == tmp_path.joinpath("sub_folder_1")


def test_get_unused_file(tmp_path):
    """Tests the get_unused_file() function."""
    # Consider sub folder and create
    file = tmp_path.joinpath("file.txt")
    file.write_text("Test")
    # Get the correct unused folder
    assert file_o.get_unused_file(file) == tmp_path.joinpath("file_1.txt")
