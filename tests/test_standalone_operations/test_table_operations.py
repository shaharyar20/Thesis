# python modules
from os import read
import pytest
import numpy as np
import pandas as pd

# torchlbm modules
from torchlbm.standalone_operations import table_operations as table_o


def test_write_read_table(tmp_path):
    """Tests the write_table() function."""
    table_path = tmp_path.joinpath("test_table.txt")
    table_cols = ["col1", "colcol2", "VeryVeryVeryLongColumn"]
    table_content = np.array([[1.0, 2.0, 3.1], [4e-6, 5e-4, 6.1], [100.0, 2e6, 3.1e9]])
    table = pd.DataFrame(table_content, columns=table_cols)
    # Write the table to file
    table_o.write_table(table, table_path, precision=10)
    # Check that the file has been written
    files_in_folder = list(tmp_path.iterdir())
    assert len(files_in_folder) == 1
    assert table_path in files_in_folder
    # Check the written content
    table_lines = open(table_path, "r").readlines()
    assert table_lines[0] == f"{' ' * 19}col1,{' ' * 16}colcol2, VeryVeryVeryLongColumn\n"
    assert table_lines[1] == f"{' ' * 7}1.0000000000e+00,{' ' * 7}2.0000000000e+00,{' ' * 7}3.1000000000e+00\n"
    assert table_lines[2] == f"{' ' * 7}4.0000000000e-06,{' ' * 7}5.0000000000e-04,{' ' * 7}6.1000000000e+00\n"
    assert table_lines[3] == f"{' ' * 7}1.0000000000e+02,{' ' * 7}2.0000000000e+06,{' ' * 7}3.1000000000e+09\n"
    # Read back the table completely from the file
    read_table = table_o.read_table(table_path)
    assert read_table.columns.tolist() == ["col1", "colcol2", "VeryVeryVeryLongColumn"]
    assert read_table.shape == (3, 3)
    # Read only the header
    read_table = table_o.read_table(table_path, only_header=True)
    assert read_table.empty == True
