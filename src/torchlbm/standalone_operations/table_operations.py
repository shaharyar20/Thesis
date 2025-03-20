# python modules
from typing import Tuple
import pandas as pd
from pathlib import Path


def read_table(csv_file_path: Path, only_header: bool = False) -> pd.DataFrame:
    """Reads the table in proper format.

    Args:
        csv_file_path (Path): The path to the table that should be read.
        only_header (bool, optional): Flag whether only the header should be read. Defaults to False.

    Returns:
        pd.DataFrame: The fully read table (or only the header).
    """
    return pd.read_csv(
        csv_file_path,
        index_col=False,
        nrows=0 if only_header else None,
        sep=" *, *",
        skipinitialspace=True,
        engine="python",
    )


def write_table(
    table_to_write: pd.DataFrame,
    csv_file_path: Path,
    precision: int = 10,
    use_index: bool = False,
) -> str:
    """Writes the table with only floating numbers to the desired file. The header is formatted to align the number of characters with the number of
        digits and vice versa. Furthermore, the table index can be printed. If so, it must be a string index.

    Args:
        table_to_write (pd.DataFrame): The table to be written.
        csv_file_path (Path): The csv file path that should be used.
        precision (int, optional): The precision that is used to write the table. Defaults to 10.
        use_index (bool): Use the index in the printed table. Only string indices allowed.

    Returns:
        str: The error message if any is thrown by the writing process.
    """
    # Save the table
    err_msg = ""
    try:
        # Format the header to be consistent with the values. The renaming is done by copying the table to avoid 'set value on copy of slice'-error from
        # pandas.
        max_header_name = max([len(name) for name in table_to_write])
        number_of_characters = max([max_header_name + 1, precision + 7])
        number_of_spaces = number_of_characters - (precision + 6)
        # Format the header
        formatted_header = {name: (f"{{:>{number_of_characters}s}}").format(name) for name in table_to_write}
        if use_index:
            # Create the formatted index name dict
            max_index_name = max([len(name) for name in table_to_write.index]) + 1
            formatted_indices = {name: (f"{{:>{max_index_name}s}}").format(name) for name in table_to_write.index}
            # Format the index name
            index_name = "Index" if not table_to_write.index.name else table_to_write.index.name
            table_to_write.index.name = f"{{:>{max_index_name}s}}".format(index_name)
        else:
            formatted_indices = {}
        table_to_write = table_to_write.rename(columns=formatted_header, index=formatted_indices)
        # if not use_index:
        table_to_write.to_csv(
            str(csv_file_path),
            sep=",",
            header=True,
            index=use_index,
            encoding="utf-8",
            na_rep="NaN",
            float_format=f"{' ' * number_of_spaces}%.{precision}e",
        )
    except Exception as err:
        err_msg = str(err)
    return err_msg


def combined_new_with_existing_table(
    reference_table_path: Path,
    new_table: pd.DataFrame,
    replace: bool = False,
    append_vertically: bool = False,
) -> Tuple[pd.DataFrame, str, str]:
    """Combines a new table with an existing one. If the existing table is neither replaced not appended vertically, horizontal concatenation
        is applied and duplicated columns are removed.

    Args:
        reference_table_path (Path): The path to the reference table.
        new_table (pd.DataFrame): The new table to be appended.
        replace (bool, optional): Flag whether the existing table should be replaced. Defaults to False.
        append_vertically (bool, optional): Flag whether the new table should be appended vertically. Defaults to False.

    Returns:
        Tuple[pd.DataFrame,str,str]: The new combined table and an success and error message if something is not possible.
    """
    # If the table should be replaced, no checks need to be done
    if replace or not reference_table_path.is_file():
        return new_table, "Create full table", ""
    # Read the existing table
    reference_table = read_table(reference_table_path, only_header=False)
    # If the data should be appended, check the number of columns for the existing and new table
    if append_vertically:
        if reference_table.shape[1] != new_table.shape[1]:
            error_msg = f"Append fail: Column mismatch between new and existing. Got {new_table.shape[1]}, but expected {reference_table.shape[1]}"
            return new_table, "", error_msg
        else:
            return (
                pd.concat([reference_table, new_table], axis=0),
                "Append tables vertically",
                "",
            )
    # Otherwise, check that the number of rows coincide for the old and new table
    if reference_table.shape[0] != new_table.shape[0]:
        error_msg = f"Append fail: Row mismatch between new and existing. Got {new_table.shape[0]}, but expected {reference_table.shape[0]}"
        return new_table, "", error_msg
    else:
        return (
            reference_table.loc[:, ~reference_table.columns.duplicated()],
            "Append tables horizontally and remove duplicated columns",
            "",
        )
