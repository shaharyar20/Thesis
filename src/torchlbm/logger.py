#!/usr/bin/env python3
# Python modules
from torchlbm.exceptions import TorchlbmError
from typing import List, IO
import shutil
import sys
import math
import numpy as np
from pathlib import Path

# torchlbm modules
import torchlbm.standalone_operations.string_operations as string_o


class Singleton(type):
    """Singleton class.

    This class provides the interface of singleton classes to use the logger in different modules, but allowing specifications
    in the main module only. All subsequent modules do not create their own class, but take the already created singleton object.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    """The torchlbm logger.

    Logger class that provides logging information into the terminal and specified logging file. In general, the log-file does not contain
    any colored information whereas the standard-output does. The data is written to the logfile whenever a starline (breakline) is written.
    Ensure that this is done at the end, best by calling the bye_message function. The logger provides an aligned interface for logging from different modules.
    Therefore, the logger should always be used instead of print().

    Attributes:
        _terminal_width (int):h of the terminal output.
        _logfile_width (int):h of the strings used for the logfile output.
        _write_to_file (bool):ther log files is generated.
        _log_file (Path): of the log file generated.
        _filestream (str):rnally used filestream for buffering current strings until they are written to file.
        active (bool):ther the logger is active.
        _colors (Dict[str,str]):ertain character to proper color code.
        _indent (int):ent indent used for the string generation.

    Note:
        This class is a singleton class and always contains the information from the first class instantiation. Therefore, only call the
        Logger() inside __init__ functions to ensure that it is an instance_variable. Using the logger as class_variable causes problems
        during module import.
    """

    # Gives the size of the terminal to provide proper logging
    _terminal_width = 120
    _logfile_width = 120
    # Information about the log file that is written and if it is desired
    _write_to_file = False
    _log_file = Path("std_out.log").resolve().absolute()
    _filestream = ""
    active = True
    # Dictionary defining the logging colors
    _colors = {
        "r": "\033[91m",
        "g": "\033[92m",
        "b": "\033[94m",
        "y": "\033[93m",
        "bold": "\033[1m",
        "uline": "\033[4m",
    }

    def __init__(self, write_to_file: bool = False, log_filename: str = "std_out.log", use_modulus_logger: bool = False) -> None:
        """Constructor.

        Args:
            write_to_file (bool, optional): Flag whether data is written to a file. Defaults to False
            log_filename (str, optional): The filename of the log file. Defaults to "std_out.log"
        """
        self._write_to_file = write_to_file
        self._log_file = Path(log_filename).resolve().absolute()
        self._terminal_width = 80 - 8  # shutil.get_terminal_size(fallback=(80, 20)).columns - 8
        self._indent = 0
        self._use_modulus_logger = use_modulus_logger
        if self._use_modulus_logger:
            from modulus.launch.logging.console import PythonLogger
        self._modulus_logger = PythonLogger("Simulation") if self._use_modulus_logger else None
        # Create all folders to the log file if not existing
        if self._write_to_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def indent(self) -> int:
        """Gives the indent (number of spaces) the logger is currently using.

        Returns:
            int: The current logging indent.
        """
        return self._indent

    @indent.setter
    def indent(self, indent: int) -> None:
        """Sets the current indent value of the logger

        Sets the indent value to another number. Instead of using logger.indent = value, it is advantageous to use logger.indent += value.
        Then the indent is increased and proper output can be generated combining different modules without needing to modify the files if a super function
        changes its indent. After the indent has been changed it should be reverted at the function end to ensure other modules
        use the previously defined indent.

        Args:
            indent (int): The new value the indent should have.
        """
        self._indent = int(indent)
        if self._indent < 0:
            self._indent = 0

    def _color_string(self, string: str, color: str = "") -> str:
        """Converts a log string into a colored string.

        Args:
            string (str): The string that should be converted.
            color (str, optional): The used color. Defaults to ""

        Returns:
            str: The colored string.
        """
        return f"{self._colors[color]}{string}\033[0m" if color in color in self._colors.keys() else string

    def _create_log_strings(self, log_text: str, width: int, color: str, center: bool) -> List[str]:
        """Creates the log string.

        Creates the log string dependent on the given width and other properties. The log text is split into separate lines if it
        exceeds the logging width.

        Args:
            log_text (str): The text that is logged.
            width (int): The width used for the writing.
            color (str): The color that is used (None if not specified).
            center (bool): Flag to center the given string.

        Returns:
            List[str]: The lines that should be logged.
        """
        # Adjust the width to the current indent
        width = width - self._indent
        # First make single strings to list
        if isinstance(log_text, str):
            log_text = [log_text]
        elif not isinstance(log_text, (list, tuple)):
            raise TorchlbmError("Cannot create log string. Must be a single string or list of strings.")
        # Split the strings at the new line characters
        text_array = [text.splitlines() for text in log_text]
        text_array = [log_text for log_text_list in text_array for log_text in log_text_list]
        # Add a single space to lines that are empty
        text_array = [" " if not string else string for string in text_array]
        # Cut the string at the appropriate positions
        text_array = [string_o.cut_string(log_text, width) for log_text in text_array]
        text_array = [log_text for log_text_list in text_array for log_text in log_text_list]

        log_strings = []
        # Add color information and other styles to each string
        for text_part in text_array:
            # Center the text if desired
            if center:
                text_part = text_part.center(width)
            # Color if desired
            if color is not None:
                text_part = self._color_string(text_part.ljust(width), color)

            log_strings.append(f"|*  {' ' * self._indent}{text_part.ljust(width)}  *|")

        # Return the created strings
        return log_strings

    def _create_tabular_strings(self, entries: List[List[str]], colors: List[List[str]], width: int) -> List[str]:
        """Creates a proper logging string in tabular form.

        Args:
            entries (List[List[str]]): The log-entries of the tabular (2D-list) specifying the row and column entries.
            colors (List[List[str]]): The colors for each entry (2D-list). Must be the same size than the entries (None if not specified).
            width (int): The width used for the writing.

        Returns:
            List[str]: The tabular lines that should be logged.
        """
        # Adjust the width to the current indent
        width = width - self._indent

        # First get the size of the cells for each column
        length_checker = np.vectorize(len)
        max_size = np.amax(length_checker(entries), axis=0)

        # Convert the rows that contain only single hyphens into a full dashed line by adding number of hyphens equalizing the
        # max size of each column
        for row in np.where((entries == "-").all(axis=1))[0]:
            for col, size in zip(range(0, entries.shape[1]), max_size):
                entries[row, col] = size * "-"

        # Check the sizes and cut the table until it fits the log_width
        split_columns = []
        if np.sum(max_size) > width:
            # First check if table is printable at all (first col + one of all other cols can be printed together)
            if width < max_size[0] + max(max_size[1:]):
                return [
                    self._create_log_strings(
                        "Table not printable. Width not sufficient to split-up table",
                        width,
                        "y",
                    )
                ]

            # Add all entries until the logwidth is reached (always add the first column)
            split_column = [0]

            for col_index, col_entry in enumerate(entries.T[1:, :]):
                if np.sum(max_size[split_column]) + max_size[col_index + 1] > width:
                    split_columns.append(split_column)
                    split_column = [0]
                split_column.append(col_index + 1)
            split_columns.append(split_column)
        else:
            split_columns.append([col for col in range(0, entries.shape[1])])

        # Create the table with appropriate strings and colors
        log_strings = []
        for cols in split_columns:
            size = max_size[cols]
            if colors is not None:
                for row_entry, row_colors in zip(entries[:, cols], colors[:, cols]):
                    text_part = ""
                    color_length = 0
                    for col_index, (col_entry, col_color) in enumerate(zip(row_entry, row_colors)):
                        if colors is not None:
                            colored_string = self._color_string(str(col_entry).center(size[col_index]), col_color)
                            color_length += len(colored_string) - size[col_index]
                            text_part += colored_string
                        else:
                            text_part += str(col_entry).center(size[col_index])
                    log_strings.append(f"|*  {' ' * self._indent}{text_part.ljust(width + color_length)}  *|")
            else:
                for row_entry in entries[:, cols]:
                    text_part = ""
                    for col_index, col_entry in enumerate(row_entry):
                        text_part += str(col_entry).center(size[col_index])
                    log_strings.append(f"|*  {' ' * self._indent}{text_part.ljust(width)}  *|")

        return log_strings

    def write(self, *log_text: str, color: str = None, center: bool = False) -> IO:
        """Function that carries out the actual writing to terminal and file.

        Args:
            log_text (str): The text that should be logged can be several entries or with \n for multiline. For List of string use the unpacking operator.
            color (str, optional): The used color. Defaults to None
            center (bool, optional): Flag whether the string should be centered or not. Defaults to False.
        """
        if not self.active:
            return None
        log_string = "\n".join(self._create_log_strings(log_text, self._terminal_width, color, center=center))
        if self._use_modulus_logger:
            self._modulus_logger.info(log_string)
        else:
            print(log_string, flush=True)
        # Write different style to file if file writing is desired
        if self._write_to_file:
            log_string = self._create_log_strings(log_text, self._logfile_width, color=None, center=center)
            for part in log_string:
                self._filestream += f"{part}\n"

    def write_status_bar(self, percentage: float, tag: str = None, print_inline: bool = False) -> IO:
        """Logs a filled repeating status bar depending on the given percentage.

        Args:
            percentage (float): The percentage how much of the status bar is filled. The value must be between 0 and 1, where 1 maps to 100%.
            tag (str, optional): An additional tag that can be used at the right-side of status bar (default: None uses the percentag in %). Defaults to None.
            print_inline (bool, optional): An additional flag indicating wheter the status bar should be printed inline to the terminal. Defaults to False.

        Note:
            Status bars are never printed to the file, since they make only sense for the real std output
        """
        if not self.active:
            return None
        # Define the width for the status bar
        width = self._terminal_width - self._indent
        # Define the tag is not specified (percentag in %)
        if tag is None:
            tag = string_o.convert_to_percentage(percentage, 3, 0) + "%"
        # Define the two parts of the status bar (filled and empty bar)
        star_width = math.floor(percentage * (width - 3 - len(tag)))
        empty_width = (width - 3 - len(tag)) - star_width
        # Define the text part with proper logging borders and log it to terminal
        if self._use_modulus_logger:
            text_part = f"{' ' * self._indent}[{'-' * star_width}{' ' * empty_width}] {tag}"
            self._modulus_logger.info(text_part)
        else:
            text_part = f"|*  {' ' * self._indent}[{'-' * star_width}{' ' * empty_width}] {tag}  *|"
            if print_inline:
                sys.stdout.write("\r")
            sys.stdout.write(text_part)
            sys.stdout.flush()

        if self._write_to_file:
            # Define the width for the status bar in the file
            width = self._logfile_width - self._indent - 6
            # Define the two parts of the status bar (filled and empty bar)
            star_width = math.floor(percentage * (width - 3 - len(tag)))
            empty_width = (width - 3 - len(tag)) - star_width
            # Define the text part with proper logging borders and log it to terminal
            log_text = f"{' ' * self._indent}[{'-' * star_width}{' ' * empty_width}] {tag}"
            log_string = self._create_log_strings(log_text, self._logfile_width, color=None, center=False)
            for part in log_string:
                self._filestream += f"{part}\n"

    def write_table(
        self,
        table_entries: List[List[str]],
        colors: List[List[str]] = None,
        vertical_lines=False,
        horizontal_lines=False,
    ) -> IO:
        """Prints a table to the terminal or file.

        Prints a list into tabular form with colored strings. A check is done that the number of columns is consistent for all rows.
        Furthermore, separating lines can be added to the table by using a completely empty row list. E.g., [ ["A", "B"], [], ["C", "D"] ] results in
              A B
              ---
              C D

        Args:
            table_entries (List[List[str]]): The entries of the tabular (2D list). First row is the header of tabular. First column is the row naming.
            colors (List[List[str]], optional): The colors for the entries (2D list). Must be the same size than the entries by default None.

        Note:
            In case the number of columns/rows is inconsistent or the colors size does not coincide with the entries size,tThe table is no printed.
            Instead, a warning message is displayed.
        """
        if not self.active:
            return None
        # Check is table_entries and colors have the same number of rows
        if colors is not None and len(table_entries) != len(colors):
            self.write(
                "Table not printable, inconsistent number of rows between table and colors",
                color="y",
            )
            return None

        # Loop through all entries and make them strings and add white space after
        entries = []
        n_cols = max([len(row) for row in table_entries])
        for row_index, row in enumerate(table_entries):
            # Add a single hyphen to fully empty rows (later converted into and complete dashed line)
            if len(row) == 0:
                if not horizontal_lines:
                    entries.append(["-" for _ in range(0, n_cols)])
                    if colors is not None:
                        colors[row_index] = ["" for _ in range(0, n_cols)]
                else:
                    colors[row_index] = []
            # Check if rows have same number of entries (only if len is not zero)
            elif len(row) != n_cols:
                self.write("Table not printable, inconsistent number of columns", color="y")
                return
            # Check if entries and colors are the same
            elif colors is not None and len(row) != len(colors[row_index]):
                self.write(
                    "Table not printable, inconsistent number of columns between entries and colors",
                    color="y",
                )
                return
            # Otherwise add a space after each column
            else:
                entries.append(["{} ".format(str(col)) for col in row])

        # Convert the entries and colors into numpy arrays
        entries = np.array(entries)
        if colors is None:
            colors = np.full(entries.shape, "")
        else:
            colors = [row for row in colors if row]
            colors = np.array(colors)

        # Add horizontal and vertical lines if desired (first vertical to get full lines containing only hyphens)
        if vertical_lines:
            index_to_insert = list(range(0, entries.shape[1] + 1))
            entries = np.insert(entries, index_to_insert[:-1], "| ", axis=1)
            entries = np.insert(entries, entries.shape[1], "|", axis=1)
            colors = np.insert(colors, index_to_insert, "", axis=1)

        if horizontal_lines:
            index_to_insert = list(range(0, entries.shape[0] + 1))
            entries = np.insert(entries, index_to_insert, "-", axis=0)
            colors = np.insert(colors, index_to_insert, "", axis=0)

        for log_string in self._create_tabular_strings(entries, colors, self._terminal_width):
            if self._use_modulus_logger:
                self._modulus_logger.info(log_string)
            else:
                print(log_string, flush=True)

        # Write different style to file
        if self._write_to_file:
            for log_string in self._create_tabular_strings(entries, None, self._logfile_width):
                self._filestream += log_string + "\n"

    def star_line_flush(self) -> IO:
        """Prints a breakline (full starline) to the terminal and file.

        Note:
            Only at this stage data is written to the file. Remember to call this method at least at the end of a module to ensure that
            logging information is written to file.
        """
        if not self.active:
            return None
        stars = f"|***{'*' * self._terminal_width}***|"
        if self._use_modulus_logger:
            self._modulus_logger.info(stars)
        else:
            print(stars, flush=True)
        if self._write_to_file:
            stars = f"|***{'*' * self._logfile_width}***|"
            self._filestream += stars
            print(self._filestream, file=open(str(self._log_file), "a+"))
            self._filestream = ""

    def blank_line(self, number_of_blank_lines: int = 1) -> IO:
        """Writes a blank line.

        Args:
            number_of_blank_lines (int, optional): The number of blank lines to be written. Defaults to 1.
        """
        for _ in range(0, number_of_blank_lines):
            self.write(" ")

    def write_compilation_result(
        self,
        successful: bool,
        prefix_msg: str = "",
        err_msg: str = "",
        time: int = None,
    ) -> IO:
        """Writes an appropriate message for the result of an execution. The status is Done for success and Failed for failure.

        Args:
            successful (bool): If the execution has been successful.
            prefix_msg (str, optional): Additional prefix message before the status. Defaults to "".
            err_msg (str, optional): Additional error message under failure. Defaults to "".
            time (int, optional): The time in seconds required for the execution in case of success. Defaults to None.
        """
        if not successful:
            self.write(
                f"{prefix_msg + (' ' if prefix_msg and prefix_msg[-1] != ' ' else '')}Failed ({err_msg})",
                color="r",
            )
        else:
            time_string = f"(Elapsed time: {string_o.convert_time(time)})" if time is not None else ""
            self.write(
                f"{prefix_msg + (' ' if prefix_msg and prefix_msg[-1] != ' ' else '')}Done {time_string}",
                color="g",
            )

    def write_summary(self, failed: int, successful: int) -> IO:
        """Gives a string for an one-line summary (successful, failed, total).

        Args:
            failed (int): The number failures.
            successful (int): The number of successfull.
        """
        return self.write(f"Summary: successful: {successful}, Failed: {failed}, Total: {failed + successful}")

    def write_numbered_prefix(self, current_number: int, total_number: int, msg: str = "") -> int:
        """Writes a numbered prefix of style [1/20] where 1 is the current number and 20 the total number.
            Additionally, the indent of the logger is increased by the given number representation.

        Args:
            current_number (int): The current number.
            total_number (int): The total number.
            msg (str, optional): An additional message after the number. Defaults to "".

        Returns:
            int: The number of indents the logger has incremented due to the number.
        """
        total_number_length = len(str(total_number))
        current_number_length = len(str(current_number))
        file_prefix = f"[{' ' * (total_number_length - current_number_length)}{current_number}/{total_number}] "
        self.write(f"{file_prefix}{msg}")
        self.indent += len(file_prefix)
        return len(file_prefix)

    def welcome_message(self, log_text: str = "") -> IO:
        """Writes a proper welcome message, including an Alpaca, with additional information.

        Args:
            log_text (str): The additional logging text used for the welcome message (always centered).
        """
        self.star_line_flush()
        self.blank_line(2)
        self.write(",--------.                   ,--.      ", center=True)
        self.write("'--.  .--',---. ,--.--. ,---.|  ,---.  ", center=True)
        self.write("   |  |  | .-. ||  .--'| .--'|  .-.  | ", center=True)
        self.write("   |  |  ' '-' '|  |   \ `--.|  | |  | ", center=True)
        self.write("   `--'   `---' `--'    `---'`--' `--' ", center=True)
        self.write("      ,--.   ,-----.  ,--.   ,--.      ", center=True)
        self.write("      |  |   |  |) /_ |   `.'   |      ", center=True)
        self.write("      |  |   |  .-.  \|  |'.'|  |      ", center=True)
        self.write("      |  '--.|  '--' /|  |   |  |      ", center=True)
        self.write("      `-----'`------' `--'   `--'      ", center=True)
        self.blank_line(2)
        if log_text:
            self.write(log_text, center=True)
            self.blank_line(2)
        self.star_line_flush()

    def bye_message(self, log_text: str = "") -> IO:
        """Writes a proper bye message, including an Alpaca, with additional information.

        Args:
            log_text (str): The additional logging text used for the bye message (always centered).
        """
        if self._write_to_file:
            self.star_line_flush()
            self.blank_line()
            self.write(f"A log file has been generated: {self._log_file}", color="bold")
            self.blank_line()
        self.star_line_flush()
        self.blank_line(2)
        self.write("The simulation finished.", center=True)
        self.blank_line(2)
        if log_text:
            self.write(log_text, center=True)
            self.blank_line(2)
        self.star_line_flush()
