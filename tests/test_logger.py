# python modules
# torchlbm modules
from torchlbm.logger import Logger


def test_indent():
    """Tests the indent getter and setter of the logger."""
    logger = Logger()
    logger.indent = 0
    # Indent assignment
    logger.indent = 2
    assert logger.indent == 2
    # Indent summation
    logger.indent += 6
    assert logger.indent == 8
    # Indent multiplication
    logger.indent *= 5
    assert logger.indent == 40
    # Indent division
    logger.indent /= 6
    assert logger.indent == 6
    # Indent subtraction (indent must be 0 at the end of function due to Singleton class)
    logger.indent -= 6
    assert logger.indent == 0
    # Indent subtraction (indent must be 0 at the end of function due to Singleton class)
    logger.indent = -1
    assert logger.indent == 0


def test_create_log_strings():
    """Tests the _create_log_string() function of the logger."""
    logger = Logger()
    logger.indent = 0
    # Single string
    log_strings = logger._create_log_strings("Hallo", 5, color=None, center=False)
    assert len(log_strings) == 1
    assert log_strings[0] == f"|*  Hallo  *|"
    # List of strings (each in a new line)
    log_strings = logger._create_log_strings(["a", "bb", "ccc"], 5, color=None, center=False)
    assert len(log_strings) == 3
    assert log_strings == [
        f"|*  a{' ' * 4}  *|",
        f"|*  bb{' ' * 3}  *|",
        f"|*  ccc{' ' * 2}  *|",
    ]
    # Multiline string (after each \n a new line is started)
    log_strings = logger._create_log_strings("a\nbb\n\nccc", 5, color=None, center=False)
    assert len(log_strings) == 4
    assert log_strings == [
        f"|*  a{' ' * 4}  *|",
        f"|*  bb{' ' * 3}  *|",
        f"|*  {' ' * 5}  *|",
        f"|*  ccc{' ' * 2}  *|",
    ]
    # Indented string
    logger.indent = 2
    log_strings = logger._create_log_strings("a\nbb", 5, color=None, center=False)
    assert len(log_strings) == 2
    assert log_strings == [
        f"|*  {' ' * 2}a{' ' * 2}  *|",
        f"|*  {' ' * 2}bb{' ' * 1}  *|",
    ]
    logger.indent = 0
    # Longer than one line string
    log_strings = logger._create_log_strings("a" * 6, 5, color=None, center=False)
    assert len(log_strings) == 2
    assert log_strings == [f"|*  {'a' * 5}  *|", f"|*  a{' ' * 4}  *|"]


def test_write(capsys):
    """Tests the write() function of the logger."""
    logger = Logger()
    logger.indent = 0
    # Single string
    logger.write("Hello")
    out, _ = capsys.readouterr()
    assert len(out) == 81
    assert out == f"|*  Hello {' ' * 66}  *|\n"
    # Single full line string
    logger.write("b" * 72)
    out, _ = capsys.readouterr()
    assert len(out) == 81
    assert out == f"|*  {'b' * 72}  *|\n"
    # Longer than one line
    logger.write("b" * 74)
    out, _ = capsys.readouterr()
    assert len(out) == 2 * 81
    assert out == f"|*  {'b' * 72}  *|\n" + f"|*  {'b' * 2}{' ' * 70}  *|\n"

    logger.write("b\na")
    out, _ = capsys.readouterr()
    assert len(out) == 2 * 81
    assert out == f"|*  b{' ' * 71}  *|\n" + f"|*  a{' ' * 71}  *|\n"

    logger.write("TwelveString", center=True)
    out, _ = capsys.readouterr()
    assert len(out) == 81
    assert out == f"|*  {' ' * 30}TwelveString{' ' * 30}  *|\n"


def test_star_line_flush(capsys):
    """Tests the star_line_flush() function of the logger."""
    logger = Logger()
    logger.indent = 0
    # Single star line
    logger.star_line_flush()
    out, _ = capsys.readouterr()
    assert len(out) == 81
    assert out == f"|{'*' * 78}|\n"


def test_blank_line(capsys):
    """Tests the blank_line() function of the logger."""
    logger = Logger()
    logger.indent = 0
    # Single blank line
    logger.blank_line()
    out, _ = capsys.readouterr()
    assert len(out) == 81
    assert out == f"|*{' ' * 76}*|\n"

    # Multiple blank line
    logger.blank_line(3)
    out, _ = capsys.readouterr()
    assert len(out) == 3 * 81
    assert out == f"|*{' ' * 76}*|\n" * 3


def test_write_compilation_result(capsys):
    """Tests the write_compilation_result() of the logger."""
    logger = Logger()
    logger.indent = 0
    # Successfull without prefix message
    logger.write_compilation_result(True, prefix_msg="", err_msg="", time=None)
    out, _ = capsys.readouterr()
    assert out == f"|*  \033[92mDone{' ' * 68}\033[0m  *|\n"
    # Successfull with prefix message
    logger.write_compilation_result(True, prefix_msg="Prefix", err_msg="", time=None)
    out, _ = capsys.readouterr()
    assert out == f"|*  \033[92mPrefix Done{' ' * 61}\033[0m  *|\n"
    # Failure without prefix and error message
    logger.write_compilation_result(False, prefix_msg="", err_msg="", time=None)
    out, _ = capsys.readouterr()
    assert out == f"|*  \033[91mFailed (){' ' * 63}\033[0m  *|\n"
    # Failure with prefix and error message
    logger.write_compilation_result(False, prefix_msg="Prefix", err_msg="Error", time=None)
    out, _ = capsys.readouterr()
    assert out == f"|*  \033[91mPrefix Failed (Error){' ' * 51}\033[0m  *|\n"


def test_write_summary(capsys):
    """Tests the write_summary() function of the logger."""
    logger = Logger()
    logger.indent = 0
    logger.write_summary(6, 5)
    out, _ = capsys.readouterr()
    assert len(out) == 81
    assert out == f"|*  Summary: successful: 5, Failed: 6, Total: 11{' ' * 28}  *|\n"


def test_write_numbered_prefix(capsys):
    """Tests the write_numbered_prefix() function of the logger."""
    logger = Logger()
    logger.indent = 0
    logger.write_numbered_prefix(1, 10, "Message")
    out, _ = capsys.readouterr()
    assert logger.indent == 8
    assert len(out) == 81
    assert out == f"|*  [ 1/10] Message{' ' * 57}  *|\n"
