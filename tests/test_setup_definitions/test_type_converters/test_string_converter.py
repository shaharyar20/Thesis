# python modules
import pytest

# torchlbm modules
from torchlbm.setup_definitions.type_converters import StringConverter
from torchlbm.exceptions import TypeConversionError


def test_conversion():
    """Checks the conversion function."""
    # Check failed instantiations
    with pytest.raises(TypeConversionError):
        StringConverter(prohibited_characters=["ie"])
    with pytest.raises(TypeConversionError):
        StringConverter(allowed_characters=["ie"])
    # Check string converter with prohibited characters
    converter = StringConverter(prohibited_characters=["i", "e"])
    with pytest.raises(TypeConversionError):
        converter.convert("WillFail")
    with pytest.raises(TypeConversionError):
        converter.convert("NoSuccess")
    assert converter.convert("Pass") == "Pass"
    assert converter.convert(" Pass\n\n") == "Pass"
    assert isinstance(converter.convert(" Pass\n\n"), str)
    # Check string converter with allowed characters
    converter = StringConverter(allowed_characters=["i", "e", "c"], case_sensitive=False)
    with pytest.raises(TypeConversionError):
        converter.convert("WillFail")
    with pytest.raises(TypeConversionError):
        converter.convert("NoSuccess")
    assert converter.convert("Ice") == "Ice"
    assert converter.convert("ice") == "ice"
    # Check string converter with allowed strings (case insensitive)
    converter = StringConverter(allowed_strings=["Fail", "Success"], case_sensitive=False)
    with pytest.raises(TypeConversionError):
        converter.convert("WillFail")
    with pytest.raises(TypeConversionError):
        converter.convert("NoSuccess")
    assert converter.convert("Fail") == "Fail"
    assert converter.convert("Success") == "Success"
    assert converter.convert("fail") == "Fail"
    assert converter.convert("succeSs") == "Success"
    # Check string converter with allowed strings (case sensitive)
    converter = StringConverter(allowed_strings=["Fail", "Success"], case_sensitive=True)
    with pytest.raises(TypeConversionError):
        converter.convert("fail")
    with pytest.raises(TypeConversionError):
        converter.convert("succeSs")
    assert converter.convert("Fail") == "Fail"
    assert converter.convert("Success") == "Success"
    # Check case insensitive
    converter = StringConverter(case_sensitive=False)
    assert converter.convert("Pass") == "Pass"
    assert converter.convert("PASS") == "PASS"


def test_string_comparison():
    """Checks the comparison function."""
    string1 = "test"
    string2 = "test1"
    string3 = "test"
    # File converter without file restrictions
    converter = StringConverter()
    assert not converter.compare(string1, string2)
    assert converter.compare(string1, string3)
