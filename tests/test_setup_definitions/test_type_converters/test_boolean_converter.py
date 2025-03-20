# python modules
import pytest

# torchlbm modules
from torchlbm.setup_definitions.type_converters import BoolConverter
from torchlbm.exceptions import TypeConversionError


def test_bool_conversion():
    """Checks the conversion function."""
    converter = BoolConverter()
    assert isinstance(converter.convert("True"), bool)
    assert converter.convert("TrUe")
    assert converter.convert("True\n")
    assert converter.convert(1)
    assert not converter.convert("FalSe")
    assert not converter.convert(0)
    with pytest.raises(TypeConversionError):
        converter.convert("Trues")
    with pytest.raises(TypeConversionError):
        converter.convert(2)
    assert converter.convert("None") == None


def test_bool_comparison():
    """Checks the comparison function."""
    converter = BoolConverter()
    assert converter.compare("TrUe", "true")
    assert not converter.compare("False", "true")
    assert not converter.compare("False", "None")
