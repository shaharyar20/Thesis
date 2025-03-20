# python modules
import pytest
from pathlib import Path

# torchlbm modules
from torchlbm.setup_definitions.type_converters import ClassConverter
from torchlbm.exceptions import TypeConversionError


def test_conversion():
    """Checks the conversion function."""
    # Without any restrictions
    converter = ClassConverter()
    assert converter.convert("a") == "a"
    assert converter.convert(1.0) == 1.0
    assert converter.convert(Path("")) == Path("")
    # With class restriction
    converter = ClassConverter([float, int])
    with pytest.raises(TypeConversionError):
        converter.convert(Path(""))
    assert converter.convert(1.0) == 1.0
    assert converter.convert(1) == 1
