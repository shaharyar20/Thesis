# python modules
import pytest

# torchlbm modules
from torchlbm.setup_definitions.type_converters import FloatConverter, IntConverter
from torchlbm.exceptions import TypeConversionError


def test_int_conversion():
    """Checks the conversion function."""
    # Int without any restriction
    converter = IntConverter()
    with pytest.raises(TypeConversionError):
        converter.convert(1.0)
    with pytest.raises(TypeConversionError):
        converter.convert("test")
    assert converter.convert(1) == 1
    assert isinstance(converter.convert(1), int)
    # Int with min and max restriction
    converter = IntConverter(5, 10)
    with pytest.raises(TypeConversionError):
        converter.convert(0)
    with pytest.raises(TypeConversionError):
        converter.convert(11)
    assert converter.convert(7) == 7
    # Int with min and max restriction but cut range
    converter = IntConverter(5, 10, cut_range=True)
    assert converter.convert(-5) == 5
    assert converter.convert(13) == 10
    assert converter.convert(6) == 6


def test_float_conversion():
    """Checks the conversion function."""
    # Int without any restriction
    converter = FloatConverter()
    with pytest.raises(TypeConversionError):
        converter.convert("test")
    assert converter.convert(1.0) == pytest.approx(1.0)
    assert converter.convert(1) == pytest.approx(1.0)
    assert isinstance(converter.convert(1), float)
    assert isinstance(converter.convert(1.0), float)
    # Int with min and max restriction
    converter = FloatConverter(3.2, 10.125)
    with pytest.raises(TypeConversionError):
        converter.convert(3.199)
    with pytest.raises(TypeConversionError):
        converter.convert(10.126)
    assert converter.convert(6.146) == pytest.approx(6.146)
    # Int with min and max restriction but cut range
    converter = FloatConverter(3.2, 10.125, cut_range=True)
    assert converter.convert(-4) == pytest.approx(3.2)
    assert converter.convert(13) == pytest.approx(10.125)


def test_int_comparison():
    """Checks the comparison function."""
    # Tuple converter
    converter = IntConverter()
    assert not converter.compare(2, 5)
    assert converter.compare(2, 2)


def test_float_comparison():
    """Checks the comparison function."""
    # Tuple converter
    converter = FloatConverter()
    assert not converter.compare(2.0, 5.0)
    assert converter.compare(2.0, 2.0)
