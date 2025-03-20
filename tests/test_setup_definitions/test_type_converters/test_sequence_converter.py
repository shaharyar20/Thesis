# python modules
import pytest

# torchlbm modules
from torchlbm.setup_definitions.type_converters import (
    TupleConverter,
    ListConverter,
    IntConverter,
)
from torchlbm.exceptions import TypeConversionError


def test_tuple_conversion():
    """Checks the conversion function."""
    # Tuple with incorrect value of elements
    with pytest.raises(TypeConversionError):
        TupleConverter(IntConverter(0, 2, False), None)
    with pytest.raises(TypeConversionError):
        TupleConverter(IntConverter(0, 2, False), 1.0)
    # tuple converter with float values and 3 elements
    converter = TupleConverter(IntConverter(0, 2, False), 3)
    with pytest.raises(TypeConversionError):
        converter.convert((1, 2))
    assert isinstance(converter.convert((0, 1, 2)), tuple)
    assert converter.convert((0, 1, 2)) == (0, 1, 2)
    assert isinstance(converter.convert([0, 1, 2]), tuple)
    assert converter.convert([0, 1, 2]) == (0, 1, 2)
    # Test if single values lead to the same result
    converter = TupleConverter(IntConverter(0, 2, False), 1)
    assert converter.convert(1) == (1,)


def test_list_conversion():
    """Checks the conversion function."""
    # List converter with undefined number of elements
    converter = ListConverter(IntConverter(0, 2, False))
    assert converter.convert((0, 1, 2)) == [0, 1, 2]
    assert isinstance(converter.convert((0, 1, 2)), list)
    assert converter.convert([0, 1, 2]) == [0, 1, 2]
    assert isinstance(converter.convert([0, 1, 2]), list)
    # List converter with minimum number of elements
    converter = ListConverter(IntConverter(0, 2, False), min_elements=2)
    with pytest.raises(TypeConversionError):
        converter.convert((1,))
    assert converter.convert((0, 1, 2, 1, 2, 1)) == [0, 1, 2, 1, 2, 1]
    assert converter.convert([0, 2, 1, 2]) == [0, 2, 1, 2]
    # List converter with maximum number of elements
    converter = ListConverter(IntConverter(0, 2, False), max_elements=4)
    with pytest.raises(TypeConversionError):
        converter.convert((0, 1, 2, 1, 2, 1))
    assert converter.convert([]) == []
    assert converter.convert([0, 2, 1, 2]) == [0, 2, 1, 2]
    # Test if single values lead to the same result
    assert converter.convert(1) == [1]


def test_tuple_comparison():
    """Checks the comparison function."""
    # Tuple converter
    converter = TupleConverter(IntConverter(0, 2, False), 3)
    assert not converter.compare((0, 1, 2), (0, 2, 1))
    assert converter.compare((1, 1, 2), (1, 1, 2))


def test_list_comparison():
    """Checks the comparison function."""
    # Tuple converter
    converter = ListConverter(IntConverter(0, 2, False), 3)
    assert not converter.compare((0, 1, 2), (0, 2, 1))
    assert converter.compare((1, 1, 2), (1, 1, 2))
