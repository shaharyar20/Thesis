# python modules
import pytest

# torchlbm modules
from torchlbm.enum_type import EnumType
from torchlbm.setup_definitions.type_converters import EnumConverter
from torchlbm.exceptions import TypeConversionError


class ValidEnum(EnumType):
    a = "a"
    b2 = "b2"
    c_d = "c_d"


class InvalidEnumWrongName(EnumType):
    b = "c"


class InvalidEnumNonUniqueNames(EnumType):
    b = "c"
    c = "c"


def test_enum_conversion():
    """Checks the conversion function."""
    with pytest.raises(TypeConversionError):
        EnumConverter(InvalidEnumWrongName)
    with pytest.raises(TypeConversionError):
        EnumConverter(InvalidEnumNonUniqueNames)
    # Enum converter without value restriction
    converter = EnumConverter(ValidEnum)
    assert converter.convert("a") == ValidEnum.a
    assert converter.convert("c_d") == ValidEnum.c_d
    assert converter.convert(ValidEnum.b2) == ValidEnum.b2
    with pytest.raises(TypeConversionError):
        converter.convert("d")
    # Enum converter with value restriction
    converter = EnumConverter(ValidEnum, ["a", "b2"])
    assert converter.convert("a") == ValidEnum.a
    with pytest.raises(TypeConversionError):
        converter.convert("c_d")


def test_enum_comparison():
    """Checks the comparison function."""
    converter = EnumConverter(ValidEnum)
    assert not converter.compare("a", "b2")
    assert converter.compare("b2", "b2")
    assert converter.compare(ValidEnum.b2, "b2")
