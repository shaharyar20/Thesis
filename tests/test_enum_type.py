# python modules
import pytest

# torchlbm modules
from torchlbm.enum_type import EnumType
from torchlbm.exceptions import TypeConversionError


def test_is_valid():
    """Checks the validity function of enum types."""
    assert EnumType.is_valid("AlphaString") == True
    assert EnumType.is_valid("12345") == True
    assert EnumType.is_valid("Digit12345") == True
    assert EnumType.is_valid("A_B") == True
    assert EnumType.is_valid("A\nB") == False
    assert EnumType.is_valid("A|B") == False


def test_generic_enum_type():
    """Tests generic EnumType definitions."""
    # Try to define a generic with incorrect values
    with pytest.raises(TypeConversionError):

        class TestEnum(EnumType):
            a = "Alpha1234"
            b = "A|B"

    # Define a generic EnumType class and test functionality
    class TestEnum(EnumType):
        a = "FirstName"
        b = "SecondName"
        c = "ThirdName"

    assert str(TestEnum.a) == "FirstName"
    assert repr(TestEnum.b) == "SecondName"
    assert TestEnum.has_value("FirstName") == True
    assert TestEnum.has_value("secondNAME") == True
    assert TestEnum.has_value("firstNames") == False
    assert TestEnum.get_value("Firstname") == TestEnum.a
    assert TestEnum.b in TestEnum.get_values(["SecondName", "Firstname"])
    assert TestEnum.a in TestEnum.get_values(["SecondName", "Firstname"])
    assert TestEnum.c not in TestEnum.get_values(["SecondName", "Firstname"])
    with pytest.raises(TypeConversionError):
        TestEnum.get_value("Firstnames")
