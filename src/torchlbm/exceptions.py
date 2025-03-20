class TorchlbmError(Exception):
    """Base class for all exceptions thrown by the torchlbm module."""

    __module__ = "torchlbm"

    def __init__(self, message: str = "") -> None:
        self.message = "\n  ".join([line_msg for line_msg in message.strip().split("\n") if line_msg])
        super().__init__(f"\n  {self.message}")


class SetupError(TorchlbmError):
    """Proxy class for all errors thrown by all setups of the torchlbm module."""

    pass


class TypeConversionError(TorchlbmError):
    """Proxy class for all errors thrown by type conversions of the torchlbm module."""

    pass


class DataReplacementError(TorchlbmError):
    """Proxy class for all errors thrown by the module that replaces data."""

    pass
