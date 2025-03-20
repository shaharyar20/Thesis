#!/usr/bin/env python3
# Python modules
from pathlib import Path

# own modules
from .enum_type import EnumType


class FileExtension(EnumType):
    """Enum class for all file extensions that can be used in the module. Add new file extensions here.

    Args:
        EnumType (EnumType): The base class of all enum classes.
    """

    png = "png"
    jpg = "jpg"
    jpeg = "jpeg"
    tex = "tex"
    pdf = "pdf"
    mp4 = "mp4"
    avi = "avi"
    csv = "csv"
    log = "log"
    h5 = "h5"
    hdf5 = "hdf5"
    xdmf = "xdmf"
    toml = "toml"
    py = "py"
    xml = "xml"

    def format(self) -> str:
        """Formats the extension with a leading dot.

        Returns:
            str: The formattted extension.
        """
        return f".{self.value}" if self.value[0] != "." else self.value

    @classmethod
    def from_path(cls, file_path: Path) -> "FileExtension":
        """Gives the FileExtension value for a given path.

        Returns:
            FileExtension: The file extension value.
        """
        file_ext = file_path.suffix.strip(".")
        if file_ext.lower() in list(map(lambda c: c.value, cls)):
            return cls[file_ext.lower()]
        else:
            return None
