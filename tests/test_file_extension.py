# python modules
from pathlib import Path

# torchlbm modules
from torchlbm.file_extension import FileExtension


def test_from_path():
    assert FileExtension.from_path(Path("test.png")) == FileExtension.png
    assert FileExtension.from_path(Path("test.png.log")) == FileExtension.log
    assert FileExtension.from_path(Path("test.no_ext")) == None


def test_format():
    assert FileExtension.png.format() == ".png"
