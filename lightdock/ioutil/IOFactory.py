"""Class to handle the creation of different type of file managers"""

from pathlib import Path
from lightdock.ioutil.IO import IO
from lightdock.ioutil.MMCIFIO import MMCIFIO
from lightdock.ioutil.PDBIO import PDBIO
from lightdock.ioutil.IOExceptions import (
    EmptyPathError,
    UnsupportedFileTypeError,
    FolderPathError
)


class IOFactory:

    def __init__(self, file_name: Path):
        self.file_name: Path = file_name

    def get_instance(self) -> IO:
        if type(self.file_name) is str:
            if not self.file_name:
                raise EmptyPathError()

        if isinstance(self.file_name, Path):
            if self.file_name.is_dir():
                raise FolderPathError()

        file_type = IOFactory.get_file_type(self.file_name)

        if file_type in ["mmcif", "cif"]:
            return MMCIFIO()

        if file_type == "pdb":
            return PDBIO()

        raise UnsupportedFileTypeError(file_type)

    @staticmethod
    def get_file_type(file_name: Path) -> str:
        return str(file_name).split('.')[-1].lower()
