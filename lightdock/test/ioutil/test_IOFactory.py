"""Test IOFactory class"""

import pytest
from unittest.mock import patch
from pathlib import Path
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.ioutil.IOExceptions import (
    UnsupportedFileTypeError,
    EmptyPathError,
    FolderPathError
)
from lightdock.ioutil.MMCIFIO import MMCIFIO
from lightdock.ioutil.PDBIO import PDBIO


class TestIOFactory():

    @pytest.mark.parametrize("file_name, expected_class", [
        (Path("1CRN.cif"), MMCIFIO),
        (Path("1CRN.mmcif"), MMCIFIO),
        (Path("1CRN.pdb"), PDBIO)
    ])
    def test_get_instance(self, file_name: Path, expected_class):
        print("Tipo ", type(file_name))
        with patch.object(
            Path,
            "exists", return_value=True
        ):
            io = IOFactory(file_name).get_instance()
            assert type(io) is expected_class

    def test_get_instance_folder_path(self):
        with patch.object(
            Path,
            "is_dir", return_value=True
        ):
            with pytest.raises(FolderPathError):
                file_name = Path("ioutil/")
                _ = IOFactory(file_name).get_instance()

    def test_get_instance_file_not_exists(self):
        with pytest.raises(EmptyPathError):
            file_name = ""
            _ = IOFactory(file_name).get_instance()

    @pytest.mark.parametrize("file_name", [Path("1CRN.png"), Path("1CRN.jpg"), Path("1CRN.pdf"), Path("1CRN.html")])
    def test_get_instance_unsupported_format(self, file_name: Path):
        with patch.multiple(
            Path,
            exists=lambda self: True,
            is_dir=lambda self: False
        ):
            with pytest.raises(UnsupportedFileTypeError):
                _ = IOFactory(file_name).get_instance()

    @pytest.mark.parametrize("file_name, result", [
        (Path("1CRN.cif"), "cif"),
        (Path("1CRN.pdb"), "pdb"),
        (Path("1CRN.mmcif"), "mmcif")
    ])
    def test_get_file_type(self, file_name: str, result: str):
        file_type = IOFactory.get_file_type(file_name)
        assert file_type == result
