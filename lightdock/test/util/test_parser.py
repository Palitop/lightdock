"""Tests for parser module"""

import pytest
import os
import argparse
from pathlib import Path
from lightdock.util.parser import (
    get_lightdock_structures,
    valid_file,
    valid_integer_number,
    valid_natural_number,
    valid_float_number,
)
from lightdock.constants import (
    DEFAULT_LIGHTDOCK_PREFIX
)


class TestParserUtils:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"

    @pytest.mark.parametrize("list_file, files, single_file", [
        ("pdbs.list", ["lightdock_1czy_protein.pdb", "lightdock_1czy_peptide.pdb"], "1czy_protein.pdb"),
        ("cifs.list", ["lightdock_1czy_protein.cif", "lightdock_1czy_peptide.cif"], "1czy_protein.cif")
    ])
    def test_get_lightdock_structures(self, list_file, files, single_file):
        os.chdir(self.golden_data_path)
        # First test list of structures
        list_of_pdbs = self.golden_data_path / list_file
        structures = get_lightdock_structures(list_of_pdbs)

        assert structures == files

        # Test single structure
        single_pdb = self.golden_data_path / single_file
        structures = get_lightdock_structures(single_pdb)

        assert Path(structures[0]).name == DEFAULT_LIGHTDOCK_PREFIX % single_file

    @pytest.mark.parametrize("file", ["lightdock_1czy_protein.pdb", "lightdock_1czy_protein.cif"])
    def test_valid_file_ok(self, file):
        os.chdir(self.golden_data_path)

        filename = valid_file(file)

        assert filename == file

    @pytest.mark.parametrize("file", ["1czy_protein.pdb", "1czy_protein.cif"])
    def test_valid_file_ko(self, file):
        with pytest.raises(argparse.ArgumentTypeError):
            os.chdir(self.golden_data_path)

            valid_file("1czy_protein.pdb")

    def test_valid_integer_number_ok(self):
        assert valid_integer_number("1")

    def test_valid_integer_number_ko(self):
        with pytest.raises(argparse.ArgumentTypeError):
            assert valid_natural_number("aa") == 0

    def test_valid_integer_number_ko_1(self):
        with pytest.raises(argparse.ArgumentTypeError):
            assert not valid_integer_number("aa")

    def test_valid_integer_number_ko_2(self):
        with pytest.raises(argparse.ArgumentTypeError):
            assert valid_integer_number("0") == 0

    def test_valid_natural_number_ok(self):
        assert valid_natural_number("1") == 1

    def test_valid_natural_number_ok_2(self):
        assert valid_natural_number("0") == 0

    def test_valid_natural_number_ko(self):
        with pytest.raises(argparse.ArgumentTypeError):
            assert valid_natural_number("-1") == 0

    def test_valid_float_number_ok(self):
        assert valid_float_number("1.0") == 1.0

    def test_valid_float_number_ko_1(self):
        with pytest.raises(argparse.ArgumentTypeError):
            assert valid_float_number("aa") == 0.0

    def test_valid_float_number_ko_2(self):
        with pytest.raises(argparse.ArgumentTypeError):
            assert valid_float_number("-1.0") == -1.0
