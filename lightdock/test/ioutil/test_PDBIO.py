"""Tests for PDBReader module"""

import pytest
import filecmp
from pathlib import Path
from lightdock.ioutil.PDBIO import PDBIO
from lightdock.structure.complex import Complex
from lightdock.prep.starting_points import points_on_sphere
from lightdock.error.lightdock_errors import PDBParsingError


class TestPDBReader:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"

    def test_read_wrong_atom_line(self):
        pdb_io = PDBIO()
        with pytest.raises(PDBParsingError):
            line = "ATOM     11  NH2 ARG A   1       2.559  16.752     AA   1.00 14.90           N"
            pdb_io.read_atom_line(line, "ATOM")
            assert False

    def test_read_nan_in_atom_line(self):
        pdb_io = PDBIO()
        with pytest.raises(PDBParsingError):
            line = "ATOM     11  NH2 ARG A   1       2.559  16.752     NaN  1.00 14.90           N"
            pdb_io.read_atom_line(line, "ATOM")
            assert False

    def test_wrong_atom_number(self):
        pdb_io = PDBIO()
        with pytest.raises(PDBParsingError):
            line = "ATOM     OO  NH2 ARG A   1       2.559  16.752    1.00  1.00 14.90           N"
            pdb_io.read_atom_line(line, "ATOM")
            assert False

    def test_wrong_residue_number(self):
        pdb_io = PDBIO()
        with pytest.raises(PDBParsingError):
            line = "ATOM     12  NH2 ARG A   A       2.559  16.752    1.00  1.00 14.90           N"
            pdb_io.read_atom_line(line, "ATOM")
            assert False

    def test_default_occupancy(self):
        pdb_io = PDBIO()
        line = "ATOM     12  NH2 ARG A   1       2.559  16.752    1.00       14.90           N"
        atom = pdb_io.read_atom_line(line, "ATOM")

        assert 1.0 == pytest.approx(atom.occupancy)

    def test_default_b_factor(self):
        pdb_io = PDBIO()
        line = "ATOM     12  NH2 ARG A   1       2.559  16.752    1.00  1.00                 N"
        atom = pdb_io.read_atom_line(line, "ATOM")

        0.0 == pytest.approx(atom.b_factor)

    def test_default_no_line_type(self):
        pdb_io = PDBIO()
        line = "ATOM     12  NH2 ARG A   1       2.559  16.752    1.00  1.00                 N"
        atom = pdb_io.read_atom_line(line)

        assert atom.__class__.__name__ == "Atom"

    def test_parse_complex_from_file(self):
        pdb_io = PDBIO()
        atoms, residues, chains = pdb_io.parse_complex_from_file(
            self.golden_data_path / "1PPE_l_u.pdb"
        )
        assert len(atoms) == 224
        assert len(residues) == 31
        assert len(chains) == 2
        assert len(chains[0].residues) == 29
        assert len(chains[1].residues) == 2

    def test_parse_multi_model_from_file(self):
        pdb_io = PDBIO()
        atoms, residues, chains = pdb_io.parse_complex_from_file(
            self.golden_data_path / "multi_model.pdb"
        )
        assert len(atoms) == 11
        assert len(residues) == 1
        assert len(chains) == 1

    def test_write_to_file(self, tmp_path):
        pdb_io = PDBIO()
        atoms, _, chains = pdb_io.parse_complex_from_file(
            self.golden_data_path / "1PPE_l_u.pdb"
        )
        protein = Complex(chains)
        assert atoms == protein.atoms

        pdb_io.write_to_file(
            protein, tmp_path / "parsed_1PPE_l_u.pdb", protein.atom_coordinates[0]
        )

        assert filecmp.cmp(
            self.golden_data_path / "parsed_1PPE_l_u.pdb",
            tmp_path / "parsed_1PPE_l_u.pdb",
        )

    def test_parse_pdb_noh(self, tmp_path):
        pdb_io = PDBIO()
        atoms_to_ignore = ["H"]
        atoms, _, chains = pdb_io.parse_complex_from_file(
            self.golden_data_path / "1PPE_lig_with_H.pdb", atoms_to_ignore
        )
        protein = Complex(chains)
        assert atoms == protein.atoms

        pdb_io.write_to_file(
            protein,
            tmp_path / "parsed_1PPE_lig_with_H.pdb",
            protein.atom_coordinates[0],
        )

        assert filecmp.cmp(
            self.golden_data_path / "parsed_1PPE_lig_with_H.pdb",
            tmp_path / "parsed_1PPE_lig_with_H.pdb",
        )

    def test_create_file_from_points(self):
        pdb_io = PDBIO()
        output_file = self.golden_data_path / "points.pdb"
        points = points_on_sphere(100)
        pdb_io.create_file_from_points(output_file, points)

        assert output_file.exists()
