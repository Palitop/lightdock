"""Tests for TOBIA2 scoring function module"""

import pytest
from pathlib import Path
from lightdock.scoring.tobia2.driver import TOBIA2Potential, TOBIA2, TOBIA2Adapter
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex


class TestTOBIA2Potential:
    def test_create_TOBIA2Potential_interface(self):
        potential = TOBIA2Potential()

        assert len(potential.tobi_a_1) == 18
        assert len(potential.tobi_a_2) == 18

        assert -1.02 == pytest.approx(potential.tobi_a_1[0][0])
        assert 10.00 == pytest.approx(potential.tobi_a_1[-1][-1])
        assert 2.73 == pytest.approx(potential.tobi_a_1[1][17])

        assert 0.10 == pytest.approx(potential.tobi_a_2[0][0])
        assert 10.00 == pytest.approx(potential.tobi_a_2[-1][-1])
        assert 1.66 == pytest.approx(potential.tobi_a_2[3][17])


class TestTOBIA2:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"
        self.tobiA2 = TOBIA2()

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1PPElig.pdb", "1PPErec.pdb"),
        ("1PPElig.cif", "1PPErec.cif")
    ])
    def test_calculate_TOBIA2_1PPE(self, lig_file, rec_file):
        io = IOFactory(self.golden_data_path / rec_file).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / rec_file
        )
        receptor = Complex(chains, atoms)
        io = IOFactory(self.golden_data_path / lig_file).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / lig_file
        )
        ligand = Complex(chains, atoms)
        adapter = TOBIA2Adapter(receptor, ligand)
        assert -358.14 == pytest.approx(
            self.tobiA2(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1EAWlig.pdb", "1EAWrec.pdb"),
        ("1EAWlig.cif", "1EAWrec.cif")
    ])
    def test_calculate_TOBIA2_1EAW(self, lig_file, rec_file):
        io = IOFactory(self.golden_data_path / rec_file).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / rec_file
        )
        receptor = Complex(chains, atoms)
        io = IOFactory(self.golden_data_path / lig_file).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / lig_file
        )
        ligand = Complex(chains, atoms)
        adapter = TOBIA2Adapter(receptor, ligand)
        assert -280.54 == pytest.approx(
            self.tobiA2(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1AY7lig.pdb", "1AY7rec.pdb"),
        ("1AY7lig.cif", "1AY7rec.cif")
    ])
    def test_calculate_TOBIA2_1AY7(self, rec_file, lig_file):
        io = IOFactory(self.golden_data_path / rec_file).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / rec_file
        )
        receptor = Complex(chains, atoms)
        io = IOFactory(self.golden_data_path / lig_file).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / lig_file
        )
        ligand = Complex(chains, atoms)
        adapter = TOBIA2Adapter(receptor, ligand)
        assert -230.36 == pytest.approx(
            self.tobiA2(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )

    @pytest.mark.parametrize("protein, peptide", [
        ("1czy_protein.pdb", "1czy_peptide.pdb"),
        ("1czy_protein.cif", "1czy_peptide.cif")
    ])
    def test_calculate_TOBIA2_1CZY(self, protein, peptide):
        io = IOFactory(self.golden_data_path / protein).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / protein
        )
        receptor = Complex(chains, atoms)
        io = IOFactory(self.golden_data_path / peptide).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / peptide
        )
        ligand = Complex(chains, atoms)
        adapter = TOBIA2Adapter(receptor, ligand)
        assert 43.23 == pytest.approx(
            self.tobiA2(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )
