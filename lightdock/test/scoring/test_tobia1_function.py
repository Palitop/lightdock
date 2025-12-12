"""Tests for TOBIA1 scoring function module"""

import pytest
from pathlib import Path
from lightdock.scoring.tobia1.driver import TOBIA1Potential, TOBIA1, TOBIA1Adapter
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex


class TestTOBISCPotential:
    def test_create_TOBIA1Potential_interface(self):
        potential = TOBIA1Potential()

        assert len(potential.tobi_a_1) == 18
        assert len(potential.tobi_a_2) == 18

        assert 1.72 == pytest.approx(potential.tobi_a_1[0][0])
        assert 0.00 == pytest.approx(potential.tobi_a_1[-1][-1])
        assert 4.18 == pytest.approx(potential.tobi_a_1[1][17])

        assert 0.29 == pytest.approx(potential.tobi_a_2[0][0])
        assert 10.00 == pytest.approx(potential.tobi_a_2[-1][-1])
        assert 0.45 == pytest.approx(potential.tobi_a_2[3][17])


class TestTOBIA1:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"
        self.tobia1 = TOBIA1()

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1PPElig.pdb", "1PPErec.pdb"),
        ("1PPElig.cif", "1PPErec.cif")
    ])
    def test_calculate_TOBIA1_1PPE(self, lig_file, rec_file):
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
        adapter = TOBIA1Adapter(receptor, ligand)
        assert -470.14 == pytest.approx(
            self.tobia1(
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
    def test_calculate_TOBIA1_1EAW(self, lig_file, rec_file):
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
        adapter = TOBIA1Adapter(receptor, ligand)
        assert -350.12 == pytest.approx(
            self.tobia1(
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
    def test_calculate_TOBIA1_1AY7(self, rec_file, lig_file):
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
        adapter = TOBIA1Adapter(receptor, ligand)
        assert -358.04 == pytest.approx(
            self.tobia1(
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
    def test_calculate_TOBIA1_1CZY(self, protein, peptide):
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
        adapter = TOBIA1Adapter(receptor, ligand)
        assert 54.24 == pytest.approx(
            self.tobia1(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )
