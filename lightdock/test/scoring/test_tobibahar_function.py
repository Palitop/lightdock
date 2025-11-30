"""Tests for TOBIBAHAR scoring function module"""

import pytest
from pathlib import Path
from lightdock.scoring.tobibahar.driver import TOBIBAHARPotential, TOBIBAHAR, TOBIBAHARAdapter
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex


class TestTOBIBAHARPotential:
    def test_create_TOBIBAHARPotential_interface(self):
        potential = TOBIBAHARPotential()

        assert len(potential.tobibahar) == 22

        assert -3.56 == pytest.approx(potential.tobibahar[0][0])
        assert 1.82 == pytest.approx(potential.tobibahar[-1][-1])
        assert 1.6 == pytest.approx(potential.tobibahar[1][20])


class TestTOBIBAHAR:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"
        self.tobisc = TOBIBAHAR()

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1PPElig.pdb", "1PPErec.pdb"),
        ("1PPElig.cif", "1PPErec.cif")
    ])
    def test_calculate_TOBIBAHAR_1PPE(self, lig_file, rec_file):
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
        adapter = TOBIBAHARAdapter(receptor, ligand)
        assert 27.67 == pytest.approx(
            self.tobisc(
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
    def test_calculate_TOBIBAHAR_1EAW(self, lig_file, rec_file):
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
        adapter = TOBIBAHARAdapter(receptor, ligand)
        assert -14.64 == pytest.approx(
            self.tobisc(
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
    def test_calculate_TOBIBAHAR_1AY7(self, rec_file, lig_file):
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
        adapter = TOBIBAHARAdapter(receptor, ligand)
        assert -65.94 == pytest.approx(
            self.tobisc(
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
    def test_calculate_TOBIBAHAR_1CZY(self, protein, peptide):
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
        adapter = TOBIBAHARAdapter(receptor, ligand)
        assert 14.70 == pytest.approx(
            self.tobisc(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )
