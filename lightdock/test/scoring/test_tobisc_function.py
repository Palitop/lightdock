"""Tests for TOBISC scoring function module"""

import pytest
from pathlib import Path
from lightdock.scoring.tobisc.driver import TOBISCPotential, TOBISC, TOBISCAdapter
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex


class TestTOBISCPotential:
    def test_create_TOBISCPotential_interface(self):
        potential = TOBISCPotential()

        assert len(potential.tobi_sc_1) == 22
        assert len(potential.tobi_sc_2) == 22

        assert -0.59 == pytest.approx(potential.tobi_sc_1[0][0])
        assert -0.09 == pytest.approx(potential.tobi_sc_1[-1][-1])
        assert 1.37 == pytest.approx(potential.tobi_sc_1[1][20])

        assert -0.58 == pytest.approx(potential.tobi_sc_2[0][0])
        assert -0.24 == pytest.approx(potential.tobi_sc_2[-1][-1])
        assert 0.39 == pytest.approx(potential.tobi_sc_2[3][20])


class TestTOBISC:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"
        self.tobisc = TOBISC()

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1PPElig.pdb", "1PPErec.pdb"),
        ("1PPElig.cif", "1PPErec.cif")
    ])
    def test_calculate_TOBISC_1PPE(self, lig_file, rec_file):
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
        adapter = TOBISCAdapter(receptor, ligand)
        assert 17.58 == pytest.approx(
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
    def test_calculate_TOBISC_1EAW(self, lig_file, rec_file):
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
        adapter = TOBISCAdapter(receptor, ligand)
        assert -9.87 == pytest.approx(
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
    def test_calculate_TOBISC_1AY7(self, rec_file, lig_file):
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
        adapter = TOBISCAdapter(receptor, ligand)
        assert 2.34 == pytest.approx(
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
    def test_calculate_TOBISC_1CZY(self, protein, peptide):
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
        adapter = TOBISCAdapter(receptor, ligand)
        assert 16.28 == pytest.approx(
            self.tobisc(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )
