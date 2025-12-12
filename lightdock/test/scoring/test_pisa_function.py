"""Tests for PISA scoring function module"""

import pytest
from pathlib import Path
from lightdock.scoring.pisa.driver import PISAPotential, PISA, PISAAdapter
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex


class TestPISAPotential:
    def test_create_PISAPotential_interface(self):
        potential = PISAPotential()
        assert potential is not None


class TestPISA:
    """Original PISA scoring energy goes from negative to positive"""

    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"
        self.pisa = PISA()

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1PPElig.pdb", "1PPErec.pdb"),
        ("1PPElig.cif", "1PPErec.cif")
    ])
    def test_calculate_PISA_1PPE(self, lig_file, rec_file):
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
        adapter = PISAAdapter(receptor, ligand)
        assert -0.4346 == pytest.approx(
            round(
                self.pisa(
                    adapter.receptor_model,
                    adapter.receptor_model.coordinates[0],
                    adapter.ligand_model,
                    adapter.ligand_model.coordinates[0],
                ),
                4
            )
        )

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1EAWlig.pdb", "1EAWrec.pdb"),
        ("1EAWlig.cif", "1EAWrec.cif")
    ])
    def test_calculate_PISA_1EAW(self, lig_file, rec_file):
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
        adapter = PISAAdapter(receptor, ligand)
        assert -0.2097 == pytest.approx(
            round(
                self.pisa(
                    adapter.receptor_model,
                    adapter.receptor_model.coordinates[0],
                    adapter.ligand_model,
                    adapter.ligand_model.coordinates[0],
                ),
                4
            )
        )

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1AY7lig.pdb", "1AY7rec.pdb"),
        ("1AY7lig.cif", "1AY7rec.cif")
    ])
    def test_calculate_PISA_1AY7(self, lig_file, rec_file):
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
        adapter = PISAAdapter(receptor, ligand)
        assert -0.2141 == pytest.approx(
            round(
                self.pisa(
                    adapter.receptor_model,
                    adapter.receptor_model.coordinates[0],
                    adapter.ligand_model,
                    adapter.ligand_model.coordinates[0],
                ),
                4
            )
        )
