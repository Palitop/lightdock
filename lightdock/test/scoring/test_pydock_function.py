"""Tests for CPyDock scoring function module"""

import pytest
from pathlib import Path
from lightdock.scoring.cpydock.driver import CPyDock, CPyDockAdapter
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex


class TestPyDock:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"
        self.pydock = CPyDock()

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1AY7_lig.pdb", "1AY7_rec.pdb")
    ])
    def test_calculate_PyDock_1AY7(self, lig_file, rec_file):
        io = IOFactory(self.golden_data_path / rec_file).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / rec_file
        )
        receptor = Complex(
            chains,
            atoms,
            structure_file_name=(self.golden_data_path / rec_file),
        )
        io = IOFactory(self.golden_data_path / lig_file).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / lig_file
        )
        ligand = Complex(
            chains,
            atoms,
            structure_file_name=(self.golden_data_path / lig_file),
        )

        adapter = CPyDockAdapter(receptor, ligand)
        assert -15.923994756 == pytest.approx(
            self.pydock(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )
