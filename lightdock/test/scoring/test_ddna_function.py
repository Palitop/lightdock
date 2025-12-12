"""Tests for DDNA scoring function module"""

import pytest
from pathlib import Path
from lightdock.scoring.ddna.driver import DDNA, DDNAAdapter
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex


class TestDDNA:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"
        self.dna = DDNA()

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1azp_dna.pdb", "1azp_prot.pdb"),
        ("1azp_dna.cif", "1azp_prot.cif")
    ])
    def test_calculate_DNA_1AZP(self, lig_file, rec_file):
        io = IOFactory(self.golden_data_path / rec_file).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / rec_file
        )
        receptor = Complex(
            chains, atoms, structure_file_name=(self.golden_data_path / rec_file)
        )
        io = IOFactory(self.golden_data_path / lig_file).get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / lig_file
        )
        ligand = Complex(
            chains, atoms, structure_file_name=(self.golden_data_path / lig_file)
        )
        adapter = DDNAAdapter(receptor, ligand)
        assert 6.915295143021656 == pytest.approx(
            self.dna(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )
