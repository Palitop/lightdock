"""Tests for CPyDockDNA scoring function module"""

import pytest
from pathlib import Path
from lightdock.scoring.dna.driver import DNA, DNAAdapter
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex


class TestPyDockDNA:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("3mfk_dna.pdb", "3mfk_homodimer.pdb"),
        ("3mfk_dna.cif", "3mfk_homodimer.cif")
    ])
    def test_calculate_DNA_3MFK(self, lig_file, rec_file):
        dna = DNA()
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
            chains, atoms, structure_file_name=(self.golden_data_path / lig_file)
        )
        adapter = DNAAdapter(receptor, ligand)
        assert -2716.68018700585 == pytest.approx(
            dna(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("3mfk_dna.pdb", "3mfk_homodimer_with_H.pdb"),
        ("3mfk_dna.cif", "3mfk_homodimer_with_H.cif")
    ])
    def test_calculate_DNA_3MFK_with_hydrogens(self, lig_file, rec_file):
        dna = DNA()
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
        atoms, _t, chains = io.parse_complex_from_file(
            self.golden_data_path / lig_file
        )
        ligand = Complex(
            chains, atoms, structure_file_name=(self.golden_data_path / lig_file)
        )
        adapter = DNAAdapter(receptor, ligand)
        assert 688.1703668834168 == pytest.approx(
            dna(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )
