"""Tests for MJ3h potentials module"""

import pytest
from pathlib import Path
from lightdock.scoring.mj3h.driver import MJPotential, MJ3h, MJ3hAdapter
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex


class TestMJPotential:
    def test_create_MJPotential_interface(self):
        mj = MJPotential()

        assert len(mj.potentials) == 4

    def test_create_MJ3h(self):
        mj3h = MJ3h()

        assert len(mj3h.potentials) == 20

        assert -0.84 == pytest.approx(mj3h.potentials[0][0])
        assert 0.05 == pytest.approx(mj3h.potentials[15][3])
        assert 0.76 == pytest.approx(mj3h.potentials[19][19])


class TestMJ3hAdapter:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1PPElig.pdb", "1PPErec.pdb"),
        ("1PPElig.cif", "1PPErec.cif")
    ])
    def test_create_adapter(self, lig_file, rec_file):
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
        adapter = MJ3hAdapter(receptor, ligand)

        assert adapter.receptor_model.coordinates[0].coordinates.shape == (223, 3)
        assert adapter.ligand_model.coordinates[0].coordinates.shape == (29, 3)
        assert len(adapter.receptor_model.objects) == 223
        assert len(adapter.ligand_model.objects) == 29


class TestMJ3h:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"
        self.mj3h = MJ3h()

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1PPElig.pdb", "1PPErec.pdb"),
        ("1PPElig.cif", "1PPErec.cif")
    ])
    def test_calculate_MJ3h_1PPE(self, lig_file, rec_file):
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
        adapter = MJ3hAdapter(receptor, ligand)
        assert 2.02 == pytest.approx(
            self.mj3h(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )
        # Original potential without min cutoff
        # assert_almost_equal(-17.94/2, self.mj3h(receptor, receptor.residue_coordinates, ligand, ligand.residue_coordinates))

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1EAWlig.pdb", "1EAWrec.pdb"),
        ("1EAWlig.cif", "1EAWrec.cif")
    ])
    def test_calculate_MJ3h_1EAW(self, lig_file, rec_file):
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
        adapter = MJ3hAdapter(receptor, ligand)
        assert -1.55 == pytest.approx(
            self.mj3h(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )
        # Original potential without min cutoff
        # assert_almost_equal(-3.14/2, self.mj3h(receptor, receptor.residue_coordinates, ligand, ligand.residue_coordinates))

    @pytest.mark.parametrize("lig_file, rec_file", [
        ("1AY7lig.pdb", "1AY7rec.pdb"),
        ("1AY7lig.cif", "1AY7rec.cif")
    ])
    def test_calculate_MJ3h_1AY7(self, lig_file, rec_file):
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
        adapter = MJ3hAdapter(receptor, ligand)
        assert -12.92 == pytest.approx(
            self.mj3h(
                adapter.receptor_model,
                adapter.receptor_model.coordinates[0],
                adapter.ligand_model,
                adapter.ligand_model.coordinates[0],
            )
        )
        # Original potential without min cutoff
        # assert_almost_equal(9.06/2, self.mj3h(receptor, receptor.residue_coordinates, ligand, ligand.residue_coordinates))
