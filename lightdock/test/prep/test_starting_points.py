"""Tests for starting_points module"""

import pytest
from pathlib import Path
from lightdock.prep.starting_points import points_on_sphere, calculate_surface_points
from lightdock.structure.complex import Complex
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.test.helpers.test_utils import TestUtils


class TestStartingPoints:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"

    def test_points_on_sphere(self):
        points = points_on_sphere(5)
        expected = [
            [0.5999999999999999, -0.8, 0.0],
            [-0.6758097397797129, -0.39999999999999997, 0.6190970809322855],
            [0.08742572471695988, 5.551115123125783e-17, -0.9961710408648278],
            [0.5576434272376701, 0.4000000000000002, 0.7273471028736039],
            [-0.590828091189257, 0.8, -0.10450917022758696],
        ]
        for correct, point in zip(expected, points):
            assert correct[0] == pytest.approx(point[0])
            assert correct[1] == pytest.approx(point[1])
            assert correct[2] == pytest.approx(point[2])

    @pytest.mark.parametrize("file, tmp_file", [
        ("100_points.pdb", "points.pdb"),
        ("100_points.cif", "points.cif")
    ])
    def test_create_file_from_points(self, file, tmp_file, tmp_path):
        points = points_on_sphere(100)
        file_name = tmp_path / tmp_file
        io = IOFactory(file_name).get_instance()
        io.create_file_from_points(file_name, points)

        assert TestUtils.compare_biological_content(
            self.golden_data_path / file, tmp_path / tmp_file
        )

    @pytest.mark.parametrize("lig_file, rec_file, file, tmp_file", [
        ("1PPE_lig.pdb", "1PPE_rec.pdb", "starting_points.pdb", "points.pdb"),
        ("1PPE_lig.cif", "1PPE_rec.cif", "starting_points.cif", "points.cif")
    ])
    def test_calculate_starting_points(self, lig_file, rec_file, file, tmp_file, tmp_path):
        # Receptor
        file_name = self.golden_data_path / rec_file
        io = IOFactory(file_name).get_instance()
        atoms, _, chains = io.parse_complex_from_file(file_name)
        receptor = Complex(chains, atoms, structure_file_name=file_name)

        # Ligand
        file_name = self.golden_data_path / lig_file
        io = IOFactory(file_name).get_instance()
        atoms, _, chains = io.parse_complex_from_file(file_name)
        ligand = Complex(chains, atoms, structure_file_name=file_name)

        starting_points, rec_diameter, lig_diameter = calculate_surface_points(
            receptor, ligand, 100, [0.0, 0.0, 0.0], 50.0, [], [],
        )

        assert 50.213210831413676 == pytest.approx(rec_diameter)
        assert 27.855559534857672 == pytest.approx(lig_diameter)

        file_name = tmp_path / tmp_file
        io = IOFactory(file_name).get_instance()
        io.create_file_from_points(tmp_path / tmp_file, starting_points)

        assert TestUtils.compare_biological_content(
            self.golden_data_path / file, tmp_path / tmp_file
        )
