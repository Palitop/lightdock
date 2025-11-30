"""Tests for Ellipsoid module"""

import pytest
import numpy as np
from pathlib import Path
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex
from lightdock.mathutil.ellipsoid import MinimumVolumeEllipsoid
from lightdock.error.lightdock_errors import MinimumVolumeEllipsoidError
from lightdock.mathutil.constants import ERROR_TOLERANCE


class TestEllipsoid:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"

    def test_calculate_min_volume_ellipsoid(self):
        io = IOFactory(self.golden_data_path / "1PPE_l_u.pdb").get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / "1PPE_l_u.pdb"
        )
        protein = Complex(chains, atoms)

        ellipsoid = MinimumVolumeEllipsoid(protein.atom_coordinates[0].coordinates)

        assert 5.79979144 == pytest.approx(ellipsoid.center[0])
        assert 13.30609275 == pytest.approx(ellipsoid.center[1])
        assert 6.28378695 == pytest.approx(ellipsoid.center[2])

        assert 11.51000999 == pytest.approx(ellipsoid.radii[0])
        assert 17.41300089 == pytest.approx(ellipsoid.radii[1])
        assert 25.1317681 == pytest.approx(ellipsoid.radii[2])

        assert -0.64868458 == pytest.approx(ellipsoid.rotation[0][0])
        assert -0.43420895 == pytest.approx(ellipsoid.rotation[0][1])
        assert 0.62503673 == pytest.approx(ellipsoid.rotation[0][2])
        assert 0.75208829 == pytest.approx(ellipsoid.rotation[1][0])
        assert -0.49144928 == pytest.approx(ellipsoid.rotation[1][1])
        assert 0.43913643 == pytest.approx(ellipsoid.rotation[1][2])
        assert 0.11649688 == pytest.approx(ellipsoid.rotation[2][0])
        assert 0.75494384 == pytest.approx(ellipsoid.rotation[2][1])
        assert 0.64535903 == pytest.approx(ellipsoid.rotation[2][2])

        expected_poles = [
            [13.266157381855532, 18.303842059830465, -0.91039204503235993],
            [-1.6665744987943629, 8.3083434397316651, 13.477965942387469],
            [-7.296322648355452, 21.863699498711949, -1.3628961564119457],
            [18.89590553141662, 4.7484860008501819, 13.930470053767054],
            [2.8720188105117521, -5.6669806736857815, -9.9352265853089641],
            [8.7275640725494164, 32.279166173247916, 22.502800482664075],
        ]

        assert np.allclose(expected_poles, ellipsoid.poles, ERROR_TOLERANCE)

    def test_calculate_min_volume_ellipsoid_cif(self):
        io = IOFactory(self.golden_data_path / "1PPE_l_u.cif").get_instance()
        atoms, _, chains = io.parse_complex_from_file(
            self.golden_data_path / "1PPE_l_u.cif"
        )
        protein = Complex(chains, atoms)

        ellipsoid = MinimumVolumeEllipsoid(protein.atom_coordinates[0].coordinates)

        print(ellipsoid.poles)

        assert 5.799791250856532 == pytest.approx(ellipsoid.center[0])
        assert 13.306092685541458 == pytest.approx(ellipsoid.center[1])
        assert 6.283786663549531 == pytest.approx(ellipsoid.center[2])

        assert 11.51000997975971 == pytest.approx(ellipsoid.radii[0])
        assert 17.41300065237483 == pytest.approx(ellipsoid.radii[1])
        assert 25.131767919940263 == pytest.approx(ellipsoid.radii[2])

        assert -0.6486846041655251 == pytest.approx(ellipsoid.rotation[0][0])
        assert -0.43420890854202066 == pytest.approx(ellipsoid.rotation[0][1])
        assert 0.625036725370088 == pytest.approx(ellipsoid.rotation[0][2])
        assert 0.7520882640674591 == pytest.approx(ellipsoid.rotation[1][0])
        assert -0.49144929949240357 == pytest.approx(ellipsoid.rotation[1][1])
        assert 0.43913645838215387 == pytest.approx(ellipsoid.rotation[1][2])
        assert 0.11649689854503213 == pytest.approx(ellipsoid.rotation[2][0])
        assert 0.7549438454422769 == pytest.approx(ellipsoid.rotation[2][1])
        assert 0.6453590185766491 == pytest.approx(ellipsoid.rotation[2][2])

        expected_poles = [
            [13.266157518518202, 18.303841556160688, -0.9103922831765114],
            [-1.6665750168051394, 8.308343814922228, 13.477965610275573],
            [-7.296322181993587, 21.863699658211836, -1.3628967727404868],
            [18.89590468370665, 4.748485712871082, 13.93047009983955],
            [2.8720182332299578, -5.666980830701096, -9.935226416359232],
            [8.727564268483107, 32.27916620178401, 22.502799743458294]
        ]

        assert np.allclose(expected_poles, ellipsoid.poles, ERROR_TOLERANCE)

    def test_exception_singular_matrix(self):
        with pytest.raises(MinimumVolumeEllipsoidError):
            coordinates = np.array([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]])

            ellipsoid = MinimumVolumeEllipsoid(coordinates)

            assert len(ellipsoid.poles) > 0
