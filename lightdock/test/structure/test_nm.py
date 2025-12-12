"""Tests for ANM calculus related module"""

import pytest
import numpy as np
from pathlib import Path
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex
from lightdock.structure.nm import calculate_nmodes, write_nmodes, read_nmodes
from lightdock.constants import STARTING_NM_SEED, DEFAULT_ANM_RMSD


class TestNM:
    def setup_class(self):
        self.path = Path(__file__).absolute().parent
        self.golden_data_path = self.path / "golden_data"

    @pytest.mark.parametrize("file", ["2UUY_lig.pdb", "2UUY_lig.cif"])
    def test_calculate_anm_protein_1(self, file):
        file_name = self.golden_data_path / "nm_prot" / file
        io = IOFactory(file_name).get_instance()
        _, _, chains = io.parse_complex_from_file(file_name)
        molecule = Complex(chains)

        nmodes = calculate_nmodes(
            file_name,
            n_modes=10,
            rmsd=DEFAULT_ANM_RMSD,
            seed=STARTING_NM_SEED,
            molecule=molecule
        )

        expected_nmodes = read_nmodes(
            self.golden_data_path / "nm_prot" / "lightdock_lig.nm.npy"
        )

        assert np.allclose(expected_nmodes, nmodes)

    @pytest.mark.parametrize("file", ["2UUY_rec.pdb", "2UUY_rec.cif"])
    def test_calculate_anm_protein_2(self, file):
        file_name = self.golden_data_path / "nm_prot" / file
        io = IOFactory(file_name).get_instance()
        _, _, chains = io.parse_complex_from_file(file_name)
        molecule = Complex(chains)

        nmodes = calculate_nmodes(
            file_name,
            n_modes=10,
            rmsd=DEFAULT_ANM_RMSD,
            seed=STARTING_NM_SEED,
            molecule=molecule
        )

        expected_nmodes = read_nmodes(
            self.golden_data_path / "nm_prot" / "lightdock_rec.nm.npy"
        )

        assert np.allclose(expected_nmodes, nmodes)

    @pytest.mark.parametrize("file", ["1DIZ_lig.pdb", "1DIZ_lig.cif"])
    def test_calculate_anm_dna(self, file):
        file_name = self.golden_data_path / "nm_dna" / file
        io = IOFactory(file_name).get_instance()
        _, _, chains = io.parse_complex_from_file(file_name)
        molecule = Complex(chains)

        nmodes = calculate_nmodes(
            file_name,
            n_modes=10,
            rmsd=DEFAULT_ANM_RMSD,
            seed=STARTING_NM_SEED,
            molecule=molecule
        )

        expected_nmodes = read_nmodes(
            self.golden_data_path / "nm_dna" / "lightdock_lig.nm.npy"
        )

        assert np.allclose(expected_nmodes, nmodes)

    @pytest.mark.parametrize("file", ["1DIZ_lig.pdb", "1DIZ_lig.cif"])
    def test_read_write(self, file, tmp_path):
        file_name = self.golden_data_path / "nm_dna" / file
        io = IOFactory(file_name).get_instance()
        _, _, chains = io.parse_complex_from_file(file_name)
        molecule = Complex(chains)

        nmodes = calculate_nmodes(
            file_name,
            n_modes=10,
            rmsd=DEFAULT_ANM_RMSD,
            seed=STARTING_NM_SEED,
            molecule=molecule
        )
        write_nmodes(nmodes, tmp_path / "test_nm")

        expected_nmodes = read_nmodes(
            self.golden_data_path / "nm_dna" / "lightdock_lig.nm.npy"
        )
        other_nmodes = read_nmodes(tmp_path / "test_nm.npy")

        assert np.allclose(expected_nmodes, other_nmodes)
