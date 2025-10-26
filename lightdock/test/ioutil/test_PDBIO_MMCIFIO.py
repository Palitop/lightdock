""""Tests to compare PDBIO and MMCIFIO outputs"""

from pathlib import Path
from lightdock.ioutil.MMCIFIO import MMCIFIO
from lightdock.ioutil.PDBIO import parse_complex_from_file


class TestPDBIOMMCIFIOReader:
    def setup_class(self):
        self.absolute_path = Path(__file__).absolute().parent / "golden_data"
        self.pdb_file = self.absolute_path / "parse_complex_from_file_1CRN.pdb"
        self.mmcif_file = self.absolute_path / "parse_complex_from_file_1CRN.cif"

    def test_compare_pdb_mmcif_outputs(self):
        atoms_pbd, residues_pdb, chains_pdb = parse_complex_from_file(self.pdb_file)

        mmcif_io = MMCIFIO()
        atoms_mmcif, residues_mmcif, chains_mmcif = mmcif_io.parse_complex_from_file(self.mmcif_file)

        assert len(atoms_pbd) == len(atoms_mmcif)
        assert len(residues_pdb) == len(residues_mmcif)
        assert len(chains_pdb) == len(chains_mmcif)

    def test_compare_pdb_mmcif_outputs_ignoring_atoms(self):
        atoms_to_ignore = ["C", "O", "CB"]

        atoms_pbd, residues_pdb, chains_pdb = parse_complex_from_file(self.pdb_file, atoms_to_ignore=atoms_to_ignore)

        mmcif_io = MMCIFIO()
        atoms_mmcif, residues_mmcif, chains_mmcif = mmcif_io.parse_complex_from_file(self.mmcif_file, atoms_to_ignore=atoms_to_ignore)

        assert len(atoms_pbd) == len(atoms_mmcif)
        assert len(residues_pdb) == len(residues_mmcif)
        assert len(chains_pdb) == len(chains_mmcif)

    def test_compare_pdb_mmcif_outputs_ignoring_residues(self):
        residues_to_ignore = ["ALA", "TYR"]

        atoms_pbd, residues_pdb, chains_pdb = parse_complex_from_file(self.pdb_file, residues_to_ignore=residues_to_ignore)

        mmcif_io = MMCIFIO()
        atoms_mmcif, residues_mmcif, chains_mmcif = mmcif_io.parse_complex_from_file(self.mmcif_file, residues_to_ignore=residues_to_ignore)

        assert len(atoms_pbd) == len(atoms_mmcif)
        assert len(residues_pdb) == len(residues_mmcif)
        assert len(chains_pdb) == len(chains_mmcif)

    def test_compare_pdb_mmcif_information(self):
        atoms_pbd, residues_pdb, chains_pdb = parse_complex_from_file(self.pdb_file)

        mmcif_io = MMCIFIO()
        atoms_mmcif, residues_mmcif, chains_mmcif = mmcif_io.parse_complex_from_file(self.mmcif_file)

        for atom_pdb, atom_mmcif in zip(atoms_pbd, atoms_mmcif):
            assert atom_pdb.number == atom_mmcif.number
            assert atom_pdb.name == atom_mmcif.name
            assert atom_pdb.alternative == atom_mmcif.alternative
            assert atom_pdb.chain_id == atom_mmcif.chain_id
            assert atom_pdb.residue_name == atom_mmcif.residue_name
            assert atom_pdb.residue_number == atom_mmcif.residue_number
            assert atom_pdb.residue_insertion == atom_mmcif.residue_insertion
            assert atom_pdb.x == atom_mmcif.x
            assert atom_pdb.y == atom_mmcif.y
            assert atom_pdb.z == atom_mmcif.z
            assert atom_pdb.occupancy == atom_mmcif.occupancy
            assert atom_pdb.b_factor == atom_mmcif.b_factor
            assert atom_pdb.element == atom_mmcif.element

        for residue_pdb, residue_mmcif in zip(residues_pdb, residues_mmcif):
            assert residue_pdb.name == residue_mmcif.name
            assert residue_pdb.number == residue_mmcif.number
            assert residue_pdb.insertion == residue_mmcif.insertion

        for chain_pdb, chain_mmcif in zip(chains_pdb, chains_mmcif):
            assert chain_pdb.cid == chain_mmcif.cid
