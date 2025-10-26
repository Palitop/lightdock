"""Tests for MMCIFReader module"""

from pathlib import Path
from lightdock.ioutil.IO import IO
from lightdock.ioutil.MMCIFIO import MMCIFIO
from lightdock.structure.atom import Atom, HetAtom
from lightdock.structure.residue import Residue
from lightdock.structure.chain import Chain


class MockBioPythonAtom:
    def __init__(self, serial_number, name, altloc, coord, occupancy, bfactor, element):
        self.serial_number = serial_number
        self.name = name
        self.altloc = altloc
        self.coord = coord
        self.occupancy = occupancy
        self.bfactor = bfactor
        self.element = element


class MockBioPythonResidue:
    def __init__(self, resname, residue_id):
        self.resname = resname
        self.id = residue_id


class MockBioPythonChain:
    def __init__(self, chain_id):
        self.id = chain_id


class TestMMCIFReader:
    def setup_class(self):
        self.absolute_path = Path(__file__).absolute().parent / "golden_data"

    def test_build_atom(self):
        chain = MockBioPythonChain("A")
        residue = MockBioPythonResidue("ARG", ("", 1, " "))
        atom = MockBioPythonAtom(1, "CA", " ", [1.0, 2.0, 3.0], 1.0, 20.0, "C")

        result = MMCIFIO._build_atom(chain, residue, atom)

        assert isinstance(result, Atom)
        assert result.number == atom.serial_number
        assert result.name == IO.cstrip(atom.name)
        assert result.alternative == IO.cstrip(atom.altloc)
        assert result.chain_id == IO.cstrip(chain.id)
        assert result.residue_name == IO.cstrip(residue.resname)
        assert result.residue_number == residue.id[1]
        assert result.residue_insertion == IO.cstrip(residue.id[2])
        assert result.x == atom.coord[0]
        assert result.y == atom.coord[1]
        assert result.z == atom.coord[2]
        assert result.occupancy == atom.occupancy
        assert result.b_factor == atom.bfactor
        assert result.element == IO.cstrip(atom.element)

    def test_build_hetatom(self):
        chain = MockBioPythonChain("B")
        residue = MockBioPythonResidue("HOH", ("H_", 99, "A"))
        atom = MockBioPythonAtom(999, "O", " ", [4.0, 5.0, 6.0], 0.8, 30.0, "O")

        result = MMCIFIO._build_hetatom(chain, residue, atom)

        assert isinstance(result, HetAtom)
        assert result.number == atom.serial_number
        assert result.name == IO.cstrip(atom.name)
        assert result.alternative == IO.cstrip(atom.altloc)
        assert result.chain_id == IO.cstrip(chain.id)
        assert result.residue_name == IO.cstrip(residue.resname)
        assert result.residue_number == residue.id[1]
        assert result.residue_insertion == IO.cstrip(residue.id[2])
        assert result.x == atom.coord[0]
        assert result.y == atom.coord[1]
        assert result.z == atom.coord[2]
        assert result.occupancy == atom.occupancy
        assert result.b_factor == atom.bfactor
        assert result.element == IO.cstrip(atom.element)

    def test_build_residue(self):
        residue = MockBioPythonResidue("GLY", ["", 42, "B"])

        result = MMCIFIO._build_residue(residue)

        assert isinstance(result, Residue)
        assert result.name == residue.resname
        assert result.number == residue.id[1]
        assert result.insertion == residue.id[2]

    def test_build_chain(self):
        residue1 = Residue("ALA", 1, " ")
        residue2 = Residue("LEU", 2, " ")
        residues = [residue1, residue2]

        chain_id = "C"

        result = MMCIFIO._build_chain(chain_id, residues)

        assert isinstance(result, Chain)
        assert result.cid == chain_id
        assert len(result.residues) == len(residues)
        assert result.residues[0].name == residue1.name
        assert result.residues[1].name == residue2.name

    def test_parse_complex_from_file_without_ignoring(self):
        mmcif_io = MMCIFIO()
        test_file = self.absolute_path / "parse_complex_from_file_1CRN.cif"

        atoms, residues, chains = mmcif_io.parse_complex_from_file(test_file)

        assert all(isinstance(atom, (Atom, HetAtom)) for atom in atoms)
        assert all(isinstance(residue, Residue) for residue in residues)
        assert all(isinstance(chain, Chain) for chain in chains)

        assert len(atoms) == 327
        assert len(residues) == 46
        assert len(chains) == 1

    def test_parse_complex_from_file_ignoring_residue(self):
        residues_to_ignore = ["ALA", "TYR"]

        mmcif_io = MMCIFIO()
        test_file = self.absolute_path / "parse_complex_from_file_1CRN.cif"

        atoms, residues, chains = mmcif_io.parse_complex_from_file(test_file, residues_to_ignore=residues_to_ignore)

        assert all(isinstance(atom, (Atom, HetAtom)) for atom in atoms)
        assert all(isinstance(residue, Residue) for residue in residues)
        assert all(isinstance(chain, Chain) for chain in chains)

        assert len(atoms) == 278
        assert len(residues) == 39
        assert len(chains) == 1

    def test_parse_complex_from_file_ignoring_atoms(self):
        atoms_to_ignore = ["C", "O", "CB"]

        mmcif_io = MMCIFIO()
        test_file = self.absolute_path / "parse_complex_from_file_1CRN.cif"

        atoms, residues, chains = mmcif_io.parse_complex_from_file(test_file, atoms_to_ignore=atoms_to_ignore)

        assert all(isinstance(atom, (Atom, HetAtom)) for atom in atoms)
        assert all(isinstance(residue, Residue) for residue in residues)
        assert all(isinstance(chain, Chain) for chain in chains)

        assert len(atoms) == 193
        assert len(residues) == 46
        assert len(chains) == 1
