"""Tests for MMCIFReader module"""

from pathlib import Path
from lightdock.ioutil.IO import IO
from lightdock.ioutil.MMCIFIO import MMCIFIO
from lightdock.structure.atom import Atom, HetAtom
from lightdock.structure.residue import Residue
from lightdock.structure.chain import Chain
from lightdock.structure.complex import Complex
from lightdock.prep.starting_points import points_on_sphere
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Residue import Residue as BioResidue
from Bio.PDB.Atom import Atom as BioAtom
from Bio.PDB.Structure import Structure as BioStructure


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
        atom1 = MockBioPythonAtom(998, "C", " ", [4.0, 5.0, 6.0], 0.8, 30.0, "C")
        atom2 = MockBioPythonAtom(999, "O", " ", [4.0, 5.0, 6.0], 0.8, 30.0, "O")
        atoms = [atom1, atom2]

        result = MMCIFIO._build_residue(residue, atoms)

        assert isinstance(result, Residue)
        assert result.name == residue.resname
        assert result.number == residue.id[1]
        assert result.insertion == residue.id[2]
        assert len(result.atoms) == len(atoms)

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

    def test_build_BioAtom(self):
        atom = Atom(
            1,
            IO.cstrip("CA"),
            IO.cstrip(" "),
            IO.cstrip("A"),
            IO.cstrip("ARG"),
            1,
            IO.cstrip(" "),
            1.0,
            2.0,
            3.0,
            1.0,
            20.0,
            IO.cstrip("C")
        )

        atom.index = 0

        result = MMCIFIO._build_BioAtom(atom, [[atom.x, atom.y, atom.z]])

        assert isinstance(result, BioAtom)
        assert result.serial_number == atom.number
        assert result.name == atom.name
        assert result.altloc == atom.alternative.ljust(1)
        assert list(result.coord) == [atom.x, atom.y, atom.z]
        assert result.occupancy == atom.occupancy
        assert result.bfactor == atom.b_factor
        assert result.element == atom.element

    def test_build_BioResidue(self):
        atom = Atom(
            1,
            IO.cstrip("CA"),
            IO.cstrip(" "),
            IO.cstrip("A"),
            IO.cstrip("ARG"),
            1,
            IO.cstrip(" "),
            1.0,
            2.0,
            3.0,
            1.0,
            20.0,
            IO.cstrip("C")
        )
        atom.index = 0

        bioAtom = MMCIFIO._build_BioAtom(atom, [[atom.x, atom.y, atom.z]])

        residue = Residue(
            IO.cstrip("ARG"),
            1,
            IO.cstrip(" ")
        )

        result = MMCIFIO._build_BioResidue(resnumber=1, resname="ARG", insertion_code=" ", atom=bioAtom)

        assert isinstance(result, BioResidue)
        assert result.resname == residue.name
        assert result.id == (" ", residue.number, residue.insertion.ljust(1))
        assert len(result.child_dict) == 1

    def test_build_BioChain(self):
        atom = Atom(
            1,
            IO.cstrip("CA"),
            IO.cstrip(" "),
            IO.cstrip("A"),
            IO.cstrip("ARG"),
            1,
            IO.cstrip(" "),
            1.0,
            2.0,
            3.0,
            1.0,
            20.0,
            IO.cstrip("C")
        )
        atom.index = 0

        bioAtom = MMCIFIO._build_BioAtom(atom, [[atom.x, atom.y, atom.z]])

        residue = MMCIFIO._build_BioResidue(resnumber=1, resname="ARG", insertion_code=" ", atom=bioAtom)

        chain_id = "A"

        result = MMCIFIO._build_BioChain(chain_id, residue)

        assert isinstance(result, BioChain)
        assert result.id == chain_id
        assert len(result.child_dict) == 1

    def test_build_BioStructure(self):
        atom = Atom(
            1,
            IO.cstrip("CA"),
            IO.cstrip(" "),
            IO.cstrip("A"),
            IO.cstrip("ARG"),
            1,
            IO.cstrip(" "),
            1.0,
            2.0,
            3.0,
            1.0,
            20.0,
            IO.cstrip("C")
        )
        atom.index = 0

        bioAtom = MMCIFIO._build_BioAtom(atom, [[atom.x, atom.y, atom.z]])

        residue = MMCIFIO._build_BioResidue(resnumber=1, resname="ARG", insertion_code=" ", atom=bioAtom)

        chain_id = "A"

        chain = MMCIFIO._build_BioChain(chain_id, residue)

        structure_id = "TestStructure"

        result = MMCIFIO._build_BioStructure(structure_id, [chain])

        assert isinstance(result, BioStructure)
        assert result.id == structure_id
        assert len(result.child_dict) == 1

    def test_write_to_file(self):
        mmcif_io = MMCIFIO()
        test_file = self.absolute_path / "parse_complex_from_file_1CRN.cif"

        atoms, residues, chains = mmcif_io.parse_complex_from_file(test_file)

        lightdock_structures = [
            {
                "atoms": atoms,
                "residues": residues,
                "chains": chains,
                "file_name": test_file,
            }
        ]
        receptor = Complex.from_structures(lightdock_structures)
        output_file = self.absolute_path / "write_to_file_written.cif"

        mmcif_io.write_to_file(receptor, str(output_file))

        atoms_output, residues_output, chains_output = mmcif_io.parse_complex_from_file(test_file)

        assert output_file.exists()
        assert len(atoms) == len(atoms_output)
        assert len(residues) == len(residues_output)
        assert len(chains) == len(chains_output)

    def test_write_to_file_information(self):
        mmcif_io = MMCIFIO()
        test_file = self.absolute_path / "parse_complex_from_file_1CRN.cif"

        atoms, residues, chains = mmcif_io.parse_complex_from_file(test_file)

        lightdock_structures = [
            {
                "atoms": atoms,
                "residues": residues,
                "chains": chains,
                "file_name": test_file,
            }
        ]
        receptor = Complex.from_structures(lightdock_structures)
        output_file = self.absolute_path / "write_to_file_written.cif"

        mmcif_io.write_to_file(receptor, str(output_file))

        atoms_output, residues_output, chains_output = mmcif_io.parse_complex_from_file(test_file)

        for atom, atom_out in zip(atoms, atoms_output):
            assert atom.number == atom_out.number
            assert atom.name == atom_out.name
            assert atom.alternative == atom_out.alternative
            assert atom.chain_id == atom_out.chain_id
            assert atom.residue_name == atom_out.residue_name
            assert atom.residue_number == atom_out.residue_number
            assert atom.residue_insertion == atom_out.residue_insertion
            assert atom.x == atom_out.x
            assert atom.y == atom_out.y
            assert atom.z == atom_out.z
            assert atom.occupancy == atom_out.occupancy
            assert atom.b_factor == atom_out.b_factor
            assert atom.element == atom_out.element

        for residue, residue_out in zip(residues, residues_output):
            assert residue.name == residue_out.name
            assert residue.number == residue_out.number
            assert residue.insertion == residue_out.insertion

        for chain, chain_out in zip(chains, chains_output):
            assert chain.cid == chain_out.cid

    def test_create_file_from_points(self):
        mmcif_io = MMCIFIO()
        output_file = self.absolute_path / "points.cif"
        points = points_on_sphere(100)
        mmcif_io.create_file_from_points(str(output_file), points)

        assert output_file.exists()
