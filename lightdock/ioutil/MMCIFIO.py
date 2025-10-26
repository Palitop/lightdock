"""Parses Atomic coordinates entries from MMCIF files"""

from os import path
from lightdock.ioutil.IO import IO
from Bio.PDB import MMCIFParser
from lightdock.structure.atom import Atom, HetAtom
from lightdock.structure.residue import Residue
from lightdock.structure.chain import Chain
from lightdock.util.logger import LoggingManager


class MMCIFIO(IO):

    def __init__(self):
        self.log = LoggingManager.get_logger("mmcif")
        pass

    @staticmethod
    def _build_atom(chain, residue, atom) -> Atom:
        return Atom(
            atom.serial_number,
            IO.cstrip(atom.name),
            IO.cstrip(atom.altloc),
            IO.cstrip(chain.id),
            IO.cstrip(residue.resname),
            residue.id[1],
            IO.cstrip(residue.id[2]),
            atom.coord[0],
            atom.coord[1],
            atom.coord[2],
            atom.occupancy,
            atom.bfactor,
            IO.cstrip(atom.element)
        )

    @staticmethod
    def _build_hetatom(chain, residue, atom) -> HetAtom:
        return HetAtom(
            atom.serial_number,
            IO.cstrip(atom.name),
            IO.cstrip(atom.altloc),
            IO.cstrip(chain.id),
            IO.cstrip(residue.resname),
            residue.id[1],
            IO.cstrip(residue.id[2]),
            atom.coord[0],
            atom.coord[1],
            atom.coord[2],
            atom.occupancy,
            atom.bfactor,
            IO.cstrip(atom.element)
        )

    @staticmethod
    def _build_residue(residue) -> Residue:
        return Residue(residue.resname, residue.id[1], residue.id[2])

    @staticmethod
    def _build_chain(id, residues) -> Chain:
        return Chain(id, residues)

    def parse_complex_from_file(
        self,
        input_file_name: str,
        atoms_to_ignore: list = [],
        residues_to_ignore: list = [],
        verbose: bool = False
    ):
        parser = MMCIFParser(QUIET=True)
        structure_id = path.splitext(path.basename(input_file_name))[0]
        structure = parser.get_structure(structure_id, input_file_name)

        atoms = []
        residues = []
        chains = []

        if len(structure) > 1:
            self.log.warning(
                "Multiple models found in %s. Only first model will be used."
                % self.filename
            )

        for chain in structure[0]:
            chain_residues = []
            for residue in chain:
                if residue.resname in residues_to_ignore:
                    if verbose:
                        for atom in residue:
                            print(f"Ignored atom {chain.id}.{residue.resname}.{residue.id[1]} {atom.name}")
                    continue

                for atom in residue:
                    if atom.name in atoms_to_ignore:
                        if verbose:
                            print(f"Ignored atom {chain.id}.{residue.resname}.{residue.id[1]} {atom.name}")
                        continue
                    atomObject = MMCIFIO._build_atom(chain, residue, atom) if not residue.id[0].strip() else MMCIFIO._build_hetatom(chain, residue, atom)
                    atoms.append(atomObject)

                residueObject = MMCIFIO._build_residue(residue)
                chain_residues.append(residueObject)
                residues.append(residueObject)
            chains.append(MMCIFIO._build_chain(chain.id, chain_residues))

        # Set backbone and side-chain atoms
        for residue in residues:
            residue.set_backbone_and_sidechain()
            try:
                residue.check()
            except Exception as e:
                self.log.warning("Possible problem: %s" % str(e))

        return atoms, residues, chains

    def write_to_file():
        pass

    def create_file_from_points():
        pass
