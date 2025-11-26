"""Parses Atomic coordinates entries from MMCIF files"""

import numpy as np
from pathlib import Path
from lightdock.ioutil.IO import IO
from Bio.PDB import MMCIFParser, MMCIFIO as BioMMCIFIO
from Bio.PDB.Atom import Atom as BioAtom
from Bio.PDB.Residue import Residue as BioResidue
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Model import Model as BioModel
from Bio.PDB.Structure import Structure as BioStructure
from lightdock.structure.atom import Atom, HetAtom
from lightdock.structure.residue import Residue
from lightdock.structure.chain import Chain
from lightdock.structure.complex import Complex
from lightdock.structure.space import SpacePoints
from lightdock.util.logger import LoggingManager


class MMCIFIO(IO):

    def __init__(self):
        self.log = LoggingManager.get_logger("mmcif")

    @staticmethod
    def _build_atom(
        chain: BioChain,
        residue: BioResidue,
        atom: BioAtom
    ) -> Atom:
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
    def _build_hetatom(
        chain: BioChain,
        residue: BioResidue,
        atom: BioAtom
    ) -> HetAtom:
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
    def _build_residue(
        residue: BioResidue,
        atoms: list[Atom]
    ) -> Residue:
        return Residue(residue.resname, residue.id[1], residue.id[2], atoms=atoms)

    @staticmethod
    def _build_chain(
        id: str,
        residues: list
    ) -> Chain:
        return Chain(id, residues)

    def parse_complex_from_file(
        self,
        input_file_name: Path,
        atoms_to_ignore: list = [],
        residues_to_ignore: list = [],
        verbose: bool = False
    ):
        parser = MMCIFParser(QUIET=True)
        structure_id = input_file_name.stem
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

                atoms_per_residue = []

                for atom in residue:
                    if atom.name in atoms_to_ignore:
                        if verbose:
                            print(f"Ignored atom {chain.id}.{residue.resname}.{residue.id[1]} {atom.name}")
                        continue
                    atomObject = MMCIFIO._build_atom(chain, residue, atom) if not residue.id[0].strip() else MMCIFIO._build_hetatom(chain, residue, atom)
                    atoms.append(atomObject)
                    atoms_per_residue.append(atomObject)

                residueObject = MMCIFIO._build_residue(residue, atoms_per_residue)
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

    @staticmethod
    def _build_BioStructure(
        output_file_name: Path,
        chains: list,
    ) -> BioStructure:
        model: BioModel = BioModel(0)
        for chain in chains:
            model.add(chain)
        structure_id = output_file_name.stem
        structure: BioStructure = BioStructure(structure_id)
        structure.add(model)
        return structure

    @staticmethod
    def _build_BioChain(
        id: str,
        residue: BioResidue,
    ) -> BioChain:
        chain = BioChain(id)
        chain.add(residue)
        return chain

    @staticmethod
    def _build_BioResidue(
        resnumber: str,
        resname: str,
        insertion_code: str,
        atom: BioAtom,
    ) -> BioResidue:
        residue = BioResidue((" ", resnumber, insertion_code.ljust(1)), resname, " ")
        residue.add(atom)
        return residue

    @staticmethod
    def _build_BioAtom(atom: Atom, coordinates: list) -> BioAtom:
        coords = np.array([coordinates[atom.index][0], coordinates[atom.index][1], coordinates[atom.index][2]], dtype=float)
        return BioAtom(
            serial_number=int(atom.number),
            name=atom.name,
            fullname=atom.name,
            altloc=atom.alternative.ljust(1),
            coord=np.round(coords, 3),
            bfactor=atom.b_factor,
            occupancy=np.round(atom.occupancy, 1),
            element=atom.element
        )

    def write_to_file(
        self,
        molecule: Complex,
        output_file_name: Path,
        atom_coordinates: SpacePoints = None,
        structure_id: int = 0
    ):
        if atom_coordinates is None:
            atom_coordinates = molecule.atom_coordinates[structure_id]

        chains: list[BioChain] = []

        for atom in molecule.atoms:
            atomObject: BioAtom = MMCIFIO._build_BioAtom(atom, atom_coordinates)
            chain: BioChain = next((c for c in chains if c.id == atom.chain_id), None)
            if chain is None:
                residue = MMCIFIO._build_BioResidue(atom.residue_number, atom.residue_name, atom.residue_insertion, atomObject)
                chain: BioChain = MMCIFIO._build_BioChain(atom.chain_id, residue)
                chains.append(chain)
            else:
                residue = next((r for r in chain if r.id[0] == " " and r.id[1] == atom.residue_number and r.id[2] == atom.residue_insertion.ljust(1)), None)
                if residue is None:
                    residue: BioResidue = MMCIFIO._build_BioResidue(atom.residue_number, atom.residue_name, atom.residue_insertion, atomObject)
                    chain.add(residue)
                else:
                    residue.add(atomObject)

        try:
            # if file exists, we add the new chains to the structure
            parser = MMCIFParser(QUIET=True)
            structure_id = output_file_name.stem
            structure = parser.get_structure(structure_id, output_file_name)

            chain_ids = [chain.id for chain in structure.get_chains()]

            for chain in chains:
                if chain.id in chain_ids:
                    last_id = chain_ids[-1]
                    chain.id = chr(ord(last_id) + 1)
                    chain_ids.append(chain.id)
                structure[0].add(chain)
        except FileNotFoundError:
            # if the file does not exist, create a new file with the structure
            structure: BioStructure = MMCIFIO._build_BioStructure(output_file_name, chains)
        except Exception as e:
            print(e)
        finally:
            io = BioMMCIFIO()
            io.set_structure(structure)
            io.save(str(output_file_name))

    def create_file_from_points(
        self,
        file_name: Path,
        points: list,
        atom_name: str = "H",
        res_name: str = "SWR",
        chain_id: str = "Z",
        element: str = "H"
    ):
        chains: list[BioChain] = []
        for index, point in enumerate(points):
            atomObject: BioAtom = BioAtom(
                serial_number=index + 1,
                name=atom_name,
                fullname=atom_name,
                altloc=" ",
                coord=np.array([point[0], point[1], point[2]], dtype=float),
                bfactor=1.0,
                occupancy=1.0,
                element=element
            )
            chain: BioChain = next((c for c in chains if c.id == chain_id), None)
            if chain is None:
                residue: BioResidue = MMCIFIO._build_BioResidue(resnumber=str(index + 1), resname=res_name, insertion_code=" ", atom=atomObject)
                chain: BioChain = MMCIFIO._build_BioChain(id=chain_id, residue=residue)
                chains.append(chain)
            else:
                residue = next((r for r in chain if r.id[0] == " " and r.id[1] == str(index + 1) and r.id[2] == " "), None)
                if residue is None:
                    residue: BioResidue = MMCIFIO._build_BioResidue(resnumber=str(index + 1), resname=res_name, insertion_code=" ", atom=atomObject)
                    chain.add(residue)
                else:
                    residue.add(atomObject)

        structure: BioStructure = MMCIFIO._build_BioStructure(file_name, chains=chains)

        try:
            io = BioMMCIFIO()
            io.set_structure(structure)
            io.save(str(file_name))
        except Exception as e:
            print(e)
