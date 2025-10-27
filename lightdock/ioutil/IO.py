from abc import ABC, abstractmethod
from lightdock.structure.complex import Complex
from lightdock.structure.space import SpacePoints


class IO (ABC):

    @abstractmethod
    def parse_complex_from_file(
        atoms_to_ignore: list = [],
        residues_to_ignore: list = [],
        verbose: bool = False
    ):
        pass

    @abstractmethod
    def write_to_file(
        molecule: Complex,
        output_file_name: str,
        atom_coordinates: SpacePoints = None,
        structure_id: int = 0
    ):
        pass

    @abstractmethod
    def create_file_from_points(
        file_name: str,
        points: list,
        atom_name: str = "H",
        res_name: str = "SWR",
        chain_id: str = "Z",
        element: str = "H"
    ):
        pass

    def cstrip(string):
        """Remove unwanted symbols from string."""
        return string.strip(" \t\n\r")
