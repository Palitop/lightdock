#!/usr/bin/env python3

"""Calculates the diameter of a given PDB/MMCIF structure"""

import argparse
from scipy import spatial
import numpy as np
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.structure.complex import Complex
from lightdock.util.logger import LoggingManager


log = LoggingManager.get_logger("lgd_calculate_diameter")


def parse_command_line():
    parser = argparse.ArgumentParser(prog="lgd_calculate_diameter")
    parser.add_argument(
        "file", help="file for structure to calculate maximum diameter"
    )
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    args = parse_command_line()

    io = IOFactory(args.file).get_instance()
    atoms, residues, chains = io.parse_complex_from_file(args.file)

    structure = Complex(chains, atoms, structure_file_name=args.file)
    distances_matrix = spatial.distance.squareform(
        spatial.distance.pdist(structure.representative())
    )
    ligand_max_diameter = np.max(distances_matrix)

    print(ligand_max_diameter)
