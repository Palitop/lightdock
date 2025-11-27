#!/usr/bin/env python3

"""Creates a PDB/MMCIF file with atom points representing the position for each of the glowworms of a swarm"""

import argparse
import os
from lightdock.ioutil.IOFactory import IOFactory
from lightdock.util.logger import LoggingManager
from lightdock.util.parser import valid_file


log = LoggingManager.get_logger("generate_glowworm_positions")


def parse_output_file(lightdock_output):
    glowworm_translations = []

    data_file = open(lightdock_output)
    lines = data_file.readlines()
    data_file.close()

    counter = 0
    for line in lines:
        if line[0] == "(":
            counter += 1
            last = line.index(")")
            coord = line[1:last].split(",")
            glowworm_translations.append(
                [float(coord[0]), float(coord[1]), float(coord[2])]
            )
    log.info("Read %s coordinate lines" % counter)
    return glowworm_translations


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="generate_glowworm_positions")
    # Lightdock output file
    parser.add_argument(
        "lightdock_output",
        help="lightdock output file",
        type=valid_file,
        metavar="lightdock_output",
    )

    parser.add_argument(
        "--output_format",
        help="Output file format (pdb or mmcif)",
        metavar="output_format",
        default="pdb",
        choices=["pdb", "cif"]
    )

    args = parser.parse_args()

    # Output file
    translations = parse_output_file(args.lightdock_output)

    # Destination path is the same as the lightdock output
    destination_path = os.path.dirname(args.lightdock_output)
    file_name = os.path.splitext(args.lightdock_output)[0] + "." + args.output_format

    output_file = os.path.join(destination_path, file_name)
    io = IOFactory(output_file).get_instance()
    io.create_file_from_points(output_file, translations, res_name="GLW")
    log.info("%s file created." % os.path.join(destination_path, file_name))
