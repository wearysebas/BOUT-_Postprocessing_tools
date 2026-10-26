"""
Add the INGRID 'topology' variable to an existing BOUT++ grid file, in place,
exactly as gridue_to_bout.py writes it. Nothing else in the file is touched.

Usage:
    python grid_topologizer.py <grid_file> <topology> [--force]
"""

import argparse
import os
import sys

from boututils.datafile import DataFile

ACCEPTED_TOPOLOGIES = [
    "CFL",
    "SN",
    "CDN",
    "UDN",
    "SF",
    "SF15",
    "SF45",
    "SF75",
    "SF105",
    "SF135",
    "SF165",
    "SF-IDEAL",
    "XPT",
]


def main():
    parser = argparse.ArgumentParser(
        description="Add the INGRID topology variable to an existing BOUT++ grid file."
    )
    parser.add_argument("grid_file", help="Path to the BOUT++ grid file (.nc)")
    parser.add_argument(
        "topology",
        type=str.upper,
        choices=ACCEPTED_TOPOLOGIES,
        help="Topology to write (case-insensitive)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing, different topology value",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.grid_file):
        sys.exit(f"Error: grid file not found: {args.grid_file}")

    with DataFile(args.grid_file, write=True) as f:
        if "topology" in f.list():
            existing = str(f.read("topology")).strip().upper()
            if existing == args.topology:
                print(f"{args.grid_file}: topology already '{existing}'. Nothing to do.")
                return
            if not args.force:
                sys.exit(
                    f"Error: {args.grid_file} already has topology '{existing}'. "
                    "Use --force to overwrite."
                )
            print(f"Overwriting topology '{existing}' -> '{args.topology}'")

        f.write("topology", args.topology)

    print(f"{args.grid_file}: topology = {args.topology}")


if __name__ == "__main__":
    main()
