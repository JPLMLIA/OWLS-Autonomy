"""
Standardize experiment names for S3 upload

Standardizes experiment names into:

YYYY_MM_DD_HH_MM_SS

as a UID for S3 and Labelbox serving. It assumes an existing schema for
existing directory names, so checks for user confirmation for each rename.
"""
import sys
import os
import os.path as op
import csv
from glob import glob
from pathlib import Path
import argparse
import re


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_dir',              help="Directory of experiments to standardize")

    parser.add_argument('--force',                  action="store_true",
                                                    help="Don't require use confirmation for each rename")

    args = parser.parse_args()

    if not op.isdir(args.dataset_dir):
        print(f"{args.dataset_dir} is not a directory.")
        sys.exit(1)
    
    exp_dirs = glob(op.join(args.dataset_dir, "*", ""))

    # Verify experiment list

    print("Experiments found:")
    for exp in exp_dirs:
        print(f"> {Path(exp).name}")
    
    cont = input("Proceed? (y/n): ")
    if cont != "y":
        print("Aborting.")
        sys.exit(1)
    
    renaming_record = []
    for exp_path in exp_dirs:
        exp_name = Path(exp_path).name
        exp_path = op.join(args.dataset_dir, exp_name)

        exp_movie = op.join(exp_path, f"{exp_name}_base_movie.mp4")
        exp_mhi = op.join(exp_path, f"{exp_name}_mhi_labeled.png")

        exp_movie_standard = op.join(exp_path, f"{exp_name}_movie.mp4")
        exp_mhi_standard = op.join(exp_path, f"{exp_name}_mhi.png")

        if not op.exists(exp_movie):
            print(f"{Path(exp_movie).name} doesn't exist, continuing.")
            continue

        if not op.exists(exp_mhi):
            print(f"{Path(exp_mhi).name} doesn't exist, continuing.")
            continue

        print("=== RENAMING FORMULAE ===")
        print("Movie Rename:")
        print(f"> {exp_movie}")
        print(f"< {exp_movie_standard}")
        print("MHI Rename:")
        print(f"> {exp_mhi}")
        print(f"< {exp_mhi_standard}")

        cont = input("Proceed? (y/n)")
        if cont != "y":
            print("Aborting.")
            sys.exit(1)

        os.rename(exp_movie, exp_movie_standard)
        os.rename(exp_mhi, exp_mhi_standard)

