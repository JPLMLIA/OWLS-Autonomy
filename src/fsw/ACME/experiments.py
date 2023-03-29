# OWLS CE-MS Data Processor for raw ACME files
import logging
import os
import os.path as op
import pickle
import glob
import subprocess
import json
import csv
from pathlib import Path

import numpy as np

def read_csvs(init_csv):
    """ Read and combine multiple BaySpec CSV outputs from filepath of 0th file

    Parameters
    ----------
    init_csv: string
        Filepath to first CSV file in the sequence
    
    Returns
    -------
    data: dictionary
        Data with keys time_axis, mass_axis, and matrix
    """

    # Get parent directory of CSV
    parent_dir = Path(init_csv).parent
    spectra_csvs = sorted(glob.glob(op.join(parent_dir, "Spectra_*.csv")))

    mass_axis = []
    time_axis = []
    matrix = []

    for csv_i, csv_file in enumerate(spectra_csvs):
        logging.info(f"Loading file ({csv_i}/{len(spectra_csvs)}): {Path(csv_file).name} ")
        csv_data = []
        with open(csv_file) as f:
            reader = csv.reader(f)
            for row in reader:
                csv_data.append(row)
        
        # Find mass axis in expected location
        if csv_data[9][0] == "Spectrum_m/z:":
            if len(mass_axis) == 0:
                # New mass_axis, set
                mass_axis = [float(x) for x in csv_data[9][1:-1]]
            else:
                # Confirm that it's the same as previous axis
                curr_mass_axis = [float(x) for x in csv_data[9][1:-1]]
                if not mass_axis == curr_mass_axis:
                    logging.error(f"Mass axis not consistent between files: {csv_file}")
        else:
            logging.error(f"Could not find mass_axis, unexpected format: {csv_file}")
            return {}
        
        # Find time axis in expected location
        if csv_data[10][46] == " MS_Intensity_Array...":
            # Build time_axis and matrix
            for matrix_row in csv_data[11:]:
                time_axis.append(float(matrix_row[2]))
                matrix.append([float(x) for x in matrix_row[46:-1]])
        else:
            logging.error(f"Could not find matrix, unexpected format: {csv_file}")
            return {}

        # Cross-check matrix and mass_axis dimensions
        if len(mass_axis) != len(matrix[0]):
            logging.error(f"mass_axis does not match matrix ({len(mass_axis)} != {len(matrix[0])})")

    return {
        'mass_axis': np.array(mass_axis),
        'time_axis': np.array(time_axis) / 60,
        'matrix': np.array(matrix)
    }






