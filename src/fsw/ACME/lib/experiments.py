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


def convert_file (raw_file, outdir, label):
    '''
        Convert single raw file to pickle file and save it to output directory
    '''
    ALLOW_CACHED_JSON = False # Reuse raw -> json result for quicker debugging
    working_dir = os.getcwd()
    json_path = os.path.join(outdir, label + ".json")
    if not (ALLOW_CACHED_JSON and os.path.exists(json_path)):
        raw_to_json_exe = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'RawFileReaderWrapper', 'RawFileReaderWrapper', 'bin', 'Release', 'RawFileReaderWrapper.exe')
        try:
            subproc_result = subprocess.run(["mono", raw_to_json_exe, raw_file, outdir, label])
            if subproc_result.returncode != 0:
                print("Failed to convert file: " + raw_file)
                return
        except:
            print("mono dependancy missing. Failed to convert file: " + raw_file)
            return

    with open(json_path) as file:

        data = json.load(file)

        spectrums = []
        axist = []
        axes_m = [] #confusing, needs to be changed
        axism = None
        num_scans = 0
        final = False
        exp_records = {}

        rid = os.path.basename(raw_file).split('.')[0]
        
        for i in range(data['first_spectrum'], data['last_spectrum']):
            tm = data['tms'][i-1]
            axes_m.append(data['positions'][i-1])

            num_scans += 1
            spectrums.append(data['intensities'][i-1])
            axist.append(tm)


        d = data['date']
        d = d[:d.find('T')].split('-')

        date = '{0}/{1}/{2}'.format(d[2], d[1], d[0])

        try:

            lg = logging.getLogger(__name__)
            
            if final: return
            # out of all given m/z axes, get one single axis
            # self.axes_m should contain either 1 m/z axis, or num_scans axes.
            if len(axes_m) < 1:
                raise Exception('Experiment missing m/z axis')
            elif len(axes_m) == 1:
                axism = axes_m[0]
            else:
                axism = []
                for axis in axes_m:
                    axism += axis
                
                # Should now be every m/z value included in all experiment scans
                axism = sorted(list(set(axism)))
                axism = np.array(axism)
            
            lg.debug('axism is ' + str(axism))
            
            axist = np.array(axist)
            
            # map scan numbers to m/z axis values
            dictm = {}
            for i, m in enumerate(axism):
                dictm[m] = i
            
            # fill the experiment matrix
            exp = np.zeros((num_scans, len(axism)))
            if len(axes_m) == num_scans:
                # scans had differing mz axes
                for i, (axis, spec) in enumerate(zip(axes_m, spectrums)):
                    for s, m in zip(spec, axis):
                        # given an intensity s in scan i which was scanned at a certain m/z m
                        # find the index of m on the axis, and place the intensity there
                        try:
                            exp[i, dictm[m]] = s
                        except KeyError as e:
                            logging.error(e)
            else:
                # all scans have the same m/z axis
                for i, spec in enumerate(spectrums):
                    exp[i] = np.array(spec)
            
            exp_records['raw'] = exp
            exp_records['reference'] = exp
            
            dictt = {}
            if len(axist) > 0:
                if len(axist) != num_scans:
                    raise Exception('Time axis mismatched')
                axist = np.array(axist)
            else:
                axist = np.arange(num_scans)
            lg.debug('Time axis for experiment is {0}'.format(axist))
            for i, t in enumerate(axist):
                dictt[t] = i

            time_slice = slice(0, num_scans)
            mass_slice = slice(0, len(axism))
            
            final = True
        
            outdata = dict()
            outdata['matrix'] = exp_records['reference'][time_slice, mass_slice]
            outdata['time_axis'] = axist[time_slice]
            outdata['mass_axis'] = axism[mass_slice]

            outfile_name = os.path.join(outdir, label + '.pickle')
            pickle.dump(outdata, open(outfile_name, 'wb'))
            print('Converted:' + raw_file + ' to ' + outfile_name)

        except Exception as e:
            print(e)
            print('Skipping experiment {0}'.format(rid))
            return None

    os.remove(json_path)

    return outfile_name
 








