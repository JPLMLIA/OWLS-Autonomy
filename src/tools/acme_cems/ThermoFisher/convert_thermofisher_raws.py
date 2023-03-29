# OWLS CE-MS Data Processor for raw ACME files
import os
import os.path as op
import pickle
import glob
import subprocess
import json
import csv
import argparse

from pathlib import Path

import numpy as np


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
        axes_m = []
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
            
            print('axism is ' + str(axism))
            
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
            print('Time axis for experiment is {0}'.format(axist))
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
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir',   required=True,
                                         help='Directory of raw files')

    parser.add_argument('--output_dir',  help='Location for output files.  If not specified output is placed in input folder.')
    
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir

    raw_files = glob.glob(os.path.join(args.input_dir,"*.raw"))
    for raw_file_fullpath in raw_files:

        filename = raw_file_fullpath.split("/")[-1]

        # ThermoFisher MS .raw handling
        print(f"Converting ThermoFisher raw file: {str(filename)}")
        filename = convert_file(str(raw_file_fullpath), args.output_dir, filename.rstrip(".raw"))



